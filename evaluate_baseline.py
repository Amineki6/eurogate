import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("1. Loading and Aggregating Reefer Data...")
df_reefer = pd.read_csv('reefer_release.csv', sep=';', decimal=',')
df_reefer['EventTime'] = pd.to_datetime(df_reefer['EventTime'])

hourly = df_reefer.groupby('EventTime').agg(
    AvPowerCons=('AvPowerCons', 'sum'),
    YardVolume=('container_uuid', 'count'),
    TempAmbient=('TemperatureAmbient', 'mean'),
    TempSetPoint=('TemperatureSetPoint', 'mean')
).reset_index()

hourly = hourly.sort_values('EventTime').reset_index(drop=True)

# Generate a continuous timeline
min_time = hourly['EventTime'].min()
max_time = hourly['EventTime'].max()
full_timeline = pd.DataFrame({'EventTime': pd.date_range(start=min_time, end=max_time, freq='H')})
full_data = full_timeline.merge(hourly, on='EventTime', how='left')

print("2. Feature Engineering...")
full_data['TempAmbient'] = full_data['TempAmbient'].ffill()
full_data['TempSetPoint'] = full_data['TempSetPoint'].ffill()
full_data['YardVolume'] = full_data['YardVolume'].ffill()

full_data['Hour'] = full_data['EventTime'].dt.hour
full_data['DayOfWeek'] = full_data['EventTime'].dt.dayofweek
full_data['Month'] = full_data['EventTime'].dt.month
full_data['DayOfYear'] = full_data['EventTime'].dt.dayofyear

full_data['Power_lag_24'] = full_data['AvPowerCons'].shift(24)
full_data['Power_lag_48'] = full_data['AvPowerCons'].shift(48)
full_data['Power_lag_168'] = full_data['AvPowerCons'].shift(168)

full_data['Yard_lag_24'] = full_data['YardVolume'].shift(24)
full_data['Temp_lag_24'] = full_data['TempAmbient'].shift(24)
full_data['Temp_Set_lag_24'] = full_data['TempSetPoint'].shift(24)

full_data['Temp_Delta_lag_24'] = full_data['Temp_lag_24'] - full_data['Temp_Set_lag_24']
full_data['Power_roll_mean_24_to_48'] = full_data['Power_lag_24'].rolling(window=24).mean()

# Drop NaNs that appear due to lagging
full_data = full_data.dropna(subset=['AvPowerCons', 'Power_lag_168'])

print("3. Time-based split for local validation...")
# Let's use the last 30 days of available data as our validation set
split_date = full_data['EventTime'].max() - pd.Timedelta(days=30)
train_df = full_data[full_data['EventTime'] < split_date].copy()
val_df = full_data[full_data['EventTime'] >= split_date].copy()

features = [
    'Hour', 'DayOfWeek', 'Month', 'DayOfYear',
    'Power_lag_24', 'Power_lag_48', 'Power_lag_168',
    'Yard_lag_24', 'Temp_lag_24', 'Temp_Set_lag_24',
    'Temp_Delta_lag_24', 'Power_roll_mean_24_to_48'
]

X_train = train_df[features]
y_train = train_df['AvPowerCons']
X_val = val_df[features]
y_val = val_df['AvPowerCons']

print("4. Training Point Forecast Model (MAE)...")
p80 = y_train.quantile(0.8)
sample_weights = np.where(y_train >= p80, 2.0, 1.0)

model_point = lgb.LGBMRegressor(
    objective='mae',
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)
model_point.fit(X_train, y_train, sample_weight=sample_weights)
preds_point = model_point.predict(X_val)

print("5. Training P90 Risk Estimate Model (Quantile)...")
model_p90 = lgb.LGBMRegressor(
    objective='quantile',
    alpha=0.90,
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)
model_p90.fit(X_train, y_train)
preds_p90 = model_p90.predict(X_val)
preds_p90_adj = np.maximum(preds_p90, preds_point * 1.05)

print("\n--- METRIC EVALUATION ---")
# mae_all
mae_all = np.mean(np.abs(y_val - preds_point))

# mae_peak
# Identify high-load hours in validation set (e.g. >= 80th percentile of training data load, 
# or 80th percentile of validation load. The challenge probably defines high-load. 
# Let's use top 20% of validation load as "high-load")
val_p80 = y_val.quantile(0.8)
high_load_mask = y_val >= val_p80
mae_peak = np.mean(np.abs(y_val[high_load_mask] - preds_point[high_load_mask]))

# pinball_p90 (quantile loss for alpha=0.90)
def pinball_loss(y_true, y_pred, alpha):
    diff = y_true - y_pred
    return np.mean(np.maximum(alpha * diff, (alpha - 1) * diff))

pinball_p90 = pinball_loss(y_val, preds_p90_adj, alpha=0.90)

combined_score = 0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball_p90

print(f"MAE All       : {mae_all:.2f} kW")
print(f"MAE Peak      : {mae_peak:.2f} kW")
print(f"Pinball P90   : {pinball_p90:.2f}")
print(f"Combined Score: {combined_score:.2f}")

