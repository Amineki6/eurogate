import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import timedelta
import os
import warnings
warnings.filterwarnings('ignore')

print("1. Loading and Aggregating Reefer Data...")
# Load reefer data
df_reefer = pd.read_csv('reefer_release.csv', sep=';', decimal=',')
df_reefer['EventTime'] = pd.to_datetime(df_reefer['EventTime'])

# Aggregate to hourly
hourly = df_reefer.groupby('EventTime').agg(
    AvPowerCons=('AvPowerCons', 'sum'),
    YardVolume=('container_uuid', 'count'),
    TempAmbient=('TemperatureAmbient', 'mean'),
    TempSetPoint=('TemperatureSetPoint', 'mean')
).reset_index()

hourly = hourly.sort_values('EventTime').reset_index(drop=True)

print("2. Loading Target Timestamps...")
targets = pd.read_csv('target_timestamps.csv')
targets['EventTime'] = pd.to_datetime(targets['timestamp_utc']).dt.tz_localize(None)

# Combine historical hourly indices with future target timestamps
# We need to create a complete timeline to compute rolling features and lags correctly
min_time = hourly['EventTime'].min()
max_time = targets['EventTime'].max()
full_timeline = pd.DataFrame({'EventTime': pd.date_range(start=min_time, end=max_time, freq='H')})

full_data = full_timeline.merge(hourly, on='EventTime', how='left')

print("3. Feature Engineering...")
# Fill missing target metrics using forward fill for Temp/Yard (as a rough estimate for lag processing)
# However, for training we only use non-NaN target AvPowerCons.
full_data['TempAmbient'] = full_data['TempAmbient'].ffill()
full_data['TempSetPoint'] = full_data['TempSetPoint'].ffill()
full_data['YardVolume'] = full_data['YardVolume'].ffill()

# Calendar features
full_data['Hour'] = full_data['EventTime'].dt.hour
full_data['DayOfWeek'] = full_data['EventTime'].dt.dayofweek
full_data['Month'] = full_data['EventTime'].dt.month
full_data['DayOfYear'] = full_data['EventTime'].dt.dayofyear

# Lag features (Important: Since it's a 24h ahead problem, the minimum lag is 24)
full_data['Power_lag_24'] = full_data['AvPowerCons'].shift(24)
full_data['Power_lag_48'] = full_data['AvPowerCons'].shift(48)
full_data['Power_lag_168'] = full_data['AvPowerCons'].shift(168)

full_data['Yard_lag_24'] = full_data['YardVolume'].shift(24)
full_data['Temp_lag_24'] = full_data['TempAmbient'].shift(24)
full_data['Temp_Set_lag_24'] = full_data['TempSetPoint'].shift(24)

# Delta Feature
full_data['Temp_Delta_lag_24'] = full_data['Temp_lag_24'] - full_data['Temp_Set_lag_24']

# Rolling Averages based on 24h lag
full_data['Power_roll_mean_24_to_48'] = full_data['Power_lag_24'].rolling(window=24).mean()

print("4. Preparing Train and Test datasets...")
# Training data: Where we have actual AvPowerCons, and lags are not NaN
train_df = full_data.dropna(subset=['AvPowerCons', 'Power_lag_168'])

# Features to use
features = [
    'Hour', 'DayOfWeek', 'Month', 'DayOfYear',
    'Power_lag_24', 'Power_lag_48', 'Power_lag_168',
    'Yard_lag_24', 'Temp_lag_24', 'Temp_Set_lag_24',
    'Temp_Delta_lag_24', 'Power_roll_mean_24_to_48'
]

X_train = train_df[features]
y_train = train_df['AvPowerCons']

# Test data: The rows specified in the target timestamps
test_df = full_data[full_data['EventTime'].isin(targets['EventTime'])].copy()
X_test = test_df[features]

print("5. Training Point Forecast Model (MAE)...")
# Weighing high-load hours higher to satisfy mae_peak tiebreaker metric
# Compute threshold for top 20%
p80 = y_train.quantile(0.8)
sample_weights = np.where(y_train >= p80, 2.0, 1.0)

model_point = lgb.LGBMRegressor(
    objective='mae',
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)
model_point.fit(X_train, y_train, sample_weight=sample_weights)
preds_point = model_point.predict(X_test)

print("6. Training P90 Risk Estimate Model (Quantile)...")
model_p90 = lgb.LGBMRegressor(
    objective='quantile',
    alpha=0.90,
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)
model_p90.fit(X_train, y_train) # Do not weight quantile to keep raw probability accurate
preds_p90 = model_p90.predict(X_test)

print("7. Post-processing and Export...")
# Ensure Rule 4: pred_p90_kw >= pred_power_kw
preds_p90_adj = np.maximum(preds_p90, preds_point * 1.05)

# Format the results
test_df['pred_power_kw'] = preds_point
test_df['pred_p90_kw'] = preds_p90_adj

# Join back to ensure exact timestamp_utc order
results = targets.merge(test_df[['EventTime', 'pred_power_kw', 'pred_p90_kw']], on='EventTime', how='left')

output_df = results[['timestamp_utc', 'pred_power_kw', 'pred_p90_kw']]
output_df.to_csv('submission.csv', index=False)

print("Done! Predictions saved to submission.csv")
