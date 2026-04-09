import pandas as pd
import numpy as np
import decimal

print("Loading data...")
# Read csv, specify decimal separator
df = pd.read_csv('reefer_release.csv', sep=';', decimal=',')

print("==== BASIC INFO ====")
print(df.info(memory_usage='deep'))

print("\n==== MISSING VALUES ====")
print(df.isna().sum())

print("\n==== DESCRIPTIVE STATISTICS ====")
print(df.describe().T)

print("\n==== UNIQUE VALUES ====")
for col in ['HardwareType', 'ContainerSize', 'stack_tier']:
    if col in df.columns:
        print(f"\n{col} unique values:")
        print(df[col].value_counts())

print("\n==== TIME RANGE ====")
if 'EventTime' in df.columns:
    df['EventTime'] = pd.to_datetime(df['EventTime'])
    print("Min time:", df['EventTime'].min())
    print("Max time:", df['EventTime'].max())
    
    print("\n==== GROUP BY TIME (Aggregating Power) ====")
    if 'AvPowerCons' in df.columns:
        df_hourly = df.groupby('EventTime')['AvPowerCons'].sum()
        print("Total hourly records:", len(df_hourly))
        print("Average total hourly power:", df_hourly.mean())
        print("Max total hourly power:", df_hourly.max())
        print("Min total hourly power:", df_hourly.min())

print("\n==== CORRELATIONS (Numeric) ====")
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(df[numeric_cols].corr())

# Let's save a summary locally
with open("summary.txt", "w") as f:
    f.write("Dataset loaded and analyzed successfully.")
