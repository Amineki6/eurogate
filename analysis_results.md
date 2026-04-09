# Exploratory Data Analysis: Reefer Dataset

This document summarizes the key findings from exploring the `reefer_release.csv` container power consumption dataset.

## 1. Dataset Overview
- **Scale**: The dataset is quite large, containing **3,774,557 rows** (hourly logs per container).
- **Columns**: 14 attributes representing hardware specs, set points, power consumption, and thermal properties.
- **Time Range**: The data spans just over a year, from **January 1, 2025, to January 10, 2026** (8,403 unique hours).
- **Data Quality**: Extremely clean. Missing values are minimal: 
  - `customer_uuid`: 11 missing
  - `ContainerSize`: 1,994 missing
  - `stack_tier`: 12,471 missing
  - No missing values in power measurements or timestamps.

## 2. Power Consumption Insights
Since the Hackathon objective is forecasting **combined hourly demand**:
- **Average Hourly Total Load**: 1.06 Megawatts (MW) or 1,066,836 kW.
- **Peak Hourly Total Load**: 2.11 MW (double the average).
- **Minimum Hourly Total Load**: Just 25 kW. 
- **Volatilitiy**: The huge gap between average/minimum and peak load indicates extreme variability, meaning accurate prediction models will be critical.

## 3. Correlation & Driving Factors
- **Ambient Temperature**: Displays the highest correlation (`0.156`) with average power consumption (`AvPowerCons`) among numeric features. Higher outside temperatures logically cause higher power demand for cooling. 
- **Set Points & Return Temps**: Interestingly, `TemperatureSetPoint` (-0.07 mean) has almost 0 linear correlation with power used. This suggests power consumption is less about the absolute target temperature and more about the differential between ambient and set-point, or hardware efficiency.

## 4. Hardware & Container Characteristics
- **Cooling Units**: `SCC6` is heavily dominant (1.56M entries), followed by `ML3` (949k) and `DecosVb` (298k). Hardware type might heavily influence efficiency.
- **Container Sizing**: The yard is dominated by 40-foot containers (3.54M logs) with fewer 20-foot (230k logs) and 45-foot containers.
- **Stack Tiering**: Represents the vertical height of the container in the reefer rack.
  - Tier 1: 1.75M logs
  - Tier 2: 1.20M logs
  - Tier 3: 803k logs
*(Tier might affect ambient temperature due to shade or thermal dynamics within the rack structure).*

## 5. Potential Anomalies to Handle
- **Thermal Extremes**: There are records where `TemperatureAmbient` reaches up to 74.8 °C. These could be due to sensor placement entirely in direct sunlight, or hardware sensor glitches. Data clipping or smoothing might be necessary.
- **Typo in naming**: Note that the feature representing Supply Temperature is misspelled as `RemperatureSupply` in the dataset.

## Recommendations for the Challenge
1. **Feature Engineering**: Creating a `Temp_Delta` feature (`TemperatureAmbient` - `TemperatureSetPoint`) might yield a much higher correlation to power consumption than individual variables.
2. **Weather Integration**: The provided `wetterdaten.zip` (weather data) will be extremely helpful to model out aggregate expected base ambient temperatures since individual container sensors might have direct-sunlight aberrations.
3. **Hardware Embeddings**: Hardware models should be One-Hot encoded or target encoded as their energy profiles will vary drastically.
