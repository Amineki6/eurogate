from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RAW_REEFER_COLUMNS = [
    "container_visit_uuid",
    "customer_uuid",
    "container_uuid",
    "HardwareType",
    "EventTime",
    "AvPowerCons",
    "TtlEnergyConsHour",
    "TtlEnergyCons",
    "TemperatureSetPoint",
    "TemperatureAmbient",
    "TemperatureReturn",
    "RemperatureSupply",
    "ContainerSize",
    "stack_tier",
]

TIME_COL = "EventTime"
TARGET_COL = "AvPowerCons"

EXOGENOUS_COLS = [
    "TemperatureSetPoint",
    "TemperatureAmbient",
    "TemperatureReturn",
    "TemperatureSupply",
    "container_count",
    "visit_count",
    "customer_count",
]

LAG_FEATURES = [
    TARGET_COL,
    "TemperatureSetPoint",
    "TemperatureAmbient",
    "TemperatureReturn",
    "TemperatureSupply",
    "container_count",
    "visit_count",
    "customer_count",
]

LAGS = [24, 48, 168]


def read_raw_reefer(path: str | Path) -> pd.DataFrame:
    """Read only the columns needed for hourly preprocessing."""
    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        usecols=lambda c: c in set(RAW_REEFER_COLUMNS),
        low_memory=False,
    )

    if "RemperatureSupply" in df.columns and "TemperatureSupply" not in df.columns:
        df = df.rename(columns={"RemperatureSupply": "TemperatureSupply"})

    # Normalize obvious types early to reduce memory and avoid silent coercion later.
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce").dt.tz_convert(None)

    numeric_cols = [
        TARGET_COL,
        "TtlEnergyConsHour",
        "TtlEnergyCons",
        "TemperatureSetPoint",
        "TemperatureAmbient",
        "TemperatureReturn",
        "TemperatureSupply",
        "stack_tier",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["container_visit_uuid", "customer_uuid", "container_uuid", "HardwareType", "ContainerSize"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df


def clean_raw_reefer(df: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid rows and collapse exact duplicates."""
    df = df.dropna(subset=[TIME_COL]).copy()
    df = df.sort_values([TIME_COL, "container_uuid", "container_visit_uuid"], kind="mergesort")
    df = df.drop_duplicates()

    # Power cannot be negative; temperature spikes are clipped later at the hourly level.
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].clip(lower=0)

    return df


def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate container-level records into a continuous hourly terminal series."""
    agg_map: dict[str, tuple[str, str]] = {
        TARGET_COL: (TARGET_COL, "sum"),
        "container_count": ("container_uuid", "count"),
        "visit_count": ("container_visit_uuid", "nunique"),
        "customer_count": ("customer_uuid", "nunique"),
        "TemperatureSetPoint": ("TemperatureSetPoint", "mean"),
        "TemperatureAmbient": ("TemperatureAmbient", "mean"),
        "TemperatureReturn": ("TemperatureReturn", "mean"),
        "TemperatureSupply": ("TemperatureSupply", "mean"),
        "TemperatureSetPoint_std": ("TemperatureSetPoint", "std"),
        "TemperatureAmbient_std": ("TemperatureAmbient", "std"),
        "TemperatureReturn_std": ("TemperatureReturn", "std"),
        "TemperatureSupply_std": ("TemperatureSupply", "std"),
        "power_per_container": (TARGET_COL, "mean"),
    }

    hourly = df.groupby(TIME_COL, as_index=False).agg(**agg_map)
    hourly = hourly.sort_values(TIME_COL).reset_index(drop=True)

    # Remove obvious sensor glitches at the hourly level without suppressing true peaks.
    temp_cols = [
        "TemperatureSetPoint",
        "TemperatureAmbient",
        "TemperatureReturn",
        "TemperatureSupply",
    ]
    for col in temp_cols:
        if col in hourly.columns and hourly[col].notna().any():
            lower = hourly[col].quantile(0.001)
            upper = hourly[col].quantile(0.999)
            hourly[col] = hourly[col].clip(lower=lower, upper=upper)

    return hourly


def build_timeline(hourly: pd.DataFrame, target_timestamps_path: str | Path | None = None) -> pd.DataFrame:
    """Create a continuous hourly index and extend it to any requested target hours."""
    min_time = hourly[TIME_COL].min()
    max_time = hourly[TIME_COL].max()

    if target_timestamps_path:
        targets = pd.read_csv(target_timestamps_path)
        target_times = pd.to_datetime(targets["timestamp_utc"], utc=True, errors="coerce").dt.tz_convert(None)
        if target_times.notna().any():
            max_time = max(max_time, target_times.max())

    timeline = pd.DataFrame({TIME_COL: pd.date_range(start=min_time, end=max_time, freq="h")})
    full = timeline.merge(hourly, on=TIME_COL, how="left")
    full = full.set_index(TIME_COL)

    # Fill only exogenous hourly series; the target stays NaN for missing hours.
    for col in ["TemperatureSetPoint", "TemperatureAmbient", "TemperatureReturn", "TemperatureSupply"]:
        if col in full.columns:
            full[col] = full[col].interpolate(method="time", limit_direction="both")

    for col in ["container_count", "visit_count", "customer_count", "power_per_container"]:
        if col in full.columns:
            full[col] = full[col].ffill().bfill()

    full = full.reset_index()
    return full


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df[TIME_COL].dt

    df["hour"] = dt.hour.astype("int16")
    df["dayofweek"] = dt.dayofweek.astype("int16")
    df["month"] = dt.month.astype("int16")
    df["dayofyear"] = dt.dayofyear.astype("int16")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")

    # Cyclical encodings help the model understand wrap-around at day and year boundaries.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in LAG_FEATURES:
        if col not in df.columns:
            continue
        for lag in LAGS:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # Rolling statistics use only past observations.
    df["power_roll_mean_24"] = df[TARGET_COL].shift(24).rolling(window=24, min_periods=6).mean()
    df["power_roll_std_24"] = df[TARGET_COL].shift(24).rolling(window=24, min_periods=6).std()
    df["power_roll_mean_168"] = df[TARGET_COL].shift(24).rolling(window=168, min_periods=24).mean()

    df["power_diff_24"] = df[TARGET_COL] - df[f"{TARGET_COL}_lag_24"]
    df["temp_delta_lag_24"] = df["TemperatureAmbient_lag_24"] - df["TemperatureSetPoint_lag_24"]

    return df


def prepare_feature_frame(
    raw_path: str | Path,
    target_timestamps_path: str | Path | None = None,
) -> pd.DataFrame:
    raw = read_raw_reefer(raw_path)
    raw = clean_raw_reefer(raw)
    hourly = aggregate_hourly(raw)
    full = build_timeline(hourly, target_timestamps_path=target_timestamps_path)
    full = add_calendar_features(full)
    full = add_lag_features(full)
    return full


def training_rows(feature_frame: pd.DataFrame) -> pd.DataFrame:
    required = [TARGET_COL, f"{TARGET_COL}_lag_168", "TemperatureAmbient_lag_24", "TemperatureSetPoint_lag_24"]
    train = feature_frame.dropna(subset=required).copy()
    return train


def default_feature_columns() -> list[str]:
    return [
        "hour",
        "dayofweek",
        "month",
        "dayofyear",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
        f"{TARGET_COL}_lag_24",
        f"{TARGET_COL}_lag_48",
        f"{TARGET_COL}_lag_168",
        "TemperatureSetPoint_lag_24",
        "TemperatureAmbient_lag_24",
        "TemperatureReturn_lag_24",
        "TemperatureSupply_lag_24",
        "container_count_lag_24",
        "visit_count_lag_24",
        "customer_count_lag_24",
        "power_roll_mean_24",
        "power_roll_std_24",
        "power_roll_mean_168",
        "power_diff_24",
        "temp_delta_lag_24",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and preprocess reefer training data.")
    parser.add_argument("--input", default="reefer_release.csv", help="Path to the raw reefer CSV.")
    parser.add_argument(
        "--targets",
        default="target_timestamps.csv",
        help="Optional target timestamp file used to extend the feature frame into the forecast horizon.",
    )
    parser.add_argument(
        "--output",
        default="processed_training_data.csv",
        help="Where to write the model-ready training rows.",
    )
    parser.add_argument(
        "--all-features-output",
        default="processed_feature_frame.csv",
        help="Optional path for the full feature frame including future target rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    feature_frame = prepare_feature_frame(args.input, args.targets)
    train_df = training_rows(feature_frame)
    feature_cols = [col for col in default_feature_columns() if col in train_df.columns]

    train_out = train_df[[TIME_COL, TARGET_COL] + feature_cols].copy()
    train_out.to_csv(args.output, index=False)

    # Save the full frame too so the same preprocessing can drive inference if needed.
    full_out = feature_frame.copy()
    full_out.to_csv(args.all_features_output, index=False)

    print(f"Saved training rows to {args.output}")
    print(f"Saved full feature frame to {args.all_features_output}")
    print(f"Training shape: {train_out.shape}")


if __name__ == "__main__":
    main()
