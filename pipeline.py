from __future__ import annotations

import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from data_prep.preprocess_training_data import (
    TARGET_COL,
    TIME_COL,
    default_feature_columns,
    prepare_feature_frame,
    training_rows,
)

warnings.filterwarnings("ignore")


def main() -> None:
    print("1. Preparing cleaned hourly feature frame...")
    feature_frame = prepare_feature_frame("reefer_release.csv", "target_timestamps.csv")

    print("2. Selecting train and target rows...")
    train_df = training_rows(feature_frame)
    targets = pd.read_csv("target_timestamps.csv")
    target_times = pd.to_datetime(targets["timestamp_utc"], utc=True, errors="coerce").dt.tz_convert(None)
    targets["EventTime"] = target_times
    test_df = feature_frame[feature_frame[TIME_COL].isin(target_times)].copy()

    features = [col for col in default_feature_columns() if col in train_df.columns]
    X_train = train_df[features]
    y_train = train_df[TARGET_COL]
    X_test = test_df[features]

    print("3. Training point forecast model (MAE)...")
    p80 = y_train.quantile(0.8)
    sample_weights = np.where(y_train >= p80, 2.0, 1.0)

    model_point = lgb.LGBMRegressor(
        objective="mae",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
    )
    model_point.fit(X_train, y_train, sample_weight=sample_weights)
    preds_point = model_point.predict(X_test)

    print("4. Training p90 quantile model...")
    model_p90 = lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.90,
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
    )
    model_p90.fit(X_train, y_train)
    preds_p90 = model_p90.predict(X_test)

    print("5. Post-processing and export...")
    preds_p90_adj = np.maximum(preds_p90, preds_point * 1.05)

    test_df["pred_power_kw"] = preds_point
    test_df["pred_p90_kw"] = preds_p90_adj

    results = targets.merge(
        test_df[[TIME_COL, "pred_power_kw", "pred_p90_kw"]],
        left_on="EventTime",
        right_on=TIME_COL,
        how="left",
    )
    output_df = results[["timestamp_utc", "pred_power_kw", "pred_p90_kw"]]
    output_df.to_csv("submission.csv", index=False)

    joblib.dump(model_point, "model_point.pkl")
    joblib.dump(model_p90, "model_p90.pkl")

    print("Done. Saved submission.csv, model_point.pkl, and model_p90.pkl")


if __name__ == "__main__":
    main()
