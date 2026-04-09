from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_prep.preprocess_training_data import TIME_COL, prepare_feature_frame, training_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a time-based train/validation split for reefer data.")
    parser.add_argument("--input", default="reefer_release.csv", help="Path to the raw reefer CSV.")
    parser.add_argument(
        "--targets",
        default="target_timestamps.csv",
        help="Target timestamp file used to extend the feature frame into the forecast horizon.",
    )
    parser.add_argument(
        "--val-days",
        type=int,
        default=30,
        help="Number of days at the end of the historical series to keep as validation.",
    )
    parser.add_argument(
        "--output-dir",
        default="training_split",
        help="Directory where train/validation CSVs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_frame = prepare_feature_frame(args.input, args.targets)
    model_rows = training_rows(feature_frame).sort_values(TIME_COL).reset_index(drop=True)

    split_date = model_rows[TIME_COL].max() - pd.Timedelta(days=args.val_days)
    train_df = model_rows[model_rows[TIME_COL] < split_date].copy()
    val_df = model_rows[model_rows[TIME_COL] >= split_date].copy()

    train_path = output_dir / "train.csv"
    val_path = output_dir / "validation.csv"
    meta_path = output_dir / "split_metadata.txt"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    with meta_path.open("w", encoding="utf-8") as f:
        f.write(f"split_date={split_date.isoformat()}\n")
        f.write(f"train_rows={len(train_df)}\n")
        f.write(f"validation_rows={len(val_df)}\n")
        f.write(f"validation_days={args.val_days}\n")

    print(f"Saved train split to {train_path}")
    print(f"Saved validation split to {val_path}")
    print(f"Saved split metadata to {meta_path}")


if __name__ == "__main__":
    main()
