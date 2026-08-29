import argparse
import json
from pathlib import Path

import pandas as pd

from training.wildfire.config import (
    PROCESSED_DATA_DIR,
    REQUIRED_COLUMNS,
    VALID_CONFIDENCE_VALUES,
    VALID_DAYNIGHT_VALUES,
    VALID_FIRE_TYPES,
)


def validate_columns(df: pd.DataFrame) -> None:
    """Ensure all required FIRMS columns are present."""

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def parse_timestamp(df: pd.DataFrame) -> pd.Series:
    """Create a UTC timestamp from FIRMS acquisition date and time."""

    dates = pd.to_datetime(
        df["acq_date"],
        errors="coerce",
        utc=True,
    )

    times = pd.to_numeric(
        df["acq_time"],
        errors="coerce",
    )

    hours = times // 100
    minutes = times % 100

    valid_time = (
        times.notna()
        & times.between(0, 2359)
        & hours.between(0, 23)
        & minutes.between(0, 59)
    )

    timestamp = (
        dates.dt.normalize()
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(minutes, unit="m")
    )

    timestamp = timestamp.where(
        dates.notna() & valid_time
    )

    return timestamp


def clean_dataset(
    input_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Clean and validate a NASA FIRMS wildfire dataset."""

    df = pd.read_csv(input_path)

    raw_rows = len(df)

    validate_columns(df)

    report = {
        "input_file": str(input_path),
        "raw_rows": raw_rows,
        "missing_values_before_cleaning": (
            df.isna().sum().to_dict()
        ),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    # ---------------------------------------------------------
    # Remove exact duplicate observations
    # ---------------------------------------------------------

    df = df.drop_duplicates().copy()

    # ---------------------------------------------------------
    # Validate coordinates
    # ---------------------------------------------------------

    valid_coordinates = (
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    )

    invalid_coordinates = int((~valid_coordinates).sum())

    df = df.loc[valid_coordinates].copy()

    # ---------------------------------------------------------
    # Parse timestamp
    # ---------------------------------------------------------

    df["timestamp"] = parse_timestamp(df)

    invalid_timestamps = int(
        df["timestamp"].isna().sum()
    )

    df = df.loc[
        df["timestamp"].notna()
    ].copy()

    # ---------------------------------------------------------
    # Validate numerical columns
    # ---------------------------------------------------------

    numerical_columns = [
        "bright_ti4",
        "bright_ti5",
        "scan",
        "track",
        "frp",
    ]

    for column in numerical_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    invalid_numerical = (
        df[numerical_columns]
        .isna()
        .any(axis=1)
    )

    invalid_numerical_rows = int(
        invalid_numerical.sum()
    )

    df = df.loc[~invalid_numerical].copy()

    # ---------------------------------------------------------
    # FRP validation
    # ---------------------------------------------------------

    negative_frp = int(
        (df["frp"] < 0).sum()
    )

    df = df.loc[df["frp"] >= 0].copy()

    # ---------------------------------------------------------
    # Validate categorical columns
    # ---------------------------------------------------------

    valid_confidence = df["confidence"].isin(
        VALID_CONFIDENCE_VALUES
    )

    valid_daynight = df["daynight"].isin(
        VALID_DAYNIGHT_VALUES
    )

    valid_type = df["type"].isin(
        VALID_FIRE_TYPES
    )

    invalid_confidence = int(
        (~valid_confidence).sum()
    )

    invalid_daynight = int(
        (~valid_daynight).sum()
    )

    invalid_type = int(
        (~valid_type).sum()
    )

    valid_categories = (
        valid_confidence
        & valid_daynight
        & valid_type
    )

    df = df.loc[valid_categories].copy()

    # ---------------------------------------------------------
    # Derived temporal features
    # ---------------------------------------------------------

    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear

    # ---------------------------------------------------------
    # Final ordering
    # ---------------------------------------------------------

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    report.update(
        {
            "invalid_coordinates_removed": invalid_coordinates,
            "invalid_timestamps_removed": invalid_timestamps,
            "invalid_numerical_rows_removed": (
                invalid_numerical_rows
            ),
            "negative_frp_removed": negative_frp,
            "invalid_confidence_removed": (
                invalid_confidence
            ),
            "invalid_daynight_removed": (
                invalid_daynight
            ),
            "invalid_type_removed": invalid_type,
            "final_rows": len(df),
            "rows_removed": (
                raw_rows - len(df)
            ),
            "date_range": {
                "start": str(df["timestamp"].min()),
                "end": str(df["timestamp"].max()),
            },
            "latitude_range": {
                "min": float(df["latitude"].min()),
                "max": float(df["latitude"].max()),
            },
            "longitude_range": {
                "min": float(df["longitude"].min()),
                "max": float(df["longitude"].max()),
            },
            "missing_values_after_cleaning": (
                df.isna().sum().to_dict()
            ),
        }
    )

    return df, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean a NASA FIRMS wildfire CSV."
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the raw FIRMS CSV.",
    )

    args = parser.parse_args()

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned_df, report = clean_dataset(
        args.input_file
    )

    output_file = (
        PROCESSED_DATA_DIR
        / f"{args.input_file.stem}_cleaned.csv"
    )

    report_file = (
        PROCESSED_DATA_DIR
        / f"{args.input_file.stem}_quality_report.json"
    )

    cleaned_df.to_csv(
        output_file,
        index=False,
    )

    with report_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            default=str,
        )

    print("=" * 60)
    print("WILDFIRE DATA CLEANING COMPLETE")
    print("=" * 60)

    print(f"\nInput rows:  {report['raw_rows']:,}")
    print(f"Output rows: {report['final_rows']:,}")
    print(f"Removed:     {report['rows_removed']:,}")

    print(f"\nCleaned dataset:")
    print(output_file)

    print(f"\nQuality report:")
    print(report_file)


if __name__ == "__main__":
    main()