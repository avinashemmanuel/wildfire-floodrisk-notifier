import argparse
from pathlib import Path

import pandas as pd

from training.wildfire.config import REQUIRED_COLUMNS


def inspect_dataset(file_path: Path) -> None:
    """Inspect a FIRMS wildfire CSV and print a data-quality summary."""

    print("=" * 60)
    print("NASA FIRMS WILDFIRE DATASET INSPECTION")
    print("=" * 60)

    df = pd.read_csv(file_path)

    print(f"\nFile: {file_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print("\nMissing required columns:")
        for column in missing_columns:
            print(f"  - {column}")
    else:
        print("\nRequired columns: OK")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    print(f"\nDuplicate rows: {df.duplicated().sum():,}")

    print("\nDate range:")
    print(f"  {df['acq_date'].min()} → {df['acq_date'].max()}")

    print("\nCoordinate range:")
    print(
        f"  Latitude:  {df['latitude'].min():.6f} → "
        f"{df['latitude'].max():.6f}"
    )
    print(
        f"  Longitude: {df['longitude'].min():.6f} → "
        f"{df['longitude'].max():.6f}"
    )

    print("\nUnique values:")

    for column in [
        "satellite",
        "instrument",
        "confidence",
        "daynight",
        "type",
    ]:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False).to_string())

    print("\nNumerical summary:")
    numerical_columns = [
        "bright_ti4",
        "bright_ti5",
        "scan",
        "track",
        "frp",
    ]

    print(
        df[numerical_columns]
        .describe()
        .round(3)
        .to_string()
    )

    print("\n" + "=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a NASA FIRMS wildfire CSV."
    )

    parser.add_argument(
        "file",
        type=Path,
        help="Path to the FIRMS CSV file.",
    )

    args = parser.parse_args()

    inspect_dataset(args.file)


if __name__ == "__main__":
    main()