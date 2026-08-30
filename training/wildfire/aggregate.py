import argparse
from pathlib import Path

import pandas as pd


GRID_SIZE_KM = 5.0
TIME_INTERVAL = "6h"


def create_spatial_grid(
    df: pd.DataFrame,
    grid_size_km: float = GRID_SIZE_KM,
) -> pd.DataFrame:
    """
    Assign each fire detection to an approximate 5 km grid cell.

    The approximation uses latitude/longitude degrees.
    At this stage, this is sufficient for our initial dataset.
    A projected CRS-based implementation can be introduced later
    when we perform more advanced geospatial operations.
    """

    # Approximate conversion:
    #
    # 1 degree latitude ≈ 111 km
    #
    # Therefore:
    # grid degrees ≈ grid_size_km / 111

    grid_degrees = grid_size_km / 111.0

    df = df.copy()

    df["grid_lat"] = (
        (df["latitude"] / grid_degrees).round()
        * grid_degrees
    )

    df["grid_lon"] = (
        (df["longitude"] / grid_degrees).round()
        * grid_degrees
    )

    return df


def create_time_bins(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Assign each detection to a six-hour time interval."""

    df = df.copy()

    df["time_bin"] = (
        df["timestamp"]
        .dt.floor(TIME_INTERVAL)
    )

    return df


def aggregate_fire_detections(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate FIRMS detections by spatial and temporal bins."""

    df = create_spatial_grid(df)
    df = create_time_bins(df)

    df["is_high_confidence"] = (
        df["confidence"] == "h"
    ).astype(int)

    df["is_nominal_confidence"] = (
        df["confidence"] == "n"
    ).astype(int)

    df["is_low_confidence"] = (
        df["confidence"] == "l"
    ).astype(int)

    grouped = (
        df.groupby(
            ["grid_lat", "grid_lon", "time_bin"],
            as_index=False,
        )
        .agg(
            fire_count=("frp", "size"),
            mean_brightness=(
                "bright_ti4",
                "mean",
            ),
            max_brightness=(
                "bright_ti4",
                "max",
            ),
            mean_frp=(
                "frp",
                "mean",
            ),
            max_frp=(
                "frp",
                "max",
            ),
            high_confidence_fire_count=(
                "is_high_confidence",
                "sum",
            ),
            nominal_confidence_fire_count=(
                "is_nominal_confidence",
                "sum",
            ),
            low_confidence_fire_count=(
                "is_low_confidence",
                "sum",
            ),
        )
    )

    grouped = grouped.rename(
        columns={
            "grid_lat": "latitude",
            "grid_lon": "longitude",
            "time_bin": "timestamp",
        }
    )

    grouped = grouped.sort_values(
        ["timestamp", "latitude", "longitude"]
    ).reset_index(drop=True)

    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate cleaned NASA FIRMS "
            "wildfire detections."
        )
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the cleaned FIRMS CSV.",
    )

    parser.add_argument(
        "output_file",
        type=Path,
        help="Path for the aggregated CSV.",
    )

    args = parser.parse_args()

    df = pd.read_csv(
        args.input_file,
        parse_dates=["timestamp"],
    )

    print("=" * 60)
    print("WILDFIRE SPATIAL-TEMPORAL AGGREGATION")
    print("=" * 60)

    print(f"\nInput rows: {len(df):,}")

    aggregated = aggregate_fire_detections(df)

    args.output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregated.to_csv(
        args.output_file,
        index=False,
    )

    print(f"Output rows: {len(aggregated):,}")

    print("\nConfiguration:")
    print(f"  Grid size: {GRID_SIZE_KM} km")
    print(f"  Time interval: {TIME_INTERVAL}")

    print("\nOutput columns:")
    for column in aggregated.columns:
        print(f"  - {column}")

    print(f"\nSaved to:")
    print(args.output_file)


if __name__ == "__main__":
    main()