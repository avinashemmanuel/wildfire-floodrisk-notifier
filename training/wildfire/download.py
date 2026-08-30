import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "wildfire"
)

load_dotenv(PROJECT_ROOT / ".env")

BASE_URL = (
    "https://firms.modaps.eosdis.nasa.gov/"
    "api/area/csv"
)

SOURCE = "VIIRS_SNPP_SP"

INDIA_BBOX = {
    "west": 68.0,
    "south": 6.0,
    "east": 97.5,
    "north": 37.0,
}

REQUEST_DELAY_SECONDS = 1


def download_firms_data(
    api_key: str,
    date: str,
    output_path: Path,
) -> bool:
    """Download one day of NASA FIRMS data."""

    bbox = (
        f"{INDIA_BBOX['west']},"
        f"{INDIA_BBOX['south']},"
        f"{INDIA_BBOX['east']},"
        f"{INDIA_BBOX['north']}"
    )

    url = (
        f"{BASE_URL}/"
        f"{api_key}/"
        f"{SOURCE}/"
        f"{bbox}/"
        f"1/"
        f"{date}"
    )

    try:
        response = requests.get(
            url,
            timeout=60,
        )

        response.raise_for_status()

        output_path.write_text(
            response.text,
            encoding="utf-8",
        )

        print(f"[OK] {date}")

        return True

    except requests.RequestException as error:
        print(f"[ERROR] {date}: {error}")
        return False


def generate_dates(
    start_date: str,
    end_date: str,
):
    """Generate every date between start and end, inclusive."""

    start = datetime.strptime(
        start_date,
        "%Y-%m-%d",
    ).date()

    end = datetime.strptime(
        end_date,
        "%Y-%m-%d",
    ).date()

    if start > end:
        raise ValueError(
            "Start date must not be after end date."
        )

    current = start

    while current <= end:
        yield current.isoformat()
        current += timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download historical NASA FIRMS "
            "wildfire data."
        )
    )

    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date: YYYY-MM-DD",
    )

    parser.add_argument(
        "--end-date",
        required=True,
        help="End date: YYYY-MM-DD",
    )

    args = parser.parse_args()

    api_key = os.getenv("FIRMS_API_KEY")

    if not api_key:
        raise RuntimeError(
            "FIRMS_API_KEY is not configured."
        )

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dates = list(
        generate_dates(
            args.start_date,
            args.end_date,
        )
    )

    print("=" * 60)
    print("NASA FIRMS HISTORICAL DOWNLOADER")
    print("=" * 60)

    print(f"\nSource: {SOURCE}")
    print(f"Start:  {args.start_date}")
    print(f"End:    {args.end_date}")
    print(f"Days:   {len(dates)}")

    successful = 0
    skipped = 0
    failed = 0

    for date in dates:

        filename = (
            f"firms_{SOURCE.lower()}_{date}.csv"
        )

        output_path = (
            RAW_DATA_DIR / filename
        )

        # Resume support.
        if output_path.exists():
            print(f"[SKIP] {date} already exists")
            skipped += 1
            continue

        success = download_firms_data(
            api_key=api_key,
            date=date,
            output_path=output_path,
        )

        if success:
            successful += 1
        else:
            failed += 1

        time.sleep(REQUEST_DELAY_SECONDS)

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    print(f"Successful: {successful}")
    print(f"Skipped:    {skipped}")
    print(f"Failed:     {failed}")


if __name__ == "__main__":
    main()