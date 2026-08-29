from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "wildfire"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "wildfire"

#Initial development bounding box
#These values can be changed later without modifying the pipeline
INDIA_BBOX = {
    "min_lat": 6.0,
    "max_lat": 37.0,
    "min_lon": 68.0,
    "max_lon": 97.5
}

REQUIRED_COLUMNS = [
    "latitude",
    "longitude",
    "bright_ti4",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "bright_ti5",
    "frp",
    "daynight",
    "type",
]

VALID_CONFIDENCE_VALUES = {"l", "n", "h"}

VALID_DAYNIGHT_VALUES = {"D", "N"}

VALID_FIRE_TYPES = {0, 1, 2, 3}
