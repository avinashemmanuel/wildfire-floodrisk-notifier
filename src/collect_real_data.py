import requests
import pandas as pd
import random

data = []

for i in range(300):

    lat = random.uniform(25, 35)
    lon = random.uniform(70, 90)

    try:
        response = requests.get(f"http://127.0.0.1:8000/risk?lat={lat}&lon={lon}")
        result = response.json()

        if "error" in result:
            continue

        temp = result["weather"]["temperature"]
        wind = result["weather"]["wind_speed"]
        fire_count = result["fire_data"]["fire_count"]
        brightness = result["fire_data"]["avg_brightness"]
        ndvi = result["ndvi"]

        # BETTER LABELING LOGIC
        risk = 1 if (
            (fire_count > 0 and brightness > 320) or
            (ndvi < 0.3 and temp > 30 and wind > 6) or
            (ndvi < 0.2 and temp > 28)
        ) else 0

        # FEATURE ENGINEERING
        dryness = (1 - ndvi) * temp
        fire_intensity = fire_count * brightness

        data.append([
            temp,
            wind,
            fire_count,
            brightness,
            ndvi,
            dryness,
            fire_intensity,
            risk
        ])

        print(f"Collected {i}")

    except Exception as e:
        print("Error:", e)
        continue

# CREATE DATAFRAME
df = pd.DataFrame(data, columns=[
    "temperature",
    "wind_speed",
    "fire_count",
    "brightness",
    "ndvi",
    "dryness",
    "fire_intensity",
    "risk"
])

# shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

# SAVE
df.to_csv("data/processed/balanced_data.csv", index=False)

print("Balanced dataset created!")