import requests
import pandas as pd
import random

data = []

#sample multiple locations around uttarakhand
for _ in range(50):
    lat = random.uniform(29.5, 31.5)
    lon = random.uniform(77.5, 79.5)

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

        #temporary label
        risk = 1 if result["risk"] == "HIGH" else 0

        data.append([temp, wind, fire_count, brightness, ndvi, risk])

        print(f"Collected: {lat}, {lon}")

    except:
        continue

df = pd.DataFrame(data, columns=[
    "temperature",
    "wind_speed",
    "fire_count",
    "brightness",
    "ndvi",
    "risk"
])

df.to_csv("data/processed/real_collected.csv", index=False)

print("Real collected dataset created!")