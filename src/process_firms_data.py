import pandas as pd
import random

df = pd.read_csv("data/raw/SUOMI_VIIRS_C2_Global_7d.csv")

df = df[[
    "latitude",
    "longitude",
    "bright_ti4",
    "bright_ti5",
    "frp",
    "confidence",
    "daynight"
]]

df = df.dropna()

confidence_map = {
    "low": 0,
    "nominal": 1,
    "high": 2
}

df["confidence"] = df["confidence"].map(confidence_map)

df["daynight"] = df["daynight"].map({"D": 1, "N": 0})

df["risk"] = 1

low_data = []

for _ in range(len(df)):
    lat = random.uniform(25, 35)
    lon = random.uniform(70, 90)

    low_data.append({
        "latitude": lat,
        "longitude": lon,
        "bright_ti4": random.uniform(290, 310),
        "bright_ti5": random.uniform(280, 300),
        "frp": random.uniform(0, 5),
        "confidence": random.choice([0, 1]),
        "daynight": random.choice([0, 1]),
        "risk": 0
    })

low_df = pd.DataFrame(low_data)

final_df = pd.concat([df, low_df])

final_df = final_df.sample(frac=1).reset_index(drop=True)

final_df.to_csv("data/processed/firms_dataset.csv", index=False)

print("Dataset Ready!")