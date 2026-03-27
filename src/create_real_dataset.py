import pandas as pd
import random

data = []

for _ in range(100):
    temp = random.uniform(15, 40)
    wind = random.uniform(1, 15)
    fire_count = random.randint(0, 20)
    brightness = random.uniform(300, 400)
    ndvi = random.uniform(0, 1)

    risk = 1 if (
        fire_count > 5 or
        (ndvi < 0.3 and temp > 30 and wind > 5)
    ) else 0

    data.append([temp, wind, fire_count, brightness, ndvi, risk])

df = pd.DataFrame(data, columns=[
    "temperature",
    "wind_speed",
    "fire_count",
    "brightness",
    "ndvi",
    "risk"
])

df.to_csv("data/processed/real_dataset.csv", index=False)

print("Real dataset created!")