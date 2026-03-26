import pandas as pd
import random

data = []

for _ in range(200):
    temp = random.uniform(20, 45)
    wind = random.uniform(1, 25)
    fire_count = random.randint(0, 20)
    brightness = random.uniform(280, 350)
    ndvi = random.uniform(0, 0.8)

    #rule-based label temp
    if fire_count > 12 or (ndvi < 0.2 and temp > 35 and wind > 10):
        risk = 1
    elif ndvi < 0.25 and temp > 30:
        risk = 1
    else:
        risk = 0

    data.append([temp, wind, fire_count, brightness, ndvi, risk])

df = pd.DataFrame(data, columns=[
    "temperature", "wind_speed", "fire_count",
    "brightness", "ndvi", "risk"
])
df.to_csv("data/processed/dataset.csv", index=False)

print("Dataset created!")