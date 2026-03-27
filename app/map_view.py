import folium
from folium.plugins import HeatMap
import requests

# multiple locations (grid)
locations = [
    (30.30, 78.00),
    (30.31, 78.01),
    (30.32, 78.02),
    (30.33, 78.03),
    (30.34, 78.04),
    (30.35, 78.05),
    (30.36, 78.06),
]

# create map
m = folium.Map(
    location=[30.3165, 78.0322],
    zoom_start=10,
    tiles="CartoDB positron"
)

for lat, lon in locations:
    response = requests.get(f"http://127.0.0.1:8000/risk?lat={lat}&lon={lon}")
    data = response.json()

    risk = data["risk"]

    if risk == "HIGH":
        color = "red"
    elif risk == "MEDIUM":
        color = "orange"
    else:
        color = "green"

    folium.CircleMarker(
        location=[lat, lon],
        radius=8,
        color=color,
        fill=True,
        fill_color=color,
        popup=f"""
            Risk: {risk}<br>
            Temp: {data['weather']['temperature']}°C<br>
            NDVI: {data['ndvi']}<br>
            Fire: {data['fire_data']['fire_count']}
        """
    ).add_to(m)

heat_data = []

for lat, lon in locations:
    response = requests.get(f"http://127.0.0.1:8000/risk?lat={lat}&lon={lon}")
    data = response.json()

    risk = data["risk"]

    # convert risk to numeric
    if risk == "HIGH":
        value = 1
    elif risk == "MEDIUM":
        value = 0.6
    else:
        value = 0.2

    heat_data.append([lat, lon, value])

HeatMap(heat_data).add_to(m)

# save map
m.save("map.html")

print("Map created!")