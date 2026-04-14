from fastapi import FastAPI
import requests
import random
import rasterio
from rasterio.io import MemoryFile
import os
from dotenv import load_dotenv
import joblib
import shap
import pandas as pd
import time

model = joblib.load("model.pkl")
explainer = shap.TreeExplainer(model)

load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
TOKEN = None
TOKEN_EXPIRY = 0
app = FastAPI()


def get_ndvi(lat, lon):
    token = get_access_token()

    if not token:
        return None

    url = "https://sh.dataspace.copernicus.eu/api/v1/process"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": {
            "bounds": {
                "bbox": [lon-0.001, lat-0.001, lon+0.001, lat+0.001]
            },
            "data": [
                {
                    "type": "sentinel-2-l2a"
                }
            ]
        },
        "output": {
            "width": 1,
            "height": 1,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/tiff"
                    }
                }
            ]
        },
        "evalscript": """
        //VERSION=3
        function setup() {
            return {
                input: ["B04", "B08"],
                output: {
                    bands: 1,
                    sampleType: "FLOAT32"
                }
            };
        }

        function evaluatePixel(sample) {
            let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
            return [ndvi];
        }
        """
    }

    response = requests.post(url, headers=headers, json=payload)

    print("NDVI STATUS:", response.status_code)

    if response.status_code != 200:
        print("NDVI ERROR:", response.text)
        return None

    with MemoryFile(response.content) as memfile:
        with memfile.open() as dataset:
            ndvi_array = dataset.read(1)
            ndvi_value = float(ndvi_array[0][0])

    print("NDVI Value:", ndvi_value)
    return round(ndvi_value, 3)
        

def get_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True
    }

    response = requests.get(url, params=params)
    data = response.json()

    weather = data.get("current_weather", {})

    return {
        "temperature": weather.get("temperature"),
        "wind_speed": weather.get("windspeed")
    }


def get_fire_data(lat, lon):
    API_KEY = "c1849540e1c71b261bff097ad70101e5"

    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{API_KEY}/VIIRS_SNPP_NRT/{lat-1},{lon-1},{lat+1},{lon+1}/1"

    try:
        response = requests.get(url, timeout=5)

        if response.status_code != 200:
            print("FIRMS error:", response.status_code)
            return []

        lines = response.text.split("\n")
        data = lines[1:]

        fires = []

        for line in data:
            if line.strip() == "":
                continue

            parts = line.split(",")

            fire = {
                "lat": float(parts[0]),
                "lon": float(parts[1]),
                "brightness": float(parts[2]),
                "confidence": parts[8]
            }

            fires.append(fire)

        return fires

    except Exception as e:
        print("Error fetching fire data:", e)
        return []


def process_fire_data(fires):
    if len(fires) == 0:
        return {
            "fire_count": 0,
            "avg_brightness": 0
        }
    
    count = len(fires)
    avg_brightness = sum(f["brightness"] for f in fires) / count

    return {
        "fire_count": count,
        "avg_brightness": avg_brightness
    }


def get_access_token():
    global TOKEN, TOKEN_EXPIRY

    if TOKEN and time.time() < TOKEN_EXPIRY:
        return TOKEN
    
    print("Fetching new token...")

    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

    response = requests.post(
        url,
        auth=(CLIENT_ID, CLIENT_SECRET),
        data={"grant_type": "client_credentials"}
    )

    if response.status_code != 200:
        print("Token error: ", response.text)
        return None
    
    data = response.json()
    
    TOKEN = data["access_token"]

    TOKEN_EXPIRY = time.time() + data["expires_in"] - 60

    return TOKEN


def explain_prediction(temp, wind, fire_count, brightness, ndvi):
    dryness = (1 - ndvi) * temp

    data = pd.DataFrame([{
        "temperature": temp,
        "wind_speed": wind,
        "fire_count": fire_count,
        "brightness": brightness,
        "ndvi": ndvi,
        "dryness": dryness
    }])

    shap_values = explainer(data)

    explanation = {}

    for i, col in enumerate(data.columns):
        explanation[col] = round(float(shap_values.values[0][i]), 3)

    return explanation


@app.get("/")
def home():
    return {"message": "Wildfire Flood Risk Notifier API is running!"}


@app.get("/risk")
def get_risk(lat: float = 30.3165, lon: float = 78.0322):
    print("Fetching weather...")
    weather = get_weather(lat, lon)

    print("Fething fire data...")
    fires = get_fire_data(lat, lon)
    fire_stats = process_fire_data(fires)

    print("Fetching NDVI...")
    ndvi = get_ndvi(lat, lon)

    temp = weather["temperature"]
    wind = weather["wind_speed"]
    fire_count = fire_stats["fire_count"]

    if temp is None or wind is None or ndvi is None:
        return {"error": "Data not available"}
    
    dryness = (1 - ndvi) * temp

    features = [[
        temp,
        wind,
        fire_count,
        fire_stats["avg_brightness"],
        ndvi,
        dryness
    ]]

    prediction = model.predict(features)[0]

    risk = "HIGH" if prediction == 1 else "LOW"

    explanation = explain_prediction(
        temp,
        wind, 
        fire_count,
        fire_stats["avg_brightness"],
        ndvi
    )

    return {
        "location": {"lat": lat, "lon": lon},
        "weather": weather,
        "fire_data": fire_stats,
        "ndvi": ndvi,
        "risk": risk,
        "dryness": round(dryness, 3),
        "explanation": explanation
    }


if __name__ == "__main__":
    ndvi = get_ndvi(30.3165, 78.0322)
    print("NDVI:", ndvi)