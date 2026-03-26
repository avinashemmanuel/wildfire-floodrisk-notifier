## Features

- Real-time weather data (Open-Meteo API)
- Fire hotspot detection (NASA FIRMS)
- Vegetation dryness analysis using NDVI (Sentinel-2)
- Risk prediction API using FastAPI

## Run the project

```bash
uvicorn app.main:app --reload

## ML Model

- Model: XGBoost Classifier
- Features:
  - Temperature
  - Wind Speed
  - Fire Count
  - Fire Brightness
  - NDVI
- Output:
  - Risk Prediction (LOW / HIGH)