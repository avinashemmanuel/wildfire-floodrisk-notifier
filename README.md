# Wildfire & Flood Risk Notifier

A full-stack machine learning system designed to predict and monitor
wildfire and flood risk for selected geographical locations.

## Project Overview

The system will eventually combine historical environmental data,
real-time environmental observations, machine learning models, and
geographical visualization to estimate wildfire and flood risk.

The project is being developed incrementally, with offline model
training separated from real-time inference.

## Features

### Planned

- Location-based wildfire risk prediction
- Location-based flood risk prediction
- Real-time environmental data ingestion
- Historical data processing
- Machine learning model training
- Risk probability and risk-level estimation
- Prediction explainability using SHAP
- Interactive geographical visualization
- Prediction history
- Monitored locations
- Configurable alerts and notifications

### Currently Implemented

- Project repository
- Python virtual environment
- Initial FastAPI backend
- Health-check endpoint
- React + TypeScript + Vite frontend
- Initial project structure
- Environment configuration template
- Git configuration
- Basic Docker configuration

## Architecture

The project separates the following major components:

```text
Historical Data
      |
      v
Preprocessing
      |
      v
Feature Engineering
      |
      v
ML Training
      |
      v
Saved Models
      |
      +----------------------+
                             |
Real-Time Data ---> Feature Engineering
                             |
                             v
                       ML Prediction
                             |
                             v
                       FastAPI Backend
                             |
                             v
                    React Frontend