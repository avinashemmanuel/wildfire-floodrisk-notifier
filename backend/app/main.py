from fastapi import FastAPI


app = FastAPI(
    title="Wildfire & Flood Risk Notifier",
    description="API for wildfire and flood risk prediction.",
    version="0.1.0",
)


@app.get("/")
async def root() -> dict[str, str]:
    """Return basic information about the API."""
    return {
        "message": "Wildfire & Flood Risk Notifier API",
        "version": "0.1.0",
    }


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Return the health status of the API."""
    return {"status": "healthy"}