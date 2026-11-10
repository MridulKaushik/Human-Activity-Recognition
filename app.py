import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference import HARInference

logger = logging.getLogger("har.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="HAR Activity Classifier", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # fine for a portfolio project; a real product would restrict this
    allow_methods=["*"],
    allow_headers=["*"],
)

inference = HARInference()


# ---------- Request schema ----------
class SensorWindow(BaseModel):
    body_acc_x: list[float] = Field(..., min_length=128, max_length=128)
    body_acc_y: list[float] = Field(..., min_length=128, max_length=128)
    body_acc_z: list[float] = Field(..., min_length=128, max_length=128)
    body_gyro_x: list[float] = Field(..., min_length=128, max_length=128)
    body_gyro_y: list[float] = Field(..., min_length=128, max_length=128)
    body_gyro_z: list[float] = Field(..., min_length=128, max_length=128)
    total_acc_x: list[float] = Field(..., min_length=128, max_length=128)
    total_acc_y: list[float] = Field(..., min_length=128, max_length=128)
    total_acc_z: list[float] = Field(..., min_length=128, max_length=128)


@app.post("/predict")
def predict(data: SensorWindow):
    started = time.perf_counter()
    try:
        result = inference.predict(data.model_dump())
        result["confidence"] = round(result["confidence"], 4)
        result["all_probabilities"] = {
            label: round(probability, 4)
            for label, probability in result["all_probabilities"].items()
        }
        return result
    except (TypeError, ValueError) as error:
        logger.warning("Sensor validation failed: %s", error)
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        logger.info("Inference latency_ms=%.2f", (time.perf_counter() - started) * 1000)


@app.post("/explain")
def explain(data: SensorWindow):
    """Return the most influential features for this prediction."""
    try:
        return inference.explain(data.model_dump())
    except (TypeError, ValueError) as error:
        logger.warning("Explanation validation failed: %s", error)
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "RandomForest",
        "n_features": len(inference.feature_columns),
    }
