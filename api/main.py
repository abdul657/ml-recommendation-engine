import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)
from models.matrix_factorization import MatrixFactorization

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "saved_models"
DATA_PATH = ROOT / "data" / "raw" / "ratings.csv"

app = FastAPI(
    title="ML Recommendation Engine API",
    description="MovieLens recommendation API using collaborative filtering and matrix factorization.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

models: dict = {}
ratings_df: pd.DataFrame | None = None


def load_models() -> None:
    """Load available serialized recommenders and the catalog interaction data."""
    global ratings_df
    models.clear()

    artifacts = {
        "user_cf": MODEL_DIR / "user_based_cf.pkl",
        "item_cf": MODEL_DIR / "item_based_cf.pkl",
        "svd": MODEL_DIR / "matrix_factorization.pkl",
    }
    for name, filepath in artifacts.items():
        if not filepath.exists():
            logger.warning("Model artifact not found: %s", filepath)
            continue
        try:
            if name == "svd":
                model = MatrixFactorization()
                model.load(filepath)
            else:
                model = joblib.load(filepath)
            models[name] = model
            logger.info("Loaded model %s", name)
        except Exception:
            logger.exception("Could not load model %s from %s", name, filepath)

    if DATA_PATH.exists():
        ratings_df = pd.read_csv(DATA_PATH)
        ratings_df["user_id"] = ratings_df["user_id"].astype(str)
        ratings_df["item_id"] = ratings_df["item_id"].astype(str)
    else:
        ratings_df = None
        logger.warning("Ratings dataset not found: %s", DATA_PATH)


@app.on_event("startup")
async def startup_event() -> None:
    load_models()


@app.get("/")
def root() -> dict:
    return {
        "name": "ML Recommendation Engine API",
        "version": app.version,
        "endpoints": {
            "/health": "Health check",
            "/predict": "POST - Predict rating for a user/item pair",
            "/recommend": "POST - Recommend unrated items for a user",
        },
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy" if models else "degraded",
        models_loaded=sorted(models.keys()),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_rating(request: PredictionRequest) -> PredictionResponse:
    model = models.get(request.model_type)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{request.model_type}' is not loaded. Run python train.py first.",
        )

    try:
        prediction = model.predict(str(request.user_id), str(request.item_id))
        return PredictionResponse(
            user_id=request.user_id,
            item_id=request.item_id,
            predicted_rating=round(float(prediction), 3),
            model_type=request.model_type,
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc


@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    model = models.get(request.model_type)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{request.model_type}' is not loaded. Run python train.py first.",
        )

    user_id = str(request.user_id)
    try:
        if request.model_type == "svd":
            if ratings_df is None:
                raise HTTPException(status_code=503, detail="Ratings dataset is not available.")
            all_items = ratings_df["item_id"].drop_duplicates().tolist()
            rated_items = ratings_df.loc[ratings_df["user_id"] == user_id, "item_id"].tolist()
            recommendations = model.recommend(
                user_id, all_items, rated_items, n=request.n
            )
        else:
            recommendations = model.recommend(user_id, n=request.n)

        items = [
            RecommendationItem(
                item_id=int(item_id),
                predicted_rating=round(float(score), 3),
            )
            for item_id, score in recommendations
        ]
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=items,
            model_type=request.model_type,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Recommendation request failed")
        raise HTTPException(status_code=500, detail="Recommendation request failed.") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
