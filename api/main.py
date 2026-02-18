from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import (
    PredictionRequest, PredictionResponse,
    RecommendationRequest, RecommendationResponse, RecommendationItem,
    HealthResponse,
)
import joblib
import pandas as pd
from typing import Dict
import os

from models.matrix_factorization import MatrixFactorization

app = FastAPI(
    title="ML Recommendation Engine API",
    description="Collaborative Filtering Recommendation System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models: Dict = {}
ratings_df: pd.DataFrame | None = None


def load_models():
    """Load trained models and data"""
    global models, ratings_df

    models.clear()

    # User-Based CF
    try:
        models["user_cf"] = joblib.load("saved_models/user_based_cf.pkl")
        print("✓ User-Based CF loaded")
    except Exception as e:
        print("✗ User-Based CF not found:", e)

    # Item-Based CF
    try:
        models["item_cf"] = joblib.load("saved_models/item_based_cf.pkl")
        print("✓ Item-Based CF loaded")
    except Exception as e:
        print("✗ Item-Based CF not found:", e)

    # Matrix Factorization (SVD) via wrapper
    try:
        mf = MatrixFactorization()
        mf.load("saved_models/matrix_factorization.pkl")
        models["svd"] = mf
        print("✓ Matrix Factorization loaded")
    except Exception as e:
        print("✗ Matrix Factorization not found:", e)

    # Load ratings data (for SVD recommendations)
    try:
        ratings_df = pd.read_csv("data/raw/ratings.csv")
        print("✓ Ratings data loaded")
    except Exception as e:
        print("✗ Ratings data not found:", e)


@app.on_event("startup")
async def startup_event():
    print("Loading models...")
    load_models()
    print(f"API Ready! Loaded {len(models)} models")


@app.get("/", response_model=dict)
def root():
    return {
        "name": "ML Recommendation Engine API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Predict rating for user-item pair",
            "/recommend": "Get recommendations for user",
        },
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        models_loaded=list(models.keys()),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_rating(request: PredictionRequest):
    if request.model_type not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_type}' not available. Available: {list(models.keys())}",
        )

    try:
        model = models[request.model_type]
        prediction = model.predict(request.user_id, request.item_id)

        return PredictionResponse(
            user_id=request.user_id,
            item_id=request.item_id,
            predicted_rating=round(float(prediction), 2),
            model_type=request.model_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    if request.model_type not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_type}' not available. Available: {list(models.keys())}",
        )

    try:
        model = models[request.model_type]

        if request.model_type == "svd":
            if ratings_df is None:
                raise HTTPException(status_code=500, detail="Ratings data not loaded")

            all_items = ratings_df["item_id"].unique().tolist()
            user_rated = ratings_df[ratings_df["user_id"] == request.user_id]["item_id"].tolist()

            recommendations = model.recommend(
                request.user_id,
                all_items,
                user_rated,
                n=request.n,
            )
        else:
            recommendations = model.recommend(request.user_id, n=request.n)

        rec_items = [
            RecommendationItem(
                item_id=int(item_id),
                predicted_rating=round(float(score), 2),
            )
            for item_id, score in recommendations
        ]

        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=rec_items,
            model_type=request.model_type,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
