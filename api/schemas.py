from typing import Literal

from pydantic import BaseModel, Field


ModelType = Literal["user_cf", "item_cf", "svd"]


class PredictionRequest(BaseModel):
    user_id: int = Field(..., ge=1, description="MovieLens user ID")
    item_id: int = Field(..., ge=1, description="MovieLens item ID")
    model_type: ModelType = Field("svd", description="Model used for rating prediction")


class PredictionResponse(BaseModel):
    user_id: int
    item_id: int
    predicted_rating: float
    model_type: ModelType


class RecommendationRequest(BaseModel):
    user_id: int = Field(..., ge=1, description="MovieLens user ID")
    n: int = Field(10, ge=1, le=50, description="Number of recommendations")
    model_type: ModelType = Field("svd", description="Recommendation model")


class RecommendationItem(BaseModel):
    item_id: int
    predicted_rating: float


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[RecommendationItem]
    model_type: ModelType


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    models_loaded: list[str]
