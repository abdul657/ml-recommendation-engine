from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    user_id: int = Field(..., description="User ID")
    item_id: int = Field(..., description="Item ID")
    model_type: str = Field("svd", description="Model type: user_cf, item_cf, or svd")

class PredictionResponse(BaseModel):
    user_id: int
    item_id: int
    predicted_rating: float
    model_type: str

class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User ID")
    n: int = Field(10, description="Number of recommendations")
    model_type: str = Field("svd", description="Model type: user_cf, item_cf, or svd")

class RecommendationItem(BaseModel):
    item_id: int
    predicted_rating: float

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationItem]
    model_type: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
