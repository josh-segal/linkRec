from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np

from models.linucb import LinUCBModel
from models.thompson import ThompsonSamplingModel
from data.data_handler import DataHandler

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data handler and models
data_handler = DataHandler("./data")  # Adjust path as needed
data_handler.load_data()

context_dim = data_handler.get_feature_dim()
n_arms = 5

models = {
    "linucb": LinUCBModel(context_dim=context_dim, n_arms=n_arms),
    "thompson": ThompsonSamplingModel(context_dim=context_dim, n_arms=n_arms)
}

# Pydantic models for request/response validation
class RecommendRequest(BaseModel):
    model_type: str  # "linucb" or "thompson"
    k: int = 3

class UpdateRequest(BaseModel):
    model_type: str
    chosen_arm: int
    reward: float

class ModelWeights(BaseModel):
    weights: Dict[str, List[float]]
    performance: Dict[str, float]

@app.get("/")
async def root():
    return {"status": "running"}

@app.get("/get_context")
async def get_context():
    """Get a random context and available articles"""
    context = data_handler.get_random_context()
    return {
        "context": context["context"],
        "articles": context["articles"],
        "true_rewards": context["true_rewards"]  # For demonstration
    }

@app.post("/recommend")
async def recommend(request: RecommendRequest):
    """Get recommendations from specified model"""
    if request.model_type not in models:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    # Get current context
    context = data_handler.get_random_context()
    scaled_features = data_handler.get_scaled_features(
        context["context"],
        context["articles"][0]
    )
    
    # Get recommendations
    model = models[request.model_type]
    top_arms, scores = model.recommend_k(scaled_features, k=request.k)
    
    return {
        "recommendations": top_arms,
        "scores": scores,
        "context": context
    }

@app.post("/update")
async def update(request: UpdateRequest):
    """Update specified model with feedback"""
    if request.model_type not in models:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    # Get current context
    context = data_handler.get_random_context()
    scaled_features = data_handler.get_scaled_features(
        context["context"],
        context["articles"][0]
    )
    
    # Update model
    model = models[request.model_type]
    model.update(
        context=scaled_features,
        chosen_arm=request.chosen_arm,
        reward=request.reward
    )
    
    return {"status": "success"}

@app.get("/model_weights/{model_type}")
async def get_weights(model_type: str):
    """Get current model weights for visualization"""
    if model_type not in models:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    model = models[model_type]
    weights = {
        str(i): model.get_arm_weights(i).tolist() 
        for i in range(model.n_arms)
    }
    
    stats = model.get_performance_stats()
    
    return {
        "weights": weights,
        "performance": stats
    }

@app.post("/reset/{model_type}")
async def reset_model(model_type: str):
    """Reset specified model"""
    if model_type not in models:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    models[model_type].reset()
    return {"status": "success"}