from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional
import numpy as np
import os
import logging
from pathlib import Path

from models.linucb import LinUCBModel
from models.thompson import ThompsonSamplingModel
from models.torch_bandits import DoublyRobustBandit, SlateRankingBandit
from models.data_handler import DataHandler


# Add periodic saving
import asyncio
from fastapi import BackgroundTasks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data handler and models with correct path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "data")

# Create models directory if it doesn't exist
MODELS_DIR = Path("saved_models")
MODELS_DIR.mkdir(exist_ok=True)

logger.info(f"Initializing data handler with path: {DATA_PATH}")
data_handler = DataHandler(DATA_PATH)
data_handler.load_data(sample_fraction=0.05)  # Using small sample for faster loading

context_dim = data_handler.get_feature_dim()
n_arms = 5

models = {
    "linucb": LinUCBModel(context_dim=context_dim, n_arms=n_arms),
    "thompson": ThompsonSamplingModel(context_dim=context_dim, n_arms=n_arms),
    "doubly_robust": DoublyRobustBandit(context_dim=context_dim, n_arms=n_arms),
    "slate_ranking": SlateRankingBandit(context_dim=context_dim, n_arms=n_arms)
}

class RecommendRequest(BaseModel):
    model_type: str
    k: int = 3

class UpdateRequest(BaseModel):
    model_type: str
    chosen_arm: int
    reward: float

# Initialize models with loading from saved state
def initialize_models(context_dim, n_arms):
    models = {
        "linucb": LinUCBModel(context_dim=context_dim, n_arms=n_arms),
        "thompson": ThompsonSamplingModel(context_dim=context_dim, n_arms=n_arms),
        "doubly_robust": DoublyRobustBandit(context_dim=context_dim, n_arms=n_arms),
        "slate_ranking": SlateRankingBandit(context_dim=context_dim, n_arms=n_arms)
    }
    
    # Load saved states if they exist
    for model_name, model in models.items():
        model_path = MODELS_DIR / f"{model_name}.pt"
        if hasattr(model, 'load_model') and model_path.exists():
            try:
                model.load_model(str(model_path))
                logger.info(f"Loaded saved state for {model_name}")
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
    
    return models

@app.get("/")
async def root():
    return {"status": "running", "feature_dim": context_dim}

@app.get("/get_context")
async def get_context():
    """Get a random context and available articles"""
    return data_handler.get_random_context()

@app.post("/recommend")
async def recommend(request: RecommendRequest):
    """Get recommendations from specified model"""
    if request.model_type not in models:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    context = data_handler.get_random_context()
    scaled_features = data_handler.get_scaled_features(
        context["context"],
        context["articles"][0]
    )
    
    model = models[request.model_type]
    top_arms, scores = model.recommend_k(scaled_features, k=request.k)
    
    # Get article details
    article_details = []
    for arm in top_arms:
        article = data_handler.news_df.iloc[arm]
        article_details.append({
            "id": arm,
            "title": article["Title"],
            "abstract": article["Abstract"],
            "url": article["url"]
        })
    
    return {
        "recommendations": top_arms,
        "scores": scores,
        "context": context,
        "article_details": article_details
    }

@app.post("/update")
async def update(request: UpdateRequest):
    """Update specified model with feedback"""
    if request.model_type not in models:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    context = data_handler.get_random_context()
    scaled_features = data_handler.get_scaled_features(
        context["context"],
        context["articles"][0]
    )
    
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

async def save_models_periodically():
    while True:
        for model_name, model in models.items():
            if hasattr(model, 'save_model'):
                try:
                    model.save_model(str(MODELS_DIR / f"{model_name}.pt"))
                    logger.info(f"Saved state for {model_name}")
                except Exception as e:
                    logger.error(f"Failed to save {model_name}: {e}")
        await asyncio.sleep(300)  # Save every 5 minutes

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(save_models_periodically())

@app.post("/reset/{model_type}")
async def reset_model(model_type: str):
    """Reset specified model"""
    if model_type not in models:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    models[model_type].reset()
    
    # Remove saved state if it exists
    model_path = MODELS_DIR / f"{model_type}.pt"
    if model_path.exists():
        model_path.unlink()
    
    return {"status": "success"}