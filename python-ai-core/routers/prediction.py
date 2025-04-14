from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from datetime import datetime
from core.prediction_engine import PredictionEngine
from auth.jwt import get_current_user
from models.user import User

router = APIRouter(prefix="/prediction", tags=["prediction"])
prediction_engine = PredictionEngine()

class PredictionRequest(BaseModel):
    user_id: str
    activity_type: str
    historical_data: Dict[str, Any]
    current_state: Dict[str, Any]
    prediction_horizon: str  # short-term, medium-term, long-term

class BatchPredictionRequest(BaseModel):
    user_id: str
    predictions: List[Dict[str, Any]]
    global_context: Optional[Dict[str, Any]] = None

@router.post("/predict")
async def predict_activity(
    request: PredictionRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Predict future activity performance
    """
    try:
        result = await prediction_engine.predict_activity(
            user_id=request.user_id,
            activity_type=request.activity_type,
            historical_data=request.historical_data,
            current_state=request.current_state,
            prediction_horizon=request.prediction_horizon
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-predict")
async def batch_predict_activities(
    request: BatchPredictionRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate predictions for multiple activities simultaneously
    """
    try:
        results = {}
        for prediction in request.predictions:
            result = await prediction_engine.predict_activity(
                user_id=request.user_id,
                activity_type=prediction["type"],
                historical_data=prediction["historical_data"],
                current_state=prediction["current_state"],
                prediction_horizon=prediction["horizon"]
            )
            results[prediction["type"]] = result
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_prediction_status(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get the current status of the prediction engine
    """
    try:
        return prediction_engine.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/horizons")
async def get_prediction_horizons(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get available prediction horizons and their descriptions
    """
    try:
        return {
            "horizons": {
                "short-term": "Predictions for the next 24 hours",
                "medium-term": "Predictions for the next 7 days",
                "long-term": "Predictions for the next 30 days"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/confidence")
async def get_prediction_confidence(
    activity_type: str,
    horizon: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get confidence metrics for specific prediction types
    """
    try:
        return {
            "activity_type": activity_type,
            "horizon": horizon,
            "confidence_metrics": {
                "mining": {
                    "short-term": 0.95,
                    "medium-term": 0.85,
                    "long-term": 0.75
                },
                "staking": {
                    "short-term": 0.98,
                    "medium-term": 0.90,
                    "long-term": 0.80
                },
                "trading": {
                    "short-term": 0.85,
                    "medium-term": 0.75,
                    "long-term": 0.65
                },
                "referral": {
                    "short-term": 0.92,
                    "medium-term": 0.82,
                    "long-term": 0.72
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 