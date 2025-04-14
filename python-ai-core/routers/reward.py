from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from datetime import datetime
from core.reward_engine import RewardEngine
from auth.jwt import get_current_user
from models.user import User

router = APIRouter(prefix="/reward", tags=["reward"])
reward_engine = RewardEngine()

class RewardRequest(BaseModel):
    user_id: str
    activity_type: str
    activity_data: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    historical_data: Optional[Dict[str, Any]] = None

class BatchRewardRequest(BaseModel):
    user_id: str
    activities: List[Dict[str, Any]]
    global_context: Optional[Dict[str, Any]] = None

@router.post("/calculate")
async def calculate_rewards(
    request: RewardRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Calculate rewards for a specific activity
    """
    try:
        result = await reward_engine.calculate_rewards(
            user_id=request.user_id,
            activity_type=request.activity_type,
            activity_data=request.activity_data,
            performance_metrics=request.performance_metrics,
            historical_data=request.historical_data
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-calculate")
async def batch_calculate_rewards(
    request: BatchRewardRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Calculate rewards for multiple activities simultaneously
    """
    try:
        results = {}
        for activity in request.activities:
            result = await reward_engine.calculate_rewards(
                user_id=request.user_id,
                activity_type=activity["type"],
                activity_data=activity["data"],
                performance_metrics=activity["metrics"],
                historical_data=activity.get("historical")
            )
            results[activity["type"]] = result
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_reward_status(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get the current status of the reward engine
    """
    try:
        return reward_engine.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/multipliers")
async def get_reward_multipliers(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get available reward multipliers and their descriptions
    """
    try:
        return {
            "multipliers": {
                "efficiency": "Multiplier based on activity efficiency",
                "consistency": "Multiplier based on consistent activity",
                "loyalty": "Multiplier based on user loyalty",
                "activity_specific": "Activity-specific bonus multipliers"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/weights")
async def get_reward_weights(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get reward weights for different activity types
    """
    try:
        return {
            "weights": {
                "mining": {
                    "efficiency": 0.4,
                    "consistency": 0.3,
                    "loyalty": 0.2,
                    "activity_specific": 0.1
                },
                "staking": {
                    "efficiency": 0.3,
                    "consistency": 0.4,
                    "loyalty": 0.2,
                    "activity_specific": 0.1
                },
                "trading": {
                    "efficiency": 0.5,
                    "consistency": 0.2,
                    "loyalty": 0.2,
                    "activity_specific": 0.1
                },
                "referral": {
                    "efficiency": 0.3,
                    "consistency": 0.3,
                    "loyalty": 0.3,
                    "activity_specific": 0.1
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 