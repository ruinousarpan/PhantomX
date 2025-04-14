from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from datetime import datetime
from core.optimization_engine import OptimizationEngine
from auth.jwt import get_current_user
from models.user import User

router = APIRouter(prefix="/optimization", tags=["optimization"])
optimization_engine = OptimizationEngine()

class OptimizationRequest(BaseModel):
    user_id: str
    activity_type: str
    current_config: Dict[str, Any]
    performance_data: Dict[str, Any]
    behavior_data: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None

class BatchOptimizationRequest(BaseModel):
    user_id: str
    activities: List[Dict[str, Any]]
    global_constraints: Optional[Dict[str, Any]] = None

@router.post("/optimize")
async def optimize_activity(
    request: OptimizationRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Optimize a specific activity configuration
    """
    try:
        result = await optimization_engine.optimize_activity(
            user_id=request.user_id,
            activity_type=request.activity_type,
            current_config=request.current_config,
            performance_data=request.performance_data,
            behavior_data=request.behavior_data,
            constraints=request.constraints
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-optimize")
async def batch_optimize_activities(
    request: BatchOptimizationRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Optimize multiple activities simultaneously
    """
    try:
        results = {}
        for activity in request.activities:
            result = await optimization_engine.optimize_activity(
                user_id=request.user_id,
                activity_type=activity["type"],
                current_config=activity["config"],
                performance_data=activity["performance"],
                behavior_data=activity.get("behavior"),
                constraints=request.global_constraints
            )
            results[activity["type"]] = result
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_optimization_status(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get the current status of the optimization engine
    """
    try:
        return optimization_engine.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/targets")
async def get_optimization_targets(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get available optimization targets and their descriptions
    """
    try:
        return {
            "targets": {
                "efficiency": "Optimize for maximum efficiency",
                "reward_rate": "Optimize for highest reward rate",
                "resource_usage": "Optimize for minimal resource usage",
                "user_satisfaction": "Optimize for user satisfaction metrics"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 