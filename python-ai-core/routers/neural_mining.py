from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
from core.neural_mining import NeuralMiningEngine
from auth.jwt import get_current_user
from models.user import User

router = APIRouter(prefix="/mining", tags=["mining"])
neural_mining_engine = NeuralMiningEngine()

class MiningSessionRequest(BaseModel):
    user_id: str
    device_type: str
    focus_score: float
    duration: int
    timestamp: datetime
    additional_data: Optional[Dict[str, Any]] = None

class MiningOptimizationRequest(BaseModel):
    user_id: str
    device_type: str
    current_config: Dict[str, Any]
    performance_data: Dict[str, Any]
    optimization_target: str
    constraints: Optional[Dict[str, Any]] = None

@router.post("/analyze")
async def analyze_mining_session(
    request: MiningSessionRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Analyze a mining session and provide insights
    """
    try:
        result = await neural_mining_engine.analyze_mining_session(
            user_id=request.user_id,
            device_type=request.device_type,
            focus_score=request.focus_score,
            duration=request.duration,
            timestamp=request.timestamp,
            additional_data=request.additional_data
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize")
async def optimize_mining(
    request: MiningOptimizationRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Optimize mining configuration based on performance data
    """
    try:
        result = await neural_mining_engine.optimize_mining(
            user_id=request.user_id,
            device_type=request.device_type,
            current_config=request.current_config,
            performance_data=request.performance_data,
            optimization_target=request.optimization_target,
            constraints=request.constraints
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_mining_status(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get the current status of the neural mining engine
    """
    try:
        return neural_mining_engine.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 