from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from datetime import datetime
from core.risk_engine import RiskEngine
from auth.jwt import get_current_user
from models.user import User

router = APIRouter(prefix="/risk", tags=["risk"])
risk_engine = RiskEngine()

class RiskAssessmentRequest(BaseModel):
    user_id: str
    activity_type: str
    activity_data: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    historical_data: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None

class BatchRiskAssessmentRequest(BaseModel):
    user_id: str
    activities: List[Dict[str, Any]]
    global_context: Optional[Dict[str, Any]] = None

@router.post("/assess")
async def assess_risk(
    request: RiskAssessmentRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Assess risk for a specific activity
    """
    try:
        result = await risk_engine.assess_risk(
            user_id=request.user_id,
            activity_type=request.activity_type,
            activity_data=request.activity_data,
            performance_metrics=request.performance_metrics,
            historical_data=request.historical_data,
            context=request.context
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-assess")
async def batch_assess_risks(
    request: BatchRiskAssessmentRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Assess risks for multiple activities simultaneously
    """
    try:
        results = {}
        for activity in request.activities:
            result = await risk_engine.assess_risk(
                user_id=request.user_id,
                activity_type=activity["type"],
                activity_data=activity["data"],
                performance_metrics=activity["metrics"],
                historical_data=activity.get("historical"),
                context=request.global_context
            )
            results[activity["type"]] = result
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_risk_status(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get the current status of the risk engine
    """
    try:
        return risk_engine.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/factors")
async def get_risk_factors(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get available risk factors and their descriptions
    """
    try:
        return {
            "factors": {
                "mining": {
                    "hardware_failure": "Risk of mining hardware failure",
                    "power_consumption": "Risk related to power usage and costs",
                    "network_issues": "Risk of network connectivity problems",
                    "market_volatility": "Risk from cryptocurrency price fluctuations"
                },
                "staking": {
                    "validator_failure": "Risk of validator node failure",
                    "slashing": "Risk of stake slashing due to misbehavior",
                    "lockup_period": "Risk from staking lockup periods",
                    "market_volatility": "Risk from cryptocurrency price fluctuations"
                },
                "trading": {
                    "market_volatility": "Risk from price volatility",
                    "liquidity": "Risk from insufficient market liquidity",
                    "execution_slippage": "Risk from trade execution slippage",
                    "counterparty": "Risk from trading counterparty issues"
                },
                "referral": {
                    "fraud": "Risk from fraudulent referrals",
                    "attrition": "Risk from referred user churn",
                    "compliance": "Risk from regulatory compliance issues",
                    "market_conditions": "Risk from changing market conditions"
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/thresholds")
async def get_risk_thresholds(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get risk thresholds for different activity types
    """
    try:
        return {
            "thresholds": {
                "mining": {
                    "low": 0.3,
                    "medium": 0.6,
                    "high": 0.8
                },
                "staking": {
                    "low": 0.2,
                    "medium": 0.5,
                    "high": 0.7
                },
                "trading": {
                    "low": 0.4,
                    "medium": 0.7,
                    "high": 0.9
                },
                "referral": {
                    "low": 0.3,
                    "medium": 0.6,
                    "high": 0.8
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/mitigation")
async def get_mitigation_strategies(
    activity_type: str,
    risk_level: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get risk mitigation strategies for specific activity types and risk levels
    """
    try:
        return {
            "strategies": {
                "mining": {
                    "low": [
                        "Regular hardware maintenance",
                        "Power usage optimization",
                        "Network redundancy setup"
                    ],
                    "medium": [
                        "Hardware upgrade consideration",
                        "Power backup systems",
                        "Multiple mining pools"
                    ],
                    "high": [
                        "Immediate hardware replacement",
                        "Emergency power systems",
                        "Geographic distribution"
                    ]
                },
                "staking": {
                    "low": [
                        "Regular validator monitoring",
                        "Basic security measures",
                        "Stake diversification"
                    ],
                    "medium": [
                        "Advanced monitoring systems",
                        "Enhanced security protocols",
                        "Multiple validator setup"
                    ],
                    "high": [
                        "24/7 monitoring",
                        "Maximum security measures",
                        "Emergency unstaking plan"
                    ]
                },
                "trading": {
                    "low": [
                        "Basic risk management",
                        "Stop-loss orders",
                        "Position sizing"
                    ],
                    "medium": [
                        "Advanced risk management",
                        "Multiple stop-loss levels",
                        "Portfolio diversification"
                    ],
                    "high": [
                        "Maximum risk controls",
                        "Emergency exit strategy",
                        "Complete position closure"
                    ]
                },
                "referral": {
                    "low": [
                        "Basic fraud detection",
                        "Regular monitoring",
                        "Standard verification"
                    ],
                    "medium": [
                        "Advanced fraud detection",
                        "Enhanced monitoring",
                        "Strict verification"
                    ],
                    "high": [
                        "Maximum fraud prevention",
                        "Real-time monitoring",
                        "Multi-factor verification"
                    ]
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 