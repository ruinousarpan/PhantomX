import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from typing import Dict, Any

from routers.optimization import router
from database.models import ActivityType

@pytest.fixture
def mining_optimization_request() -> Dict[str, Any]:
    """Create a test mining optimization request"""
    return {
        "user_id": "test_user",
        "activity_type": ActivityType.MINING,
        "current_config": {
            "hash_rate": 95.0,
            "power_limit": 450.0,
            "temperature_limit": 80.0,
            "fan_speed": 70.0
        },
        "performance_metrics": {
            "hash_rate": 95.0,
            "power_efficiency": 0.8,
            "temperature": 78.0,
            "overall_efficiency": 0.82
        },
        "historical_data": [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "hash_rate": 90.0,
                "power_usage": 440.0,
                "temperature": 74.0,
                "efficiency": 0.8,
                "duration": 3600
            }
        ]
    }

@pytest.fixture
def staking_optimization_request() -> Dict[str, Any]:
    """Create a test staking optimization request"""
    return {
        "user_id": "test_user",
        "activity_type": ActivityType.STAKING,
        "current_config": {
            "stake_amount": 1000.0,
            "lock_period": 30,
            "rewards_rate": 0.05
        },
        "performance_metrics": {
            "current_rewards": 45.0,
            "uptime": 0.98,
            "network_health": 0.95
        },
        "historical_data": [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "stake_amount": 900.0,
                "rewards": 40.0,
                "uptime": 0.97,
                "network_health": 0.94,
                "duration": 86400
            }
        ]
    }

@pytest.fixture
def trading_optimization_request() -> Dict[str, Any]:
    """Create a test trading optimization request"""
    return {
        "user_id": "test_user",
        "activity_type": ActivityType.TRADING,
        "current_config": {
            "trading_pairs": ["BTC/USD", "ETH/USD"],
            "position_size": 0.1,
            "leverage": 1.0
        },
        "performance_metrics": {
            "win_rate": 0.65,
            "profit_factor": 1.8,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.15
        },
        "historical_data": [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "win_rate": 0.62,
                "profit_factor": 1.7,
                "sharpe_ratio": 1.4,
                "max_drawdown": 0.16,
                "duration": 86400
            }
        ]
    }

@pytest.mark.asyncio
async def test_optimize_mining_activity(
    client: TestClient,
    test_user_token: str,
    mining_optimization_request: Dict[str, Any]
):
    """Test mining activity optimization endpoint"""
    # Make request
    response = client.post(
        "/optimization/optimize",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json=mining_optimization_request
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Check result structure
    assert "optimized_config" in data
    assert "expected_improvements" in data
    assert "recommendations" in data
    assert "risk_assessment" in data
    
    # Check optimized configuration
    config = data["optimized_config"]
    assert "hash_rate" in config
    assert "power_limit" in config
    assert "temperature_limit" in config
    assert "fan_speed" in config
    
    # Check expected improvements
    improvements = data["expected_improvements"]
    assert "hash_rate" in improvements
    assert "power_efficiency" in improvements
    assert "temperature" in improvements
    assert "overall_efficiency" in improvements
    
    # Check recommendations
    recommendations = data["recommendations"]
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    
    # Check risk assessment
    risk = data["risk_assessment"]
    assert "risk_score" in risk
    assert "risk_factors" in risk
    assert "mitigation_strategies" in risk

@pytest.mark.asyncio
async def test_optimize_staking_activity(
    client: TestClient,
    test_user_token: str,
    staking_optimization_request: Dict[str, Any]
):
    """Test staking activity optimization endpoint"""
    # Make request
    response = client.post(
        "/optimization/optimize",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json=staking_optimization_request
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Check result structure
    assert "optimized_config" in data
    assert "expected_improvements" in data
    assert "recommendations" in data
    assert "risk_assessment" in data
    
    # Check optimized configuration
    config = data["optimized_config"]
    assert "stake_amount" in config
    assert "lock_period" in config
    assert "rewards_rate" in config
    
    # Check expected improvements
    improvements = data["expected_improvements"]
    assert "rewards" in improvements
    assert "uptime" in improvements
    assert "network_health" in improvements
    
    # Check recommendations
    recommendations = data["recommendations"]
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    
    # Check risk assessment
    risk = data["risk_assessment"]
    assert "risk_score" in risk
    assert "risk_factors" in risk
    assert "mitigation_strategies" in risk

@pytest.mark.asyncio
async def test_optimize_trading_activity(
    client: TestClient,
    test_user_token: str,
    trading_optimization_request: Dict[str, Any]
):
    """Test trading activity optimization endpoint"""
    # Make request
    response = client.post(
        "/optimization/optimize",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json=trading_optimization_request
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Check result structure
    assert "optimized_config" in data
    assert "expected_improvements" in data
    assert "recommendations" in data
    assert "risk_assessment" in data
    
    # Check optimized configuration
    config = data["optimized_config"]
    assert "trading_pairs" in config
    assert "position_size" in config
    assert "leverage" in config
    
    # Check expected improvements
    improvements = data["expected_improvements"]
    assert "win_rate" in improvements
    assert "profit_factor" in improvements
    assert "sharpe_ratio" in improvements
    assert "max_drawdown" in improvements
    
    # Check recommendations
    recommendations = data["recommendations"]
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    
    # Check risk assessment
    risk = data["risk_assessment"]
    assert "risk_score" in risk
    assert "risk_factors" in risk
    assert "mitigation_strategies" in risk

@pytest.mark.asyncio
async def test_get_optimization_status(
    client: TestClient,
    test_user_token: str
):
    """Test optimization status endpoint"""
    # Make request
    response = client.get(
        "/optimization/status",
        headers={"Authorization": f"Bearer {test_user_token}"}
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Check status structure
    assert "is_operational" in data
    assert "model_loaded" in data
    assert "device" in data
    assert "model_name" in data
    
    # Check status values
    assert isinstance(data["is_operational"], bool)
    assert isinstance(data["model_loaded"], bool)
    assert isinstance(data["device"], str)
    assert isinstance(data["model_name"], str)
    assert data["model_name"] == "bert-base-uncased"

@pytest.mark.asyncio
async def test_optimize_activity_unauthorized(
    client: TestClient,
    mining_optimization_request: Dict[str, Any]
):
    """Test activity optimization endpoint without authorization"""
    # Make request without token
    response = client.post(
        "/optimization/optimize",
        json=mining_optimization_request
    )
    
    # Check response
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_get_optimization_status_unauthorized(client: TestClient):
    """Test optimization status endpoint without authorization"""
    # Make request without token
    response = client.get("/optimization/status")
    
    # Check response
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_optimize_activity_invalid_data(
    client: TestClient,
    test_user_token: str
):
    """Test activity optimization endpoint with invalid data"""
    # Make request with invalid data
    response = client.post(
        "/optimization/optimize",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json={"invalid": "data"}
    )
    
    # Check response
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_optimize_activity_invalid_type(
    client: TestClient,
    test_user_token: str,
    mining_optimization_request: Dict[str, Any]
):
    """Test activity optimization endpoint with invalid activity type"""
    # Modify request with invalid activity type
    invalid_request = mining_optimization_request.copy()
    invalid_request["activity_type"] = "invalid_type"
    
    # Make request
    response = client.post(
        "/optimization/optimize",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json=invalid_request
    )
    
    # Check response
    assert response.status_code == 422 