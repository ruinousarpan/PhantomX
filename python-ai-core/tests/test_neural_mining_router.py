import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from typing import Dict, Any

from routers.neural_mining import router
from database.models import ActivityType

@pytest.fixture
def mining_session_request() -> Dict[str, Any]:
    """Create a test mining session request"""
    return {
        "user_id": "test_user",
        "device_type": "gpu",
        "focus_score": 0.85,
        "duration": 3600,  # 1 hour
        "timestamp": datetime.utcnow().isoformat(),
        "performance_metrics": {
            "hash_rate": 95.0,
            "power_efficiency": 0.8,
            "temperature": 78.0,
            "overall_efficiency": 0.82
        }
    }

@pytest.fixture
def mining_optimization_request() -> Dict[str, Any]:
    """Create a test mining optimization request"""
    return {
        "user_id": "test_user",
        "device_type": "gpu",
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

@pytest.mark.asyncio
async def test_analyze_mining_session(
    client: TestClient,
    test_user_token: str,
    mining_session_request: Dict[str, Any]
):
    """Test mining session analysis endpoint"""
    # Make request
    response = client.post(
        "/mining/analyze",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json=mining_session_request
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Check result structure
    assert "efficiency_score" in data
    assert "performance_metrics" in data
    assert "behavior_analysis" in data
    assert "recommendations" in data
    assert "potential_rewards" in data
    
    # Check efficiency score
    assert isinstance(data["efficiency_score"], float)
    assert 0 <= data["efficiency_score"] <= 1
    
    # Check performance metrics
    metrics = data["performance_metrics"]
    assert "hash_rate" in metrics
    assert "power_efficiency" in metrics
    assert "temperature" in metrics
    assert "overall_efficiency" in metrics
    
    # Check behavior analysis
    behavior = data["behavior_analysis"]
    assert "consistency_score" in behavior
    assert "duration_score" in behavior
    assert "focus_score" in behavior
    
    # Check recommendations
    recommendations = data["recommendations"]
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    
    # Check potential rewards
    rewards = data["potential_rewards"]
    assert "estimated_rewards" in rewards
    assert "multipliers" in rewards
    assert "optimization_bonus" in rewards

@pytest.mark.asyncio
async def test_optimize_mining_configuration(
    client: TestClient,
    test_user_token: str,
    mining_optimization_request: Dict[str, Any]
):
    """Test mining configuration optimization endpoint"""
    # Make request
    response = client.post(
        "/mining/optimize",
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
async def test_get_mining_status(
    client: TestClient,
    test_user_token: str
):
    """Test mining status endpoint"""
    # Make request
    response = client.get(
        "/mining/status",
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
async def test_analyze_mining_session_unauthorized(
    client: TestClient,
    mining_session_request: Dict[str, Any]
):
    """Test mining session analysis endpoint without authorization"""
    # Make request without token
    response = client.post(
        "/mining/analyze",
        json=mining_session_request
    )
    
    # Check response
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_optimize_mining_configuration_unauthorized(
    client: TestClient,
    mining_optimization_request: Dict[str, Any]
):
    """Test mining configuration optimization endpoint without authorization"""
    # Make request without token
    response = client.post(
        "/mining/optimize",
        json=mining_optimization_request
    )
    
    # Check response
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_get_mining_status_unauthorized(client: TestClient):
    """Test mining status endpoint without authorization"""
    # Make request without token
    response = client.get("/mining/status")
    
    # Check response
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_analyze_mining_session_invalid_data(
    client: TestClient,
    test_user_token: str
):
    """Test mining session analysis endpoint with invalid data"""
    # Make request with invalid data
    response = client.post(
        "/mining/analyze",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json={"invalid": "data"}
    )
    
    # Check response
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_optimize_mining_configuration_invalid_data(
    client: TestClient,
    test_user_token: str
):
    """Test mining configuration optimization endpoint with invalid data"""
    # Make request with invalid data
    response = client.post(
        "/mining/optimize",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json={"invalid": "data"}
    )
    
    # Check response
    assert response.status_code == 422 