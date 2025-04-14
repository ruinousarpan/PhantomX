import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from typing import Dict, Any, List

from routers.prediction import router
from database.models import ActivityType

@pytest.fixture
def mining_prediction_request() -> Dict[str, Any]:
    """Create a test mining prediction request"""
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
                "timestamp": (datetime.utcnow() - timedelta(days=i)).isoformat(),
                "hash_rate": 90.0 + i * 0.5,
                "power_usage": 440.0 + i * 2.0,
                "temperature": 74.0 + i * 0.2,
                "efficiency": 0.8 + i * 0.01,
                "duration": 3600
            }
            for i in range(30)
        ]
    }

@pytest.fixture
def staking_prediction_request() -> Dict[str, Any]:
    """Create a test staking prediction request"""
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
                "timestamp": (datetime.utcnow() - timedelta(days=i)).isoformat(),
                "stake_amount": 900.0 + i * 10.0,
                "rewards": 40.0 + i * 0.5,
                "uptime": 0.97 - i * 0.001,
                "network_health": 0.94 - i * 0.002,
                "duration": 86400
            }
            for i in range(30)
        ]
    }

@pytest.fixture
def trading_prediction_request() -> Dict[str, Any]:
    """Create a test trading prediction request"""
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
                "timestamp": (datetime.utcnow() - timedelta(days=i)).isoformat(),
                "win_rate": 0.62 + i * 0.001,
                "profit_factor": 1.7 + i * 0.01,
                "sharpe_ratio": 1.4 + i * 0.005,
                "max_drawdown": 0.16 - i * 0.001,
                "duration": 86400
            }
            for i in range(30)
        ]
    }

@pytest.mark.asyncio
async def test_predict_mining_activity(
    client: TestClient,
    test_user_token: str,
    mining_prediction_request: Dict[str, Any]
):
    """Test mining activity prediction endpoint"""
    # Make request
    response = client.post(
        "/prediction/predict",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json=mining_prediction_request
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Check result structure
    assert "predictions" in data
    assert "confidence_scores" in data
    assert "trend_analysis" in data
    assert "recommendations" in data
    
    # Check predictions
    predictions = data["predictions"]
    assert "hash_rate" in predictions
    assert "power_efficiency" in predictions
    assert "temperature" in predictions
    assert "overall_efficiency" in predictions
    
    # Check confidence scores
    confidence = data["confidence_scores"]
    assert "hash_rate" in confidence
    assert "power_efficiency" in confidence
    assert "temperature" in confidence
    assert "overall_efficiency" in confidence
    
    # Check trend analysis
    trends = data["trend_analysis"]
    assert "hash_rate_trend" in trends
    assert "power_efficiency_trend" in trends
    assert "temperature_trend" in trends
    assert "overall_efficiency_trend" in trends
    
    # Check recommendations
    recommendations = data["recommendations"]
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0

@pytest.mark.asyncio
async def test_predict_staking_activity(
    client: TestClient,
    test_user_token: str,
    staking_prediction_request: Dict[str, Any]
):
    """Test staking activity prediction endpoint"""
    # Make request
    response = client.post(
        "/prediction/predict",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json=staking_prediction_request
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Check result structure
    assert "predictions" in data
    assert "confidence_scores" in data
    assert "trend_analysis" in data
    assert "recommendations" in data
    
    # Check predictions
    predictions = data["predictions"]
    assert "rewards" in predictions
    assert "uptime" in predictions
    assert "network_health" in predictions
    
    # Check confidence scores
    confidence = data["confidence_scores"]
    assert "rewards" in confidence
    assert "uptime" in confidence
    assert "network_health" in confidence
    
    # Check trend analysis
    trends = data["trend_analysis"]
    assert "rewards_trend" in trends
    assert "uptime_trend" in trends
    assert "network_health_trend" in trends
    
    # Check recommendations
    recommendations = data["recommendations"]
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0

@pytest.mark.asyncio
async def test_predict_trading_activity(
    client: TestClient,
    test_user_token: str,
    trading_prediction_request: Dict[str, Any]
):
    """Test trading activity prediction endpoint"""
    # Make request
    response = client.post(
        "/prediction/predict",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json=trading_prediction_request
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Check result structure
    assert "predictions" in data
    assert "confidence_scores" in data
    assert "trend_analysis" in data
    assert "recommendations" in data
    
    # Check predictions
    predictions = data["predictions"]
    assert "win_rate" in predictions
    assert "profit_factor" in predictions
    assert "sharpe_ratio" in predictions
    assert "max_drawdown" in predictions
    
    # Check confidence scores
    confidence = data["confidence_scores"]
    assert "win_rate" in confidence
    assert "profit_factor" in confidence
    assert "sharpe_ratio" in confidence
    assert "max_drawdown" in confidence
    
    # Check trend analysis
    trends = data["trend_analysis"]
    assert "win_rate_trend" in trends
    assert "profit_factor_trend" in trends
    assert "sharpe_ratio_trend" in trends
    assert "max_drawdown_trend" in trends
    
    # Check recommendations
    recommendations = data["recommendations"]
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0

@pytest.mark.asyncio
async def test_analyze_trends(
    client: TestClient,
    test_user_token: str,
    mining_prediction_request: Dict[str, Any]
):
    """Test trend analysis endpoint"""
    # Make request
    response = client.post(
        "/prediction/trends",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json=mining_prediction_request
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Check result structure
    assert "trends" in data
    assert "patterns" in data
    assert "seasonality" in data
    assert "correlations" in data
    
    # Check trends
    trends = data["trends"]
    assert "hash_rate" in trends
    assert "power_efficiency" in trends
    assert "temperature" in trends
    assert "overall_efficiency" in trends
    
    # Check patterns
    patterns = data["patterns"]
    assert isinstance(patterns, list)
    assert len(patterns) > 0
    
    # Check seasonality
    seasonality = data["seasonality"]
    assert "daily" in seasonality
    assert "weekly" in seasonality
    assert "monthly" in seasonality
    
    # Check correlations
    correlations = data["correlations"]
    assert isinstance(correlations, dict)
    assert len(correlations) > 0

@pytest.mark.asyncio
async def test_get_prediction_status(
    client: TestClient,
    test_user_token: str
):
    """Test prediction status endpoint"""
    # Make request
    response = client.get(
        "/prediction/status",
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
async def test_predict_activity_unauthorized(
    client: TestClient,
    mining_prediction_request: Dict[str, Any]
):
    """Test activity prediction endpoint without authorization"""
    # Make request without token
    response = client.post(
        "/prediction/predict",
        json=mining_prediction_request
    )
    
    # Check response
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_analyze_trends_unauthorized(
    client: TestClient,
    mining_prediction_request: Dict[str, Any]
):
    """Test trend analysis endpoint without authorization"""
    # Make request without token
    response = client.post(
        "/prediction/trends",
        json=mining_prediction_request
    )
    
    # Check response
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_get_prediction_status_unauthorized(client: TestClient):
    """Test prediction status endpoint without authorization"""
    # Make request without token
    response = client.get("/prediction/status")
    
    # Check response
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_predict_activity_invalid_data(
    client: TestClient,
    test_user_token: str
):
    """Test activity prediction endpoint with invalid data"""
    # Make request with invalid data
    response = client.post(
        "/prediction/predict",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json={"invalid": "data"}
    )
    
    # Check response
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_analyze_trends_invalid_data(
    client: TestClient,
    test_user_token: str
):
    """Test trend analysis endpoint with invalid data"""
    # Make request with invalid data
    response = client.post(
        "/prediction/trends",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json={"invalid": "data"}
    )
    
    # Check response
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_predict_activity_invalid_type(
    client: TestClient,
    test_user_token: str,
    mining_prediction_request: Dict[str, Any]
):
    """Test activity prediction endpoint with invalid activity type"""
    # Modify request with invalid activity type
    invalid_request = mining_prediction_request.copy()
    invalid_request["activity_type"] = "invalid_type"
    
    # Make request
    response = client.post(
        "/prediction/predict",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json=invalid_request
    )
    
    # Check response
    assert response.status_code == 422 