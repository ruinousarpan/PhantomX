import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from typing import Dict, Any, List
from sqlalchemy.orm import Session

from routers.risk import router
from database.models import ActivityType, User, Activity, RiskAssessment
from core.config import settings

client = TestClient(router)

@pytest.fixture
def mining_risk_request() -> Dict[str, Any]:
    """Create a test mining risk request"""
    return {
        "user_id": "test_user",
        "activity_type": ActivityType.MINING,
        "activity_data": {
            "hash_rate": 95.0,
            "power_usage": 450.0,
            "temperature": 78.0,
            "efficiency": 0.82,
            "duration": 3600  # 1 hour
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
def staking_risk_request() -> Dict[str, Any]:
    """Create a test staking risk request"""
    return {
        "user_id": "test_user",
        "activity_type": ActivityType.STAKING,
        "activity_data": {
            "stake_amount": 1000.0,
            "lock_period": 30,
            "rewards_rate": 0.05,
            "duration": 86400  # 1 day
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
def trading_risk_request() -> Dict[str, Any]:
    """Create a test trading risk request"""
    return {
        "user_id": "test_user",
        "activity_type": ActivityType.TRADING,
        "activity_data": {
            "trading_pairs": ["BTC/USD", "ETH/USD"],
            "position_size": 0.1,
            "leverage": 1.0,
            "duration": 86400  # 1 day
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

@pytest.fixture
def test_user_data() -> Dict[str, Any]:
    """Create test user data"""
    return {
        "user_id": "test_user",
        "username": "testuser",
        "email": "test@example.com",
        "created_at": datetime.utcnow(),
        "last_login": datetime.utcnow(),
        "is_active": True,
        "is_verified": True,
        "preferences": {
            "notification_enabled": True,
            "theme": "dark",
            "language": "en"
        }
    }

@pytest.fixture
def test_mining_activity_data() -> Dict[str, Any]:
    """Create test mining activity data"""
    return {
        "activity_id": "test_mining_activity",
        "user_id": "test_user",
        "activity_type": "mining",
        "start_time": datetime.utcnow() - timedelta(hours=1),
        "end_time": datetime.utcnow(),
        "status": "completed",
        "device_type": "gpu",
        "hash_rate": 100.0,
        "power_usage": 500.0,
        "efficiency_score": 0.85,
        "performance_metrics": {
            "hash_rate": 100.0,
            "power_usage": 500.0,
            "efficiency": 0.85,
            "temperature": 75.0,
            "fan_speed": 80.0
        }
    }

@pytest.fixture
def test_staking_activity_data() -> Dict[str, Any]:
    """Create test staking activity data"""
    return {
        "activity_id": "test_staking_activity",
        "user_id": "test_user",
        "activity_type": "staking",
        "start_time": datetime.utcnow() - timedelta(days=1),
        "end_time": datetime.utcnow(),
        "status": "completed",
        "stake_amount": 1000.0,
        "validator_id": "test_validator",
        "uptime": 0.99,
        "performance_metrics": {
            "stake_amount": 1000.0,
            "uptime": 0.99,
            "rewards_earned": 50.0,
            "network_health": 0.95
        }
    }

@pytest.fixture
def test_trading_activity_data() -> Dict[str, Any]:
    """Create test trading activity data"""
    return {
        "activity_id": "test_trading_activity",
        "user_id": "test_user",
        "activity_type": "trading",
        "start_time": datetime.utcnow() - timedelta(hours=2),
        "end_time": datetime.utcnow(),
        "status": "completed",
        "trading_pairs": ["BTC/USDT", "ETH/USDT"],
        "position_size": 1000.0,
        "leverage": 1.0,
        "performance_metrics": {
            "win_rate": 0.65,
            "profit_factor": 1.8,
            "sharpe_ratio": 2.1,
            "max_drawdown": 0.15
        }
    }

@pytest.fixture
def test_risk_assessment_data() -> Dict[str, Any]:
    """Create test risk assessment data"""
    return {
        "assessment_id": "test_assessment",
        "user_id": "test_user",
        "activity_id": "test_mining_activity",
        "activity_type": "mining",
        "timestamp": datetime.utcnow(),
        "risk_score": 0.75,
        "risk_level": "high",
        "risk_factors": [
            {
                "factor": "temperature",
                "score": 0.8,
                "threshold": 80.0,
                "current": 75.0
            },
            {
                "factor": "power_usage",
                "score": 0.7,
                "threshold": 600.0,
                "current": 500.0
            }
        ],
        "mitigation_suggestions": [
            {
                "factor": "temperature",
                "suggestion": "Increase fan speed",
                "expected_improvement": 0.1
            },
            {
                "factor": "power_usage",
                "suggestion": "Optimize power settings",
                "expected_improvement": 0.15
            }
        ]
    }

@pytest.mark.asyncio
async def test_assess_mining_risk(
    client: TestClient,
    test_user_token: str,
    mining_risk_request: Dict[str, Any]
):
    """Test mining risk assessment endpoint"""
    # Make request
    response = client.post(
        "/risk/assess",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json=mining_risk_request
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Check result structure
    assert "risk_score" in data
    assert "risk_level" in data
    assert "risk_factors" in data
    assert "risk_breakdown" in data
    assert "mitigation_strategies" in data
    assert "impact_analysis" in data
    
    # Check risk score
    assert isinstance(data["risk_score"], float)
    assert 0 <= data["risk_score"] <= 1
    
    # Check risk level
    assert isinstance(data["risk_level"], str)
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    
    # Check risk factors
    factors = data["risk_factors"]
    assert "hardware_risk" in factors
    assert "network_risk" in factors
    assert "market_risk" in factors
    assert "operational_risk" in factors
    
    # Check risk breakdown
    breakdown = data["risk_breakdown"]
    assert "hardware_risk" in breakdown
    assert "network_risk" in breakdown
    assert "market_risk" in breakdown
    assert "operational_risk" in breakdown
    
    # Check mitigation strategies
    strategies = data["mitigation_strategies"]
    assert isinstance(strategies, list)
    assert len(strategies) > 0
    
    # Check impact analysis
    impact = data["impact_analysis"]
    assert "performance_impact" in impact
    assert "financial_impact" in impact
    assert "reliability_impact" in impact

@pytest.mark.asyncio
async def test_assess_staking_risk(
    client: TestClient,
    test_user_token: str,
    staking_risk_request: Dict[str, Any]
):
    """Test staking risk assessment endpoint"""
    # Make request
    response = client.post(
        "/risk/assess",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json=staking_risk_request
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Check result structure
    assert "risk_score" in data
    assert "risk_level" in data
    assert "risk_factors" in data
    assert "risk_breakdown" in data
    assert "mitigation_strategies" in data
    assert "impact_analysis" in data
    
    # Check risk score
    assert isinstance(data["risk_score"], float)
    assert 0 <= data["risk_score"] <= 1
    
    # Check risk level
    assert isinstance(data["risk_level"], str)
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    
    # Check risk factors
    factors = data["risk_factors"]
    assert "network_risk" in factors
    assert "slashing_risk" in factors
    assert "market_risk" in factors
    assert "validator_risk" in factors
    
    # Check risk breakdown
    breakdown = data["risk_breakdown"]
    assert "network_risk" in breakdown
    assert "slashing_risk" in breakdown
    assert "market_risk" in breakdown
    assert "validator_risk" in breakdown
    
    # Check mitigation strategies
    strategies = data["mitigation_strategies"]
    assert isinstance(strategies, list)
    assert len(strategies) > 0
    
    # Check impact analysis
    impact = data["impact_analysis"]
    assert "performance_impact" in impact
    assert "financial_impact" in impact
    assert "reliability_impact" in impact

@pytest.mark.asyncio
async def test_assess_trading_risk(
    client: TestClient,
    test_user_token: str,
    trading_risk_request: Dict[str, Any]
):
    """Test trading risk assessment endpoint"""
    # Make request
    response = client.post(
        "/risk/assess",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json=trading_risk_request
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Check result structure
    assert "risk_score" in data
    assert "risk_level" in data
    assert "risk_factors" in data
    assert "risk_breakdown" in data
    assert "mitigation_strategies" in data
    assert "impact_analysis" in data
    
    # Check risk score
    assert isinstance(data["risk_score"], float)
    assert 0 <= data["risk_score"] <= 1
    
    # Check risk level
    assert isinstance(data["risk_level"], str)
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    
    # Check risk factors
    factors = data["risk_factors"]
    assert "market_risk" in factors
    assert "liquidity_risk" in factors
    assert "volatility_risk" in factors
    assert "leverage_risk" in factors
    
    # Check risk breakdown
    breakdown = data["risk_breakdown"]
    assert "market_risk" in breakdown
    assert "liquidity_risk" in breakdown
    assert "volatility_risk" in breakdown
    assert "leverage_risk" in breakdown
    
    # Check mitigation strategies
    strategies = data["mitigation_strategies"]
    assert isinstance(strategies, list)
    assert len(strategies) > 0
    
    # Check impact analysis
    impact = data["impact_analysis"]
    assert "performance_impact" in impact
    assert "financial_impact" in impact
    assert "reliability_impact" in impact

@pytest.mark.asyncio
async def test_get_risk_status(
    client: TestClient,
    test_user_token: str
):
    """Test risk status endpoint"""
    # Make request
    response = client.get(
        "/risk/status",
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
async def test_assess_risk_unauthorized(
    client: TestClient,
    mining_risk_request: Dict[str, Any]
):
    """Test risk assessment endpoint without authorization"""
    # Make request without token
    response = client.post(
        "/risk/assess",
        json=mining_risk_request
    )
    
    # Check response
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_get_risk_status_unauthorized(client: TestClient):
    """Test risk status endpoint without authorization"""
    # Make request without token
    response = client.get("/risk/status")
    
    # Check response
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_assess_risk_invalid_data(
    client: TestClient,
    test_user_token: str
):
    """Test risk assessment endpoint with invalid data"""
    # Make request with invalid data
    response = client.post(
        "/risk/assess",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json={"invalid": "data"}
    )
    
    # Check response
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_assess_risk_invalid_type(
    client: TestClient,
    test_user_token: str,
    mining_risk_request: Dict[str, Any]
):
    """Test risk assessment endpoint with invalid activity type"""
    # Modify request with invalid activity type
    invalid_request = mining_risk_request.copy()
    invalid_request["activity_type"] = "invalid_type"
    
    # Make request
    response = client.post(
        "/risk/assess",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json=invalid_request
    )
    
    # Check response
    assert response.status_code == 422

def test_assess_mining_risk(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any]):
    """Test mining risk assessment"""
    # Create test user and activity
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Assess risk
    response = client.post(
        "/risk/assess",
        json={
            "user_id": test_user_data["user_id"],
            "activity_id": test_mining_activity_data["activity_id"],
            "activity_type": "mining",
            "performance_metrics": test_mining_activity_data["performance_metrics"]
        }
    )

    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "assessment" in data
    assert data["assessment"]["user_id"] == test_user_data["user_id"]
    assert data["assessment"]["activity_id"] == test_mining_activity_data["activity_id"]
    assert data["assessment"]["activity_type"] == "mining"
    assert "risk_score" in data["assessment"]
    assert "risk_level" in data["assessment"]
    assert "risk_factors" in data["assessment"]
    assert "mitigation_suggestions" in data["assessment"]

def test_assess_staking_risk(db_session: Session, test_user_data: Dict[str, Any], test_staking_activity_data: Dict[str, Any]):
    """Test staking risk assessment"""
    # Create test user and activity
    user = User(**test_user_data)
    activity = Activity(**test_staking_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Assess risk
    response = client.post(
        "/risk/assess",
        json={
            "user_id": test_user_data["user_id"],
            "activity_id": test_staking_activity_data["activity_id"],
            "activity_type": "staking",
            "performance_metrics": test_staking_activity_data["performance_metrics"]
        }
    )

    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "assessment" in data
    assert data["assessment"]["user_id"] == test_user_data["user_id"]
    assert data["assessment"]["activity_id"] == test_staking_activity_data["activity_id"]
    assert data["assessment"]["activity_type"] == "staking"
    assert "risk_score" in data["assessment"]
    assert "risk_level" in data["assessment"]
    assert "risk_factors" in data["assessment"]
    assert "mitigation_suggestions" in data["assessment"]

def test_assess_trading_risk(db_session: Session, test_user_data: Dict[str, Any], test_trading_activity_data: Dict[str, Any]):
    """Test trading risk assessment"""
    # Create test user and activity
    user = User(**test_user_data)
    activity = Activity(**test_trading_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Assess risk
    response = client.post(
        "/risk/assess",
        json={
            "user_id": test_user_data["user_id"],
            "activity_id": test_trading_activity_data["activity_id"],
            "activity_type": "trading",
            "performance_metrics": test_trading_activity_data["performance_metrics"]
        }
    )

    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "assessment" in data
    assert data["assessment"]["user_id"] == test_user_data["user_id"]
    assert data["assessment"]["activity_id"] == test_trading_activity_data["activity_id"]
    assert data["assessment"]["activity_type"] == "trading"
    assert "risk_score" in data["assessment"]
    assert "risk_level" in data["assessment"]
    assert "risk_factors" in data["assessment"]
    assert "mitigation_suggestions" in data["assessment"]

def test_monitor_risk(db_session: Session, test_risk_assessment_data: Dict[str, Any]):
    """Test risk monitoring"""
    # Create test risk assessment
    assessment = RiskAssessment(**test_risk_assessment_data)
    db_session.add(assessment)
    db_session.commit()

    # Monitor risk
    response = client.get(
        f"/risk/monitor/{test_risk_assessment_data['assessment_id']}"
    )

    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "monitoring" in data
    assert data["monitoring"]["assessment_id"] == test_risk_assessment_data["assessment_id"]
    assert "current_risk_score" in data["monitoring"]
    assert "risk_trend" in data["monitoring"]
    assert "alerts" in data["monitoring"]

def test_get_risk_mitigation(db_session: Session, test_risk_assessment_data: Dict[str, Any]):
    """Test getting risk mitigation suggestions"""
    # Create test risk assessment
    assessment = RiskAssessment(**test_risk_assessment_data)
    db_session.add(assessment)
    db_session.commit()

    # Get mitigation suggestions
    response = client.get(
        f"/risk/mitigation/{test_risk_assessment_data['assessment_id']}"
    )

    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "mitigation" in data
    assert data["mitigation"]["assessment_id"] == test_risk_assessment_data["assessment_id"]
    assert "suggestions" in data["mitigation"]
    assert len(data["mitigation"]["suggestions"]) > 0
    for suggestion in data["mitigation"]["suggestions"]:
        assert "factor" in suggestion
        assert "suggestion" in suggestion
        assert "expected_improvement" in suggestion

def test_get_user_risk_history(db_session: Session, test_user_data: Dict[str, Any], test_risk_assessment_data: Dict[str, Any]):
    """Test getting user risk history"""
    # Create test user and risk assessment
    user = User(**test_user_data)
    assessment = RiskAssessment(**test_risk_assessment_data)
    db_session.add(user)
    db_session.add(assessment)
    db_session.commit()

    # Get user risk history
    response = client.get(
        f"/risk/history/{test_user_data['user_id']}"
    )

    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "history" in data
    assert len(data["history"]) == 1
    assert data["history"][0]["assessment_id"] == test_risk_assessment_data["assessment_id"]
    assert data["history"][0]["user_id"] == test_user_data["user_id"]

def test_get_activity_risk_history(db_session: Session, test_mining_activity_data: Dict[str, Any], test_risk_assessment_data: Dict[str, Any]):
    """Test getting activity risk history"""
    # Create test activity and risk assessment
    activity = Activity(**test_mining_activity_data)
    assessment = RiskAssessment(**test_risk_assessment_data)
    db_session.add(activity)
    db_session.add(assessment)
    db_session.commit()

    # Get activity risk history
    response = client.get(
        f"/risk/history/activity/{test_mining_activity_data['activity_id']}"
    )

    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "history" in data
    assert len(data["history"]) == 1
    assert data["history"][0]["assessment_id"] == test_risk_assessment_data["assessment_id"]
    assert data["history"][0]["activity_id"] == test_mining_activity_data["activity_id"]

def test_assess_risk_unauthorized():
    """Test risk assessment without authorization"""
    response = client.post(
        "/risk/assess",
        json={
            "user_id": "test_user",
            "activity_id": "test_activity",
            "activity_type": "mining",
            "performance_metrics": {}
        }
    )
    assert response.status_code == 401

def test_monitor_risk_unauthorized():
    """Test risk monitoring without authorization"""
    response = client.get("/risk/monitor/test_assessment")
    assert response.status_code == 401

def test_get_mitigation_unauthorized():
    """Test getting mitigation suggestions without authorization"""
    response = client.get("/risk/mitigation/test_assessment")
    assert response.status_code == 401

def test_assess_risk_invalid_data():
    """Test risk assessment with invalid data"""
    response = client.post(
        "/risk/assess",
        json={
            "user_id": "test_user",
            "activity_id": "test_activity",
            "activity_type": "invalid_type",
            "performance_metrics": {}
        }
    )
    assert response.status_code == 422

def test_monitor_risk_invalid_data():
    """Test risk monitoring with invalid data"""
    response = client.get("/risk/monitor/invalid_assessment")
    assert response.status_code == 422 