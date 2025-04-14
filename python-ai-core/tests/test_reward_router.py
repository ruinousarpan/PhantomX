import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from typing import Dict, Any, List
from sqlalchemy.orm import Session

from routers.reward import router
from database.models import ActivityType, User, Activity, Reward
from core.config import settings

client = TestClient(router)

@pytest.fixture
def mining_reward_request() -> Dict[str, Any]:
    """Create a test mining reward request"""
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
def staking_reward_request() -> Dict[str, Any]:
    """Create a test staking reward request"""
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
def trading_reward_request() -> Dict[str, Any]:
    """Create a test trading reward request"""
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
def referral_reward_request() -> Dict[str, Any]:
    """Create a test referral reward request"""
    return {
        "user_id": "test_user",
        "activity_type": ActivityType.REFERRAL,
        "activity_data": {
            "referral_code": "TEST123",
            "referred_user_id": "referred_user",
            "referral_date": datetime.utcnow().isoformat()
        },
        "performance_metrics": {
            "total_referrals": 5,
            "active_referrals": 3,
            "conversion_rate": 0.6
        },
        "historical_data": [
            {
                "timestamp": (datetime.utcnow() - timedelta(days=i)).isoformat(),
                "referral_code": f"TEST{i}",
                "referred_user_id": f"user_{i}",
                "conversion_rate": 0.5 + i * 0.02
            }
            for i in range(10)
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
def test_mining_reward_data() -> Dict[str, Any]:
    """Create test mining reward data"""
    return {
        "reward_id": "test_mining_reward",
        "user_id": "test_user",
        "activity_id": "test_mining_activity",
        "reward_type": "mining",
        "amount": 0.001,
        "currency": "BTC",
        "timestamp": datetime.utcnow(),
        "status": "pending",
        "transaction_hash": "0x1234567890abcdef",
        "metadata": {
            "block_height": 12345678,
            "block_hash": "0xabcdef1234567890",
            "transaction_index": 0
        }
    }

@pytest.fixture
def test_staking_reward_data() -> Dict[str, Any]:
    """Create test staking reward data"""
    return {
        "reward_id": "test_staking_reward",
        "user_id": "test_user",
        "activity_id": "test_staking_activity",
        "reward_type": "staking",
        "amount": 0.05,
        "currency": "ETH",
        "timestamp": datetime.utcnow(),
        "status": "pending",
        "transaction_hash": "0xabcdef1234567890",
        "metadata": {
            "epoch": 123,
            "validator_index": 456,
            "attestations": 100
        }
    }

@pytest.fixture
def test_trading_reward_data() -> Dict[str, Any]:
    """Create test trading reward data"""
    return {
        "reward_id": "test_trading_reward",
        "user_id": "test_user",
        "activity_id": "test_trading_activity",
        "reward_type": "trading",
        "amount": 100.0,
        "currency": "USDT",
        "timestamp": datetime.utcnow(),
        "status": "pending",
        "transaction_hash": "0x0987654321fedcba",
        "metadata": {
            "trading_pair": "BTC/USDT",
            "position_size": 1000.0,
            "leverage": 1.0,
            "pnl": 100.0
        }
    }

@pytest.mark.asyncio
async def test_calculate_mining_rewards(
    client: TestClient,
    test_user_token: str,
    mining_reward_request: Dict[str, Any]
):
    """Test mining reward calculation endpoint"""
    # Make request
    response = client.post(
        "/reward/calculate",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json=mining_reward_request
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Check result structure
    assert "base_reward" in data
    assert "multipliers" in data
    assert "final_reward" in data
    assert "reward_breakdown" in data
    assert "optimization_suggestions" in data
    
    # Check base reward
    assert isinstance(data["base_reward"], float)
    assert data["base_reward"] > 0
    
    # Check multipliers
    multipliers = data["multipliers"]
    assert "efficiency_multiplier" in multipliers
    assert "consistency_multiplier" in multipliers
    assert "loyalty_multiplier" in multipliers
    assert "activity_bonus" in multipliers
    
    # Check final reward
    assert isinstance(data["final_reward"], float)
    assert data["final_reward"] > 0
    
    # Check reward breakdown
    breakdown = data["reward_breakdown"]
    assert "base_reward" in breakdown
    assert "efficiency_bonus" in breakdown
    assert "consistency_bonus" in breakdown
    assert "loyalty_bonus" in breakdown
    assert "activity_bonus" in breakdown
    
    # Check optimization suggestions
    suggestions = data["optimization_suggestions"]
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0

@pytest.mark.asyncio
async def test_calculate_staking_rewards(
    client: TestClient,
    test_user_token: str,
    staking_reward_request: Dict[str, Any]
):
    """Test staking reward calculation endpoint"""
    # Make request
    response = client.post(
        "/reward/calculate",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json=staking_reward_request
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Check result structure
    assert "base_reward" in data
    assert "multipliers" in data
    assert "final_reward" in data
    assert "reward_breakdown" in data
    assert "optimization_suggestions" in data
    
    # Check base reward
    assert isinstance(data["base_reward"], float)
    assert data["base_reward"] > 0
    
    # Check multipliers
    multipliers = data["multipliers"]
    assert "efficiency_multiplier" in multipliers
    assert "consistency_multiplier" in multipliers
    assert "loyalty_multiplier" in multipliers
    assert "activity_bonus" in multipliers
    
    # Check final reward
    assert isinstance(data["final_reward"], float)
    assert data["final_reward"] > 0
    
    # Check reward breakdown
    breakdown = data["reward_breakdown"]
    assert "base_reward" in breakdown
    assert "efficiency_bonus" in breakdown
    assert "consistency_bonus" in breakdown
    assert "loyalty_bonus" in breakdown
    assert "activity_bonus" in breakdown
    
    # Check optimization suggestions
    suggestions = data["optimization_suggestions"]
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0

@pytest.mark.asyncio
async def test_calculate_trading_rewards(
    client: TestClient,
    test_user_token: str,
    trading_reward_request: Dict[str, Any]
):
    """Test trading reward calculation endpoint"""
    # Make request
    response = client.post(
        "/reward/calculate",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json=trading_reward_request
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Check result structure
    assert "base_reward" in data
    assert "multipliers" in data
    assert "final_reward" in data
    assert "reward_breakdown" in data
    assert "optimization_suggestions" in data
    
    # Check base reward
    assert isinstance(data["base_reward"], float)
    assert data["base_reward"] > 0
    
    # Check multipliers
    multipliers = data["multipliers"]
    assert "efficiency_multiplier" in multipliers
    assert "consistency_multiplier" in multipliers
    assert "loyalty_multiplier" in multipliers
    assert "activity_bonus" in multipliers
    
    # Check final reward
    assert isinstance(data["final_reward"], float)
    assert data["final_reward"] > 0
    
    # Check reward breakdown
    breakdown = data["reward_breakdown"]
    assert "base_reward" in breakdown
    assert "efficiency_bonus" in breakdown
    assert "consistency_bonus" in breakdown
    assert "loyalty_bonus" in breakdown
    assert "activity_bonus" in breakdown
    
    # Check optimization suggestions
    suggestions = data["optimization_suggestions"]
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0

@pytest.mark.asyncio
async def test_calculate_referral_rewards(
    client: TestClient,
    test_user_token: str,
    referral_reward_request: Dict[str, Any]
):
    """Test referral reward calculation endpoint"""
    # Make request
    response = client.post(
        "/reward/calculate",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json=referral_reward_request
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Check result structure
    assert "base_reward" in data
    assert "multipliers" in data
    assert "final_reward" in data
    assert "reward_breakdown" in data
    assert "optimization_suggestions" in data
    
    # Check base reward
    assert isinstance(data["base_reward"], float)
    assert data["base_reward"] > 0
    
    # Check multipliers
    multipliers = data["multipliers"]
    assert "conversion_multiplier" in multipliers
    assert "activity_multiplier" in multipliers
    assert "loyalty_multiplier" in multipliers
    
    # Check final reward
    assert isinstance(data["final_reward"], float)
    assert data["final_reward"] > 0
    
    # Check reward breakdown
    breakdown = data["reward_breakdown"]
    assert "base_reward" in breakdown
    assert "conversion_bonus" in breakdown
    assert "activity_bonus" in breakdown
    assert "loyalty_bonus" in breakdown
    
    # Check optimization suggestions
    suggestions = data["optimization_suggestions"]
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0

@pytest.mark.asyncio
async def test_get_reward_status(
    client: TestClient,
    test_user_token: str
):
    """Test reward status endpoint"""
    # Make request
    response = client.get(
        "/reward/status",
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
async def test_calculate_rewards_unauthorized(
    client: TestClient,
    mining_reward_request: Dict[str, Any]
):
    """Test reward calculation endpoint without authorization"""
    # Make request without token
    response = client.post(
        "/reward/calculate",
        json=mining_reward_request
    )
    
    # Check response
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_get_reward_status_unauthorized(client: TestClient):
    """Test reward status endpoint without authorization"""
    # Make request without token
    response = client.get("/reward/status")
    
    # Check response
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_calculate_rewards_invalid_data(
    client: TestClient,
    test_user_token: str
):
    """Test reward calculation endpoint with invalid data"""
    # Make request with invalid data
    response = client.post(
        "/reward/calculate",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json={"invalid": "data"}
    )
    
    # Check response
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_calculate_rewards_invalid_type(
    client: TestClient,
    test_user_token: str,
    mining_reward_request: Dict[str, Any]
):
    """Test reward calculation endpoint with invalid activity type"""
    # Modify request with invalid activity type
    invalid_request = mining_reward_request.copy()
    invalid_request["activity_type"] = "invalid_type"
    
    # Make request
    response = client.post(
        "/reward/calculate",
        headers={"Authorization": f"Bearer {test_user_token}"},
        json=invalid_request
    )
    
    # Check response
    assert response.status_code == 422

def test_calculate_mining_reward(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any]):
    """Test calculating mining reward"""
    # Create test user and activity
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Calculate reward
    response = client.post(
        "/reward/calculate",
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
    assert "reward" in data
    assert data["reward"]["user_id"] == test_user_data["user_id"]
    assert data["reward"]["activity_id"] == test_mining_activity_data["activity_id"]
    assert data["reward"]["reward_type"] == "mining"
    assert "amount" in data["reward"]
    assert "currency" in data["reward"]
    assert data["reward"]["status"] == "pending"

def test_calculate_staking_reward(db_session: Session, test_user_data: Dict[str, Any], test_staking_activity_data: Dict[str, Any]):
    """Test calculating staking reward"""
    # Create test user and activity
    user = User(**test_user_data)
    activity = Activity(**test_staking_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Calculate reward
    response = client.post(
        "/reward/calculate",
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
    assert "reward" in data
    assert data["reward"]["user_id"] == test_user_data["user_id"]
    assert data["reward"]["activity_id"] == test_staking_activity_data["activity_id"]
    assert data["reward"]["reward_type"] == "staking"
    assert "amount" in data["reward"]
    assert "currency" in data["reward"]
    assert data["reward"]["status"] == "pending"

def test_calculate_trading_reward(db_session: Session, test_user_data: Dict[str, Any], test_trading_activity_data: Dict[str, Any]):
    """Test calculating trading reward"""
    # Create test user and activity
    user = User(**test_user_data)
    activity = Activity(**test_trading_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Calculate reward
    response = client.post(
        "/reward/calculate",
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
    assert "reward" in data
    assert data["reward"]["user_id"] == test_user_data["user_id"]
    assert data["reward"]["activity_id"] == test_trading_activity_data["activity_id"]
    assert data["reward"]["reward_type"] == "trading"
    assert "amount" in data["reward"]
    assert "currency" in data["reward"]
    assert data["reward"]["status"] == "pending"

def test_distribute_reward(db_session: Session, test_mining_reward_data: Dict[str, Any]):
    """Test distributing reward"""
    # Create test reward
    reward = Reward(**test_mining_reward_data)
    db_session.add(reward)
    db_session.commit()

    # Distribute reward
    response = client.post(
        "/reward/distribute",
        json={
            "reward_id": test_mining_reward_data["reward_id"],
            "transaction_hash": test_mining_reward_data["transaction_hash"]
        }
    )

    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "reward" in data
    assert data["reward"]["reward_id"] == test_mining_reward_data["reward_id"]
    assert data["reward"]["status"] == "distributed"
    assert data["reward"]["transaction_hash"] == test_mining_reward_data["transaction_hash"]

def test_get_reward_status(db_session: Session, test_mining_reward_data: Dict[str, Any]):
    """Test getting reward status"""
    # Create test reward
    reward = Reward(**test_mining_reward_data)
    db_session.add(reward)
    db_session.commit()

    # Get reward status
    response = client.get(f"/reward/status/{test_mining_reward_data['reward_id']}")

    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "reward" in data
    assert data["reward"]["reward_id"] == test_mining_reward_data["reward_id"]
    assert data["reward"]["status"] == test_mining_reward_data["status"]

def test_get_user_rewards(db_session: Session, test_user_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any]):
    """Test getting user rewards"""
    # Create test user and reward
    user = User(**test_user_data)
    reward = Reward(**test_mining_reward_data)
    db_session.add(user)
    db_session.add(reward)
    db_session.commit()

    # Get user rewards
    response = client.get(f"/reward/user/{test_user_data['user_id']}")

    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "rewards" in data
    assert len(data["rewards"]) == 1
    assert data["rewards"][0]["reward_id"] == test_mining_reward_data["reward_id"]
    assert data["rewards"][0]["user_id"] == test_user_data["user_id"]

def test_get_activity_rewards(db_session: Session, test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any]):
    """Test getting activity rewards"""
    # Create test activity and reward
    activity = Activity(**test_mining_activity_data)
    reward = Reward(**test_mining_reward_data)
    db_session.add(activity)
    db_session.add(reward)
    db_session.commit()

    # Get activity rewards
    response = client.get(f"/reward/activity/{test_mining_activity_data['activity_id']}")

    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "rewards" in data
    assert len(data["rewards"]) == 1
    assert data["rewards"][0]["reward_id"] == test_mining_reward_data["reward_id"]
    assert data["rewards"][0]["activity_id"] == test_mining_activity_data["activity_id"]

def test_calculate_reward_unauthorized():
    """Test calculating reward without authorization"""
    response = client.post(
        "/reward/calculate",
        json={
            "user_id": "test_user",
            "activity_id": "test_activity",
            "activity_type": "mining",
            "performance_metrics": {}
        }
    )
    assert response.status_code == 401

def test_distribute_reward_unauthorized():
    """Test distributing reward without authorization"""
    response = client.post(
        "/reward/distribute",
        json={
            "reward_id": "test_reward",
            "transaction_hash": "0x1234567890abcdef"
        }
    )
    assert response.status_code == 401

def test_get_reward_status_unauthorized():
    """Test getting reward status without authorization"""
    response = client.get("/reward/status/test_reward")
    assert response.status_code == 401

def test_calculate_reward_invalid_data():
    """Test calculating reward with invalid data"""
    response = client.post(
        "/reward/calculate",
        json={
            "user_id": "test_user",
            "activity_id": "test_activity",
            "activity_type": "invalid_type",
            "performance_metrics": {}
        }
    )
    assert response.status_code == 422

def test_distribute_reward_invalid_data():
    """Test distributing reward with invalid data"""
    response = client.post(
        "/reward/distribute",
        json={
            "reward_id": "test_reward",
            "transaction_hash": "invalid_hash"
        }
    )
    assert response.status_code == 422 