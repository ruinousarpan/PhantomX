import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List

from database.models import (
    User,
    Activity,
    Reward,
    ActivityType,
    RewardType,
    RiskLevel,
    DeviceType
)

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
def test_activity_data() -> Dict[str, Any]:
    """Create test activity data"""
    return {
        "activity_id": "test_activity",
        "user_id": "test_user",
        "activity_type": ActivityType.MINING,
        "start_time": datetime.utcnow(),
        "end_time": datetime.utcnow() + timedelta(hours=1),
        "duration": 3600,  # 1 hour
        "status": "completed",
        "performance_metrics": {
            "hash_rate": 95.0,
            "power_efficiency": 0.8,
            "temperature": 78.0,
            "overall_efficiency": 0.82
        },
        "device_info": {
            "device_type": DeviceType.GPU,
            "model": "RTX 3080",
            "memory": 10,
            "driver_version": "515.65.01"
        },
        "network_info": {
            "pool": "stratum+tcp://pool.example.com:3333",
            "latency": 50,
            "connection_quality": 0.95
        }
    }

@pytest.fixture
def test_reward_data() -> Dict[str, Any]:
    """Create test reward data"""
    return {
        "reward_id": "test_reward",
        "user_id": "test_user",
        "activity_id": "test_activity",
        "reward_type": RewardType.MINING,
        "amount": 100.0,
        "currency": "BTC",
        "timestamp": datetime.utcnow(),
        "status": "pending",
        "multipliers": {
            "efficiency_multiplier": 1.2,
            "consistency_multiplier": 1.1,
            "loyalty_multiplier": 1.05,
            "activity_bonus": 1.1
        },
        "breakdown": {
            "base_reward": 80.0,
            "efficiency_bonus": 16.0,
            "consistency_bonus": 8.8,
            "loyalty_bonus": 4.2,
            "activity_bonus": 8.8
        }
    }

def test_user_creation(test_user_data: Dict[str, Any]):
    """Test user model creation"""
    # Create user
    user = User(**test_user_data)
    
    # Check attributes
    assert user.user_id == test_user_data["user_id"]
    assert user.username == test_user_data["username"]
    assert user.email == test_user_data["email"]
    assert user.created_at == test_user_data["created_at"]
    assert user.last_login == test_user_data["last_login"]
    assert user.is_active == test_user_data["is_active"]
    assert user.is_verified == test_user_data["is_verified"]
    assert user.preferences == test_user_data["preferences"]
    
    # Check relationships
    assert hasattr(user, "activities")
    assert hasattr(user, "rewards")

def test_activity_creation(test_activity_data: Dict[str, Any]):
    """Test activity model creation"""
    # Create activity
    activity = Activity(**test_activity_data)
    
    # Check attributes
    assert activity.activity_id == test_activity_data["activity_id"]
    assert activity.user_id == test_activity_data["user_id"]
    assert activity.activity_type == test_activity_data["activity_type"]
    assert activity.start_time == test_activity_data["start_time"]
    assert activity.end_time == test_activity_data["end_time"]
    assert activity.duration == test_activity_data["duration"]
    assert activity.status == test_activity_data["status"]
    assert activity.performance_metrics == test_activity_data["performance_metrics"]
    assert activity.device_info == test_activity_data["device_info"]
    assert activity.network_info == test_activity_data["network_info"]
    
    # Check relationships
    assert hasattr(activity, "user")
    assert hasattr(activity, "rewards")

def test_reward_creation(test_reward_data: Dict[str, Any]):
    """Test reward model creation"""
    # Create reward
    reward = Reward(**test_reward_data)
    
    # Check attributes
    assert reward.reward_id == test_reward_data["reward_id"]
    assert reward.user_id == test_reward_data["user_id"]
    assert reward.activity_id == test_reward_data["activity_id"]
    assert reward.reward_type == test_reward_data["reward_type"]
    assert reward.amount == test_reward_data["amount"]
    assert reward.currency == test_reward_data["currency"]
    assert reward.timestamp == test_reward_data["timestamp"]
    assert reward.status == test_reward_data["status"]
    assert reward.multipliers == test_reward_data["multipliers"]
    assert reward.breakdown == test_reward_data["breakdown"]
    
    # Check relationships
    assert hasattr(reward, "user")
    assert hasattr(reward, "activity")

def test_user_activity_relationship(test_user_data: Dict[str, Any], test_activity_data: Dict[str, Any]):
    """Test user-activity relationship"""
    # Create user and activity
    user = User(**test_user_data)
    activity = Activity(**test_activity_data)
    
    # Set relationship
    activity.user = user
    user.activities.append(activity)
    
    # Check relationship
    assert activity.user == user
    assert activity in user.activities

def test_activity_reward_relationship(test_activity_data: Dict[str, Any], test_reward_data: Dict[str, Any]):
    """Test activity-reward relationship"""
    # Create activity and reward
    activity = Activity(**test_activity_data)
    reward = Reward(**test_reward_data)
    
    # Set relationship
    reward.activity = activity
    activity.rewards.append(reward)
    
    # Check relationship
    assert reward.activity == activity
    assert reward in activity.rewards

def test_user_reward_relationship(test_user_data: Dict[str, Any], test_reward_data: Dict[str, Any]):
    """Test user-reward relationship"""
    # Create user and reward
    user = User(**test_user_data)
    reward = Reward(**test_reward_data)
    
    # Set relationship
    reward.user = user
    user.rewards.append(reward)
    
    # Check relationship
    assert reward.user == user
    assert reward in user.rewards

def test_activity_type_enum():
    """Test activity type enum"""
    # Check enum values
    assert ActivityType.MINING == "mining"
    assert ActivityType.STAKING == "staking"
    assert ActivityType.TRADING == "trading"
    assert ActivityType.REFERRAL == "referral"

def test_reward_type_enum():
    """Test reward type enum"""
    # Check enum values
    assert RewardType.MINING == "mining"
    assert RewardType.STAKING == "staking"
    assert RewardType.TRADING == "trading"
    assert RewardType.REFERRAL == "referral"

def test_risk_level_enum():
    """Test risk level enum"""
    # Check enum values
    assert RiskLevel.LOW == "LOW"
    assert RiskLevel.MEDIUM == "MEDIUM"
    assert RiskLevel.HIGH == "HIGH"
    assert RiskLevel.CRITICAL == "CRITICAL"

def test_device_type_enum():
    """Test device type enum"""
    # Check enum values
    assert DeviceType.CPU == "cpu"
    assert DeviceType.GPU == "gpu"
    assert DeviceType.ASIC == "asic"
    assert DeviceType.FPGA == "fpga"

def test_user_validation():
    """Test user model validation"""
    # Test required fields
    with pytest.raises(ValueError):
        User()
    
    # Test email validation
    with pytest.raises(ValueError):
        User(
            user_id="test_user",
            username="testuser",
            email="invalid_email",
            created_at=datetime.utcnow(),
            last_login=datetime.utcnow(),
            is_active=True,
            is_verified=True,
            preferences={}
        )

def test_activity_validation():
    """Test activity model validation"""
    # Test required fields
    with pytest.raises(ValueError):
        Activity()
    
    # Test duration validation
    with pytest.raises(ValueError):
        Activity(
            activity_id="test_activity",
            user_id="test_user",
            activity_type=ActivityType.MINING,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() - timedelta(hours=1),  # End before start
            duration=3600,
            status="completed",
            performance_metrics={},
            device_info={},
            network_info={}
        )

def test_reward_validation():
    """Test reward model validation"""
    # Test required fields
    with pytest.raises(ValueError):
        Reward()
    
    # Test amount validation
    with pytest.raises(ValueError):
        Reward(
            reward_id="test_reward",
            user_id="test_user",
            activity_id="test_activity",
            reward_type=RewardType.MINING,
            amount=-100.0,  # Negative amount
            currency="BTC",
            timestamp=datetime.utcnow(),
            status="pending",
            multipliers={},
            breakdown={}
        ) 