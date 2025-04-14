import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
from sqlalchemy.orm import Session

from database.models import (
    User,
    Activity,
    Reward,
    ActivityType,
    RewardType,
    RiskLevel,
    DeviceType
)
from database.operations import (
    create_user,
    get_user,
    update_user,
    delete_user,
    create_activity,
    get_activity,
    update_activity,
    delete_activity,
    create_reward,
    get_reward,
    update_reward,
    delete_reward,
    get_user_activities,
    get_user_rewards,
    get_activity_rewards,
    get_recent_activities,
    get_recent_rewards
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

def test_create_user(db_session: Session, test_user_data: Dict[str, Any]):
    """Test user creation"""
    # Create user
    user = create_user(db_session, test_user_data)
    
    # Check user was created
    assert user is not None
    assert user.user_id == test_user_data["user_id"]
    assert user.username == test_user_data["username"]
    assert user.email == test_user_data["email"]
    
    # Check user was saved to database
    db_user = get_user(db_session, test_user_data["user_id"])
    assert db_user is not None
    assert db_user.user_id == test_user_data["user_id"]

def test_get_user(db_session: Session, test_user_data: Dict[str, Any]):
    """Test user retrieval"""
    # Create user
    user = create_user(db_session, test_user_data)
    
    # Get user
    db_user = get_user(db_session, test_user_data["user_id"])
    
    # Check user was retrieved
    assert db_user is not None
    assert db_user.user_id == test_user_data["user_id"]
    assert db_user.username == test_user_data["username"]
    assert db_user.email == test_user_data["email"]

def test_update_user(db_session: Session, test_user_data: Dict[str, Any]):
    """Test user update"""
    # Create user
    user = create_user(db_session, test_user_data)
    
    # Update user
    update_data = {
        "username": "updated_user",
        "email": "updated@example.com",
        "is_active": False
    }
    updated_user = update_user(db_session, test_user_data["user_id"], update_data)
    
    # Check user was updated
    assert updated_user is not None
    assert updated_user.username == update_data["username"]
    assert updated_user.email == update_data["email"]
    assert updated_user.is_active == update_data["is_active"]
    
    # Check user was saved to database
    db_user = get_user(db_session, test_user_data["user_id"])
    assert db_user is not None
    assert db_user.username == update_data["username"]
    assert db_user.email == update_data["email"]
    assert db_user.is_active == update_data["is_active"]

def test_delete_user(db_session: Session, test_user_data: Dict[str, Any]):
    """Test user deletion"""
    # Create user
    user = create_user(db_session, test_user_data)
    
    # Delete user
    delete_user(db_session, test_user_data["user_id"])
    
    # Check user was deleted
    db_user = get_user(db_session, test_user_data["user_id"])
    assert db_user is None

def test_create_activity(db_session: Session, test_user_data: Dict[str, Any], test_activity_data: Dict[str, Any]):
    """Test activity creation"""
    # Create user
    user = create_user(db_session, test_user_data)
    
    # Create activity
    activity = create_activity(db_session, test_activity_data)
    
    # Check activity was created
    assert activity is not None
    assert activity.activity_id == test_activity_data["activity_id"]
    assert activity.user_id == test_activity_data["user_id"]
    assert activity.activity_type == test_activity_data["activity_type"]
    
    # Check activity was saved to database
    db_activity = get_activity(db_session, test_activity_data["activity_id"])
    assert db_activity is not None
    assert db_activity.activity_id == test_activity_data["activity_id"]

def test_get_activity(db_session: Session, test_user_data: Dict[str, Any], test_activity_data: Dict[str, Any]):
    """Test activity retrieval"""
    # Create user and activity
    user = create_user(db_session, test_user_data)
    activity = create_activity(db_session, test_activity_data)
    
    # Get activity
    db_activity = get_activity(db_session, test_activity_data["activity_id"])
    
    # Check activity was retrieved
    assert db_activity is not None
    assert db_activity.activity_id == test_activity_data["activity_id"]
    assert db_activity.user_id == test_activity_data["user_id"]
    assert db_activity.activity_type == test_activity_data["activity_type"]

def test_update_activity(db_session: Session, test_user_data: Dict[str, Any], test_activity_data: Dict[str, Any]):
    """Test activity update"""
    # Create user and activity
    user = create_user(db_session, test_user_data)
    activity = create_activity(db_session, test_activity_data)
    
    # Update activity
    update_data = {
        "status": "failed",
        "performance_metrics": {
            "hash_rate": 90.0,
            "power_efficiency": 0.75,
            "temperature": 82.0,
            "overall_efficiency": 0.78
        }
    }
    updated_activity = update_activity(db_session, test_activity_data["activity_id"], update_data)
    
    # Check activity was updated
    assert updated_activity is not None
    assert updated_activity.status == update_data["status"]
    assert updated_activity.performance_metrics == update_data["performance_metrics"]
    
    # Check activity was saved to database
    db_activity = get_activity(db_session, test_activity_data["activity_id"])
    assert db_activity is not None
    assert db_activity.status == update_data["status"]
    assert db_activity.performance_metrics == update_data["performance_metrics"]

def test_delete_activity(db_session: Session, test_user_data: Dict[str, Any], test_activity_data: Dict[str, Any]):
    """Test activity deletion"""
    # Create user and activity
    user = create_user(db_session, test_user_data)
    activity = create_activity(db_session, test_activity_data)
    
    # Delete activity
    delete_activity(db_session, test_activity_data["activity_id"])
    
    # Check activity was deleted
    db_activity = get_activity(db_session, test_activity_data["activity_id"])
    assert db_activity is None

def test_create_reward(db_session: Session, test_user_data: Dict[str, Any], test_activity_data: Dict[str, Any], test_reward_data: Dict[str, Any]):
    """Test reward creation"""
    # Create user, activity, and reward
    user = create_user(db_session, test_user_data)
    activity = create_activity(db_session, test_activity_data)
    reward = create_reward(db_session, test_reward_data)
    
    # Check reward was created
    assert reward is not None
    assert reward.reward_id == test_reward_data["reward_id"]
    assert reward.user_id == test_reward_data["user_id"]
    assert reward.activity_id == test_reward_data["activity_id"]
    assert reward.reward_type == test_reward_data["reward_type"]
    
    # Check reward was saved to database
    db_reward = get_reward(db_session, test_reward_data["reward_id"])
    assert db_reward is not None
    assert db_reward.reward_id == test_reward_data["reward_id"]

def test_get_reward(db_session: Session, test_user_data: Dict[str, Any], test_activity_data: Dict[str, Any], test_reward_data: Dict[str, Any]):
    """Test reward retrieval"""
    # Create user, activity, and reward
    user = create_user(db_session, test_user_data)
    activity = create_activity(db_session, test_activity_data)
    reward = create_reward(db_session, test_reward_data)
    
    # Get reward
    db_reward = get_reward(db_session, test_reward_data["reward_id"])
    
    # Check reward was retrieved
    assert db_reward is not None
    assert db_reward.reward_id == test_reward_data["reward_id"]
    assert db_reward.user_id == test_reward_data["user_id"]
    assert db_reward.activity_id == test_reward_data["activity_id"]
    assert db_reward.reward_type == test_reward_data["reward_type"]

def test_update_reward(db_session: Session, test_user_data: Dict[str, Any], test_activity_data: Dict[str, Any], test_reward_data: Dict[str, Any]):
    """Test reward update"""
    # Create user, activity, and reward
    user = create_user(db_session, test_user_data)
    activity = create_activity(db_session, test_activity_data)
    reward = create_reward(db_session, test_reward_data)
    
    # Update reward
    update_data = {
        "status": "completed",
        "amount": 110.0,
        "breakdown": {
            "base_reward": 85.0,
            "efficiency_bonus": 17.0,
            "consistency_bonus": 9.3,
            "loyalty_bonus": 4.4,
            "activity_bonus": 9.3
        }
    }
    updated_reward = update_reward(db_session, test_reward_data["reward_id"], update_data)
    
    # Check reward was updated
    assert updated_reward is not None
    assert updated_reward.status == update_data["status"]
    assert updated_reward.amount == update_data["amount"]
    assert updated_reward.breakdown == update_data["breakdown"]
    
    # Check reward was saved to database
    db_reward = get_reward(db_session, test_reward_data["reward_id"])
    assert db_reward is not None
    assert db_reward.status == update_data["status"]
    assert db_reward.amount == update_data["amount"]
    assert db_reward.breakdown == update_data["breakdown"]

def test_delete_reward(db_session: Session, test_user_data: Dict[str, Any], test_activity_data: Dict[str, Any], test_reward_data: Dict[str, Any]):
    """Test reward deletion"""
    # Create user, activity, and reward
    user = create_user(db_session, test_user_data)
    activity = create_activity(db_session, test_activity_data)
    reward = create_reward(db_session, test_reward_data)
    
    # Delete reward
    delete_reward(db_session, test_reward_data["reward_id"])
    
    # Check reward was deleted
    db_reward = get_reward(db_session, test_reward_data["reward_id"])
    assert db_reward is None

def test_get_user_activities(db_session: Session, test_user_data: Dict[str, Any], test_activity_data: Dict[str, Any]):
    """Test getting user activities"""
    # Create user and activity
    user = create_user(db_session, test_user_data)
    activity = create_activity(db_session, test_activity_data)
    
    # Create additional activity
    additional_activity_data = test_activity_data.copy()
    additional_activity_data["activity_id"] = "test_activity2"
    additional_activity_data["activity_type"] = ActivityType.STAKING
    additional_activity = create_activity(db_session, additional_activity_data)
    
    # Get user activities
    activities = get_user_activities(db_session, test_user_data["user_id"])
    
    # Check activities were retrieved
    assert len(activities) == 2
    assert any(a.activity_id == test_activity_data["activity_id"] for a in activities)
    assert any(a.activity_id == additional_activity_data["activity_id"] for a in activities)

def test_get_user_rewards(db_session: Session, test_user_data: Dict[str, Any], test_activity_data: Dict[str, Any], test_reward_data: Dict[str, Any]):
    """Test getting user rewards"""
    # Create user, activity, and reward
    user = create_user(db_session, test_user_data)
    activity = create_activity(db_session, test_activity_data)
    reward = create_reward(db_session, test_reward_data)
    
    # Create additional reward
    additional_reward_data = test_reward_data.copy()
    additional_reward_data["reward_id"] = "test_reward2"
    additional_reward_data["activity_id"] = "test_activity2"
    additional_reward_data["reward_type"] = RewardType.STAKING
    additional_reward = create_reward(db_session, additional_reward_data)
    
    # Get user rewards
    rewards = get_user_rewards(db_session, test_user_data["user_id"])
    
    # Check rewards were retrieved
    assert len(rewards) == 2
    assert any(r.reward_id == test_reward_data["reward_id"] for r in rewards)
    assert any(r.reward_id == additional_reward_data["reward_id"] for r in rewards)

def test_get_activity_rewards(db_session: Session, test_user_data: Dict[str, Any], test_activity_data: Dict[str, Any], test_reward_data: Dict[str, Any]):
    """Test getting activity rewards"""
    # Create user, activity, and reward
    user = create_user(db_session, test_user_data)
    activity = create_activity(db_session, test_activity_data)
    reward = create_reward(db_session, test_reward_data)
    
    # Create additional reward
    additional_reward_data = test_reward_data.copy()
    additional_reward_data["reward_id"] = "test_reward2"
    additional_reward_data["reward_type"] = RewardType.STAKING
    additional_reward = create_reward(db_session, additional_reward_data)
    
    # Get activity rewards
    rewards = get_activity_rewards(db_session, test_activity_data["activity_id"])
    
    # Check rewards were retrieved
    assert len(rewards) == 1
    assert rewards[0].reward_id == test_reward_data["reward_id"]

def test_get_recent_activities(db_session: Session, test_user_data: Dict[str, Any], test_activity_data: Dict[str, Any]):
    """Test getting recent activities"""
    # Create user and activity
    user = create_user(db_session, test_user_data)
    activity = create_activity(db_session, test_activity_data)
    
    # Create additional activity
    additional_activity_data = test_activity_data.copy()
    additional_activity_data["activity_id"] = "test_activity2"
    additional_activity_data["activity_type"] = ActivityType.STAKING
    additional_activity_data["start_time"] = datetime.utcnow() - timedelta(days=1)
    additional_activity_data["end_time"] = datetime.utcnow() - timedelta(days=1) + timedelta(hours=1)
    additional_activity = create_activity(db_session, additional_activity_data)
    
    # Get recent activities
    activities = get_recent_activities(db_session, limit=1)
    
    # Check activities were retrieved
    assert len(activities) == 1
    assert activities[0].activity_id == test_activity_data["activity_id"]

def test_get_recent_rewards(db_session: Session, test_user_data: Dict[str, Any], test_activity_data: Dict[str, Any], test_reward_data: Dict[str, Any]):
    """Test getting recent rewards"""
    # Create user, activity, and reward
    user = create_user(db_session, test_user_data)
    activity = create_activity(db_session, test_activity_data)
    reward = create_reward(db_session, test_reward_data)
    
    # Create additional reward
    additional_reward_data = test_reward_data.copy()
    additional_reward_data["reward_id"] = "test_reward2"
    additional_reward_data["activity_id"] = "test_activity2"
    additional_reward_data["reward_type"] = RewardType.STAKING
    additional_reward_data["timestamp"] = datetime.utcnow() - timedelta(days=1)
    additional_reward = create_reward(db_session, additional_reward_data)
    
    # Get recent rewards
    rewards = get_recent_rewards(db_session, limit=1)
    
    # Check rewards were retrieved
    assert len(rewards) == 1
    assert rewards[0].reward_id == test_reward_data["reward_id"] 