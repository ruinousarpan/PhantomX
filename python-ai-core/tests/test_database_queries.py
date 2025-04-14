import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_

from database.models import User, Activity, Reward, RiskAssessment, ActivityType, RewardType, RiskLevel
from database.exceptions import DatabaseError

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
        "activity_type": ActivityType.MINING,
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
        "activity_type": ActivityType.STAKING,
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
        "activity_type": ActivityType.TRADING,
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
        "reward_type": RewardType.MINING,
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
        "reward_type": RewardType.STAKING,
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
        "reward_type": RewardType.TRADING,
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

@pytest.fixture
def test_risk_assessment_data() -> Dict[str, Any]:
    """Create test risk assessment data"""
    return {
        "assessment_id": "test_assessment",
        "user_id": "test_user",
        "activity_id": "test_mining_activity",
        "activity_type": ActivityType.MINING,
        "timestamp": datetime.utcnow(),
        "risk_score": 0.75,
        "risk_level": RiskLevel.HIGH,
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

def test_get_user_by_id(db_session: Session, test_user_data: Dict[str, Any]):
    """Test getting user by ID"""
    # Create user
    user = User(**test_user_data)
    db_session.add(user)
    db_session.commit()

    # Get user by ID
    retrieved_user = db_session.query(User).filter_by(user_id=test_user_data["user_id"]).first()

    # Check user data
    assert retrieved_user is not None
    assert retrieved_user.user_id == test_user_data["user_id"]
    assert retrieved_user.username == test_user_data["username"]
    assert retrieved_user.email == test_user_data["email"]

def test_get_user_by_email(db_session: Session, test_user_data: Dict[str, Any]):
    """Test getting user by email"""
    # Create user
    user = User(**test_user_data)
    db_session.add(user)
    db_session.commit()

    # Get user by email
    retrieved_user = db_session.query(User).filter_by(email=test_user_data["email"]).first()

    # Check user data
    assert retrieved_user is not None
    assert retrieved_user.user_id == test_user_data["user_id"]
    assert retrieved_user.username == test_user_data["username"]
    assert retrieved_user.email == test_user_data["email"]

def test_get_user_by_username(db_session: Session, test_user_data: Dict[str, Any]):
    """Test getting user by username"""
    # Create user
    user = User(**test_user_data)
    db_session.add(user)
    db_session.commit()

    # Get user by username
    retrieved_user = db_session.query(User).filter_by(username=test_user_data["username"]).first()

    # Check user data
    assert retrieved_user is not None
    assert retrieved_user.user_id == test_user_data["user_id"]
    assert retrieved_user.username == test_user_data["username"]
    assert retrieved_user.email == test_user_data["email"]

def test_get_activity_by_id(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any]):
    """Test getting activity by ID"""
    # Create user and activity
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Get activity by ID
    retrieved_activity = db_session.query(Activity).filter_by(activity_id=test_mining_activity_data["activity_id"]).first()

    # Check activity data
    assert retrieved_activity is not None
    assert retrieved_activity.activity_id == test_mining_activity_data["activity_id"]
    assert retrieved_activity.user_id == test_mining_activity_data["user_id"]
    assert retrieved_activity.activity_type == test_mining_activity_data["activity_type"]

def test_get_activity_by_user_id(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_staking_activity_data: Dict[str, Any]):
    """Test getting activities by user ID"""
    # Create user and activities
    user = User(**test_user_data)
    mining_activity = Activity(**test_mining_activity_data)
    staking_activity = Activity(**test_staking_activity_data)
    db_session.add(user)
    db_session.add(mining_activity)
    db_session.add(staking_activity)
    db_session.commit()

    # Get activities by user ID
    retrieved_activities = db_session.query(Activity).filter_by(user_id=test_user_data["user_id"]).all()

    # Check activities data
    assert len(retrieved_activities) == 2
    activity_types = [activity.activity_type for activity in retrieved_activities]
    assert ActivityType.MINING in activity_types
    assert ActivityType.STAKING in activity_types

def test_get_activity_by_type(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_staking_activity_data: Dict[str, Any]):
    """Test getting activities by type"""
    # Create user and activities
    user = User(**test_user_data)
    mining_activity = Activity(**test_mining_activity_data)
    staking_activity = Activity(**test_staking_activity_data)
    db_session.add(user)
    db_session.add(mining_activity)
    db_session.add(staking_activity)
    db_session.commit()

    # Get activities by type
    retrieved_activities = db_session.query(Activity).filter_by(activity_type=ActivityType.MINING).all()

    # Check activities data
    assert len(retrieved_activities) == 1
    assert retrieved_activities[0].activity_id == test_mining_activity_data["activity_id"]
    assert retrieved_activities[0].activity_type == ActivityType.MINING

def test_get_activity_by_date_range(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_staking_activity_data: Dict[str, Any]):
    """Test getting activities by date range"""
    # Create user and activities
    user = User(**test_user_data)
    mining_activity = Activity(**test_mining_activity_data)
    staking_activity = Activity(**test_staking_activity_data)
    db_session.add(user)
    db_session.add(mining_activity)
    db_session.add(staking_activity)
    db_session.commit()

    # Get activities by date range
    start_time = datetime.utcnow() - timedelta(hours=2)
    end_time = datetime.utcnow() + timedelta(hours=1)
    retrieved_activities = db_session.query(Activity).filter(
        Activity.start_time >= start_time,
        Activity.end_time <= end_time
    ).all()

    # Check activities data
    assert len(retrieved_activities) == 2
    activity_types = [activity.activity_type for activity in retrieved_activities]
    assert ActivityType.MINING in activity_types
    assert ActivityType.STAKING in activity_types

def test_get_activity_by_status(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any]):
    """Test getting activities by status"""
    # Create user and activity
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Get activities by status
    retrieved_activities = db_session.query(Activity).filter_by(status="completed").all()

    # Check activities data
    assert len(retrieved_activities) == 1
    assert retrieved_activities[0].activity_id == test_mining_activity_data["activity_id"]
    assert retrieved_activities[0].status == "completed"

def test_get_reward_by_id(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any]):
    """Test getting reward by ID"""
    # Create user, activity, and reward
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    reward = Reward(**test_mining_reward_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.add(reward)
    db_session.commit()

    # Get reward by ID
    retrieved_reward = db_session.query(Reward).filter_by(reward_id=test_mining_reward_data["reward_id"]).first()

    # Check reward data
    assert retrieved_reward is not None
    assert retrieved_reward.reward_id == test_mining_reward_data["reward_id"]
    assert retrieved_reward.user_id == test_mining_reward_data["user_id"]
    assert retrieved_reward.activity_id == test_mining_reward_data["activity_id"]
    assert retrieved_reward.reward_type == test_mining_reward_data["reward_type"]

def test_get_reward_by_user_id(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any], test_staking_reward_data: Dict[str, Any]):
    """Test getting rewards by user ID"""
    # Create user, activity, and rewards
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    mining_reward = Reward(**test_mining_reward_data)
    staking_reward = Reward(**test_staking_reward_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.add(mining_reward)
    db_session.add(staking_reward)
    db_session.commit()

    # Get rewards by user ID
    retrieved_rewards = db_session.query(Reward).filter_by(user_id=test_user_data["user_id"]).all()

    # Check rewards data
    assert len(retrieved_rewards) == 2
    reward_types = [reward.reward_type for reward in retrieved_rewards]
    assert RewardType.MINING in reward_types
    assert RewardType.STAKING in reward_types

def test_get_reward_by_activity_id(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any]):
    """Test getting rewards by activity ID"""
    # Create user, activity, and reward
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    reward = Reward(**test_mining_reward_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.add(reward)
    db_session.commit()

    # Get rewards by activity ID
    retrieved_rewards = db_session.query(Reward).filter_by(activity_id=test_mining_activity_data["activity_id"]).all()

    # Check rewards data
    assert len(retrieved_rewards) == 1
    assert retrieved_rewards[0].reward_id == test_mining_reward_data["reward_id"]
    assert retrieved_rewards[0].activity_id == test_mining_activity_data["activity_id"]

def test_get_reward_by_type(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any], test_staking_reward_data: Dict[str, Any]):
    """Test getting rewards by type"""
    # Create user, activity, and rewards
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    mining_reward = Reward(**test_mining_reward_data)
    staking_reward = Reward(**test_staking_reward_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.add(mining_reward)
    db_session.add(staking_reward)
    db_session.commit()

    # Get rewards by type
    retrieved_rewards = db_session.query(Reward).filter_by(reward_type=RewardType.MINING).all()

    # Check rewards data
    assert len(retrieved_rewards) == 1
    assert retrieved_rewards[0].reward_id == test_mining_reward_data["reward_id"]
    assert retrieved_rewards[0].reward_type == RewardType.MINING

def test_get_reward_by_date_range(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any], test_staking_reward_data: Dict[str, Any]):
    """Test getting rewards by date range"""
    # Create user, activity, and rewards
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    mining_reward = Reward(**test_mining_reward_data)
    staking_reward = Reward(**test_staking_reward_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.add(mining_reward)
    db_session.add(staking_reward)
    db_session.commit()

    # Get rewards by date range
    start_time = datetime.utcnow() - timedelta(hours=1)
    end_time = datetime.utcnow() + timedelta(hours=1)
    retrieved_rewards = db_session.query(Reward).filter(
        Reward.timestamp >= start_time,
        Reward.timestamp <= end_time
    ).all()

    # Check rewards data
    assert len(retrieved_rewards) == 2
    reward_types = [reward.reward_type for reward in retrieved_rewards]
    assert RewardType.MINING in reward_types
    assert RewardType.STAKING in reward_types

def test_get_reward_by_status(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any]):
    """Test getting rewards by status"""
    # Create user, activity, and reward
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    reward = Reward(**test_mining_reward_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.add(reward)
    db_session.commit()

    # Get rewards by status
    retrieved_rewards = db_session.query(Reward).filter_by(status="pending").all()

    # Check rewards data
    assert len(retrieved_rewards) == 1
    assert retrieved_rewards[0].reward_id == test_mining_reward_data["reward_id"]
    assert retrieved_rewards[0].status == "pending"

def test_get_risk_assessment_by_id(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_risk_assessment_data: Dict[str, Any]):
    """Test getting risk assessment by ID"""
    # Create user, activity, and risk assessment
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    assessment = RiskAssessment(**test_risk_assessment_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.add(assessment)
    db_session.commit()

    # Get risk assessment by ID
    retrieved_assessment = db_session.query(RiskAssessment).filter_by(assessment_id=test_risk_assessment_data["assessment_id"]).first()

    # Check risk assessment data
    assert retrieved_assessment is not None
    assert retrieved_assessment.assessment_id == test_risk_assessment_data["assessment_id"]
    assert retrieved_assessment.user_id == test_risk_assessment_data["user_id"]
    assert retrieved_assessment.activity_id == test_risk_assessment_data["activity_id"]
    assert retrieved_assessment.activity_type == test_risk_assessment_data["activity_type"]

def test_get_risk_assessment_by_user_id(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_risk_assessment_data: Dict[str, Any]):
    """Test getting risk assessments by user ID"""
    # Create user, activity, and risk assessment
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    assessment = RiskAssessment(**test_risk_assessment_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.add(assessment)
    db_session.commit()

    # Get risk assessments by user ID
    retrieved_assessments = db_session.query(RiskAssessment).filter_by(user_id=test_user_data["user_id"]).all()

    # Check risk assessments data
    assert len(retrieved_assessments) == 1
    assert retrieved_assessments[0].assessment_id == test_risk_assessment_data["assessment_id"]
    assert retrieved_assessments[0].user_id == test_user_data["user_id"]

def test_get_risk_assessment_by_activity_id(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_risk_assessment_data: Dict[str, Any]):
    """Test getting risk assessments by activity ID"""
    # Create user, activity, and risk assessment
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    assessment = RiskAssessment(**test_risk_assessment_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.add(assessment)
    db_session.commit()

    # Get risk assessments by activity ID
    retrieved_assessments = db_session.query(RiskAssessment).filter_by(activity_id=test_mining_activity_data["activity_id"]).all()

    # Check risk assessments data
    assert len(retrieved_assessments) == 1
    assert retrieved_assessments[0].assessment_id == test_risk_assessment_data["assessment_id"]
    assert retrieved_assessments[0].activity_id == test_mining_activity_data["activity_id"]

def test_get_risk_assessment_by_activity_type(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_risk_assessment_data: Dict[str, Any]):
    """Test getting risk assessments by activity type"""
    # Create user, activity, and risk assessment
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    assessment = RiskAssessment(**test_risk_assessment_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.add(assessment)
    db_session.commit()

    # Get risk assessments by activity type
    retrieved_assessments = db_session.query(RiskAssessment).filter_by(activity_type=ActivityType.MINING).all()

    # Check risk assessments data
    assert len(retrieved_assessments) == 1
    assert retrieved_assessments[0].assessment_id == test_risk_assessment_data["assessment_id"]
    assert retrieved_assessments[0].activity_type == ActivityType.MINING

def test_get_risk_assessment_by_risk_level(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_risk_assessment_data: Dict[str, Any]):
    """Test getting risk assessments by risk level"""
    # Create user, activity, and risk assessment
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    assessment = RiskAssessment(**test_risk_assessment_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.add(assessment)
    db_session.commit()

    # Get risk assessments by risk level
    retrieved_assessments = db_session.query(RiskAssessment).filter_by(risk_level=RiskLevel.HIGH).all()

    # Check risk assessments data
    assert len(retrieved_assessments) == 1
    assert retrieved_assessments[0].assessment_id == test_risk_assessment_data["assessment_id"]
    assert retrieved_assessments[0].risk_level == RiskLevel.HIGH

def test_get_recent_activities(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_staking_activity_data: Dict[str, Any], test_trading_activity_data: Dict[str, Any]):
    """Test getting recent activities"""
    # Create user and activities
    user = User(**test_user_data)
    mining_activity = Activity(**test_mining_activity_data)
    staking_activity = Activity(**test_staking_activity_data)
    trading_activity = Activity(**test_trading_activity_data)
    db_session.add(user)
    db_session.add(mining_activity)
    db_session.add(staking_activity)
    db_session.add(trading_activity)
    db_session.commit()

    # Get recent activities
    retrieved_activities = db_session.query(Activity).order_by(desc(Activity.start_time)).limit(2).all()

    # Check activities data
    assert len(retrieved_activities) == 2
    assert retrieved_activities[0].activity_id == test_mining_activity_data["activity_id"]
    assert retrieved_activities[1].activity_id == test_trading_activity_data["activity_id"]

def test_get_recent_rewards(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any], test_staking_reward_data: Dict[str, Any], test_trading_reward_data: Dict[str, Any]):
    """Test getting recent rewards"""
    # Create user, activity, and rewards
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    mining_reward = Reward(**test_mining_reward_data)
    staking_reward = Reward(**test_staking_reward_data)
    trading_reward = Reward(**test_trading_reward_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.add(mining_reward)
    db_session.add(staking_reward)
    db_session.add(trading_reward)
    db_session.commit()

    # Get recent rewards
    retrieved_rewards = db_session.query(Reward).order_by(desc(Reward.timestamp)).limit(2).all()

    # Check rewards data
    assert len(retrieved_rewards) == 2
    assert retrieved_rewards[0].reward_id == test_trading_reward_data["reward_id"]
    assert retrieved_rewards[1].reward_id == test_staking_reward_data["reward_id"]

def test_get_user_statistics(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_staking_activity_data: Dict[str, Any], test_trading_activity_data: Dict[str, Any]):
    """Test getting user statistics"""
    # Create user and activities
    user = User(**test_user_data)
    mining_activity = Activity(**test_mining_activity_data)
    staking_activity = Activity(**test_staking_activity_data)
    trading_activity = Activity(**test_trading_activity_data)
    db_session.add(user)
    db_session.add(mining_activity)
    db_session.add(staking_activity)
    db_session.add(trading_activity)
    db_session.commit()

    # Get user statistics
    activity_count = db_session.query(func.count(Activity.activity_id)).filter_by(user_id=test_user_data["user_id"]).scalar()
    mining_count = db_session.query(func.count(Activity.activity_id)).filter_by(user_id=test_user_data["user_id"], activity_type=ActivityType.MINING).scalar()
    staking_count = db_session.query(func.count(Activity.activity_id)).filter_by(user_id=test_user_data["user_id"], activity_type=ActivityType.STAKING).scalar()
    trading_count = db_session.query(func.count(Activity.activity_id)).filter_by(user_id=test_user_data["user_id"], activity_type=ActivityType.TRADING).scalar()

    # Check user statistics
    assert activity_count == 3
    assert mining_count == 1
    assert staking_count == 1
    assert trading_count == 1

def test_get_activity_statistics(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_staking_activity_data: Dict[str, Any], test_trading_activity_data: Dict[str, Any]):
    """Test getting activity statistics"""
    # Create user and activities
    user = User(**test_user_data)
    mining_activity = Activity(**test_mining_activity_data)
    staking_activity = Activity(**test_staking_activity_data)
    trading_activity = Activity(**test_trading_activity_data)
    db_session.add(user)
    db_session.add(mining_activity)
    db_session.add(staking_activity)
    db_session.add(trading_activity)
    db_session.commit()

    # Get activity statistics
    activity_count = db_session.query(func.count(Activity.activity_id)).scalar()
    mining_count = db_session.query(func.count(Activity.activity_id)).filter_by(activity_type=ActivityType.MINING).scalar()
    staking_count = db_session.query(func.count(Activity.activity_id)).filter_by(activity_type=ActivityType.STAKING).scalar()
    trading_count = db_session.query(func.count(Activity.activity_id)).filter_by(activity_type=ActivityType.TRADING).scalar()

    # Check activity statistics
    assert activity_count == 3
    assert mining_count == 1
    assert staking_count == 1
    assert trading_count == 1

def test_get_reward_statistics(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any], test_staking_reward_data: Dict[str, Any], test_trading_reward_data: Dict[str, Any]):
    """Test getting reward statistics"""
    # Create user, activity, and rewards
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    mining_reward = Reward(**test_mining_reward_data)
    staking_reward = Reward(**test_staking_reward_data)
    trading_reward = Reward(**test_trading_reward_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.add(mining_reward)
    db_session.add(staking_reward)
    db_session.add(trading_reward)
    db_session.commit()

    # Get reward statistics
    reward_count = db_session.query(func.count(Reward.reward_id)).scalar()
    mining_count = db_session.query(func.count(Reward.reward_id)).filter_by(reward_type=RewardType.MINING).scalar()
    staking_count = db_session.query(func.count(Reward.reward_id)).filter_by(reward_type=RewardType.STAKING).scalar()
    trading_count = db_session.query(func.count(Reward.reward_id)).filter_by(reward_type=RewardType.TRADING).scalar()

    # Check reward statistics
    assert reward_count == 3
    assert mining_count == 1
    assert staking_count == 1
    assert trading_count == 1

def test_get_mining_efficiency(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any]):
    """Test getting mining efficiency"""
    # Create user and activity
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Get mining efficiency
    retrieved_activity = db_session.query(Activity).filter_by(activity_id=test_mining_activity_data["activity_id"]).first()

    # Check mining efficiency
    assert retrieved_activity is not None
    assert retrieved_activity.efficiency_score == test_mining_activity_data["efficiency_score"]
    assert retrieved_activity.performance_metrics["efficiency"] == test_mining_activity_data["performance_metrics"]["efficiency"]

def test_get_staking_uptime(db_session: Session, test_user_data: Dict[str, Any], test_staking_activity_data: Dict[str, Any]):
    """Test getting staking uptime"""
    # Create user and activity
    user = User(**test_user_data)
    activity = Activity(**test_staking_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Get staking uptime
    retrieved_activity = db_session.query(Activity).filter_by(activity_id=test_staking_activity_data["activity_id"]).first()

    # Check staking uptime
    assert retrieved_activity is not None
    assert retrieved_activity.uptime == test_staking_activity_data["uptime"]
    assert retrieved_activity.performance_metrics["uptime"] == test_staking_activity_data["performance_metrics"]["uptime"]

def test_get_trading_performance(db_session: Session, test_user_data: Dict[str, Any], test_trading_activity_data: Dict[str, Any]):
    """Test getting trading performance"""
    # Create user and activity
    user = User(**test_user_data)
    activity = Activity(**test_trading_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Get trading performance
    retrieved_activity = db_session.query(Activity).filter_by(activity_id=test_trading_activity_data["activity_id"]).first()

    # Check trading performance
    assert retrieved_activity is not None
    assert retrieved_activity.performance_metrics["win_rate"] == test_trading_activity_data["performance_metrics"]["win_rate"]
    assert retrieved_activity.performance_metrics["profit_factor"] == test_trading_activity_data["performance_metrics"]["profit_factor"]
    assert retrieved_activity.performance_metrics["sharpe_ratio"] == test_trading_activity_data["performance_metrics"]["sharpe_ratio"]
    assert retrieved_activity.performance_metrics["max_drawdown"] == test_trading_activity_data["performance_metrics"]["max_drawdown"]

def test_get_reward_amount(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any], test_staking_reward_data: Dict[str, Any], test_trading_reward_data: Dict[str, Any]):
    """Test getting reward amount"""
    # Create user, activity, and rewards
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    mining_reward = Reward(**test_mining_reward_data)
    staking_reward = Reward(**test_staking_reward_data)
    trading_reward = Reward(**test_trading_reward_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.add(mining_reward)
    db_session.add(staking_reward)
    db_session.add(trading_reward)
    db_session.commit()

    # Get reward amount
    btc_amount = db_session.query(func.sum(Reward.amount)).filter_by(currency="BTC").scalar()
    eth_amount = db_session.query(func.sum(Reward.amount)).filter_by(currency="ETH").scalar()
    usdt_amount = db_session.query(func.sum(Reward.amount)).filter_by(currency="USDT").scalar()

    # Check reward amount
    assert btc_amount == test_mining_reward_data["amount"]
    assert eth_amount == test_staking_reward_data["amount"]
    assert usdt_amount == test_trading_reward_data["amount"]

def test_get_activity_duration(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_staking_activity_data: Dict[str, Any], test_trading_activity_data: Dict[str, Any]):
    """Test getting activity duration"""
    # Create user and activities
    user = User(**test_user_data)
    mining_activity = Activity(**test_mining_activity_data)
    staking_activity = Activity(**test_staking_activity_data)
    trading_activity = Activity(**test_trading_activity_data)
    db_session.add(user)
    db_session.add(mining_activity)
    db_session.add(staking_activity)
    db_session.add(trading_activity)
    db_session.commit()

    # Get activity duration
    mining_duration = (test_mining_activity_data["end_time"] - test_mining_activity_data["start_time"]).total_seconds() / 3600
    staking_duration = (test_staking_activity_data["end_time"] - test_staking_activity_data["start_time"]).total_seconds() / 3600
    trading_duration = (test_trading_activity_data["end_time"] - test_trading_activity_data["start_time"]).total_seconds() / 3600

    # Check activity duration
    assert mining_duration == 1.0
    assert staking_duration == 24.0
    assert trading_duration == 2.0

def test_get_reward_status(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any]):
    """Test getting reward status"""
    # Create user, activity, and reward
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    reward = Reward(**test_mining_reward_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.add(reward)
    db_session.commit()

    # Get reward status
    retrieved_reward = db_session.query(Reward).filter_by(reward_id=test_mining_reward_data["reward_id"]).first()

    # Check reward status
    assert retrieved_reward is not None
    assert retrieved_reward.status == test_mining_reward_data["status"]

def test_get_activities_with_rewards(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any]):
    """Test getting activities with rewards"""
    # Create user, activity, and reward
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    reward = Reward(**test_mining_reward_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.add(reward)
    db_session.commit()

    # Get activities with rewards
    retrieved_activities = db_session.query(Activity).join(Reward).all()

    # Check activities with rewards
    assert len(retrieved_activities) == 1
    assert retrieved_activities[0].activity_id == test_mining_activity_data["activity_id"]
    assert len(retrieved_activities[0].rewards) == 1
    assert retrieved_activities[0].rewards[0].reward_id == test_mining_reward_data["reward_id"]

def test_get_users_with_activities_and_rewards(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any]):
    """Test getting users with activities and rewards"""
    # Create user, activity, and reward
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    reward = Reward(**test_mining_reward_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.add(reward)
    db_session.commit()

    # Get users with activities and rewards
    retrieved_users = db_session.query(User).join(Activity).join(Reward).all()

    # Check users with activities and rewards
    assert len(retrieved_users) == 1
    assert retrieved_users[0].user_id == test_user_data["user_id"]
    assert len(retrieved_users[0].activities) == 1
    assert retrieved_users[0].activities[0].activity_id == test_mining_activity_data["activity_id"]
    assert len(retrieved_users[0].rewards) == 1
    assert retrieved_users[0].rewards[0].reward_id == test_mining_reward_data["reward_id"]

def test_count_activities_by_type(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_staking_activity_data: Dict[str, Any], test_trading_activity_data: Dict[str, Any]):
    """Test counting activities by type"""
    # Create user and activities
    user = User(**test_user_data)
    mining_activity = Activity(**test_mining_activity_data)
    staking_activity = Activity(**test_staking_activity_data)
    trading_activity = Activity(**test_trading_activity_data)
    db_session.add(user)
    db_session.add(mining_activity)
    db_session.add(staking_activity)
    db_session.add(trading_activity)
    db_session.commit()

    # Count activities by type
    activity_counts = db_session.query(
        Activity.activity_type,
        func.count(Activity.activity_id).label("count")
    ).group_by(Activity.activity_type).all()

    # Check activity counts
    assert len(activity_counts) == 3
    activity_type_counts = {activity_type: count for activity_type, count in activity_counts}
    assert activity_type_counts[ActivityType.MINING] == 1
    assert activity_type_counts[ActivityType.STAKING] == 1
    assert activity_type_counts[ActivityType.TRADING] == 1

def test_sum_reward_amounts_by_currency(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any], test_staking_reward_data: Dict[str, Any], test_trading_reward_data: Dict[str, Any]):
    """Test summing reward amounts by currency"""
    # Create user, activity, and rewards
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    mining_reward = Reward(**test_mining_reward_data)
    staking_reward = Reward(**test_staking_reward_data)
    trading_reward = Reward(**test_trading_reward_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.add(mining_reward)
    db_session.add(staking_reward)
    db_session.add(trading_reward)
    db_session.commit()

    # Sum reward amounts by currency
    reward_amounts = db_session.query(
        Reward.currency,
        func.sum(Reward.amount).label("total")
    ).group_by(Reward.currency).all()

    # Check reward amounts
    assert len(reward_amounts) == 3
    currency_amounts = {currency: total for currency, total in reward_amounts}
    assert currency_amounts["BTC"] == test_mining_reward_data["amount"]
    assert currency_amounts["ETH"] == test_staking_reward_data["amount"]
    assert currency_amounts["USDT"] == test_trading_reward_data["amount"] 