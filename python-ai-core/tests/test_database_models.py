import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from database.models import (
    User,
    Activity,
    Reward,
    RiskAssessment,
    ActivityType,
    RewardType,
    RiskLevel
)
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

def test_create_user(db_session: Session, test_user_data: Dict[str, Any]):
    """Test creating a user"""
    # Create user
    user = User(**test_user_data)
    db_session.add(user)
    db_session.commit()

    # Retrieve user
    retrieved_user = db_session.query(User).filter_by(user_id=test_user_data["user_id"]).first()

    # Check user data
    assert retrieved_user is not None
    assert retrieved_user.user_id == test_user_data["user_id"]
    assert retrieved_user.username == test_user_data["username"]
    assert retrieved_user.email == test_user_data["email"]
    assert retrieved_user.created_at == test_user_data["created_at"]
    assert retrieved_user.last_login == test_user_data["last_login"]
    assert retrieved_user.is_active == test_user_data["is_active"]
    assert retrieved_user.is_verified == test_user_data["is_verified"]
    assert retrieved_user.preferences == test_user_data["preferences"]

def test_create_mining_activity(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any]):
    """Test creating a mining activity"""
    # Create user
    user = User(**test_user_data)
    db_session.add(user)
    db_session.commit()

    # Create activity
    activity = Activity(**test_mining_activity_data)
    db_session.add(activity)
    db_session.commit()

    # Retrieve activity
    retrieved_activity = db_session.query(Activity).filter_by(activity_id=test_mining_activity_data["activity_id"]).first()

    # Check activity data
    assert retrieved_activity is not None
    assert retrieved_activity.activity_id == test_mining_activity_data["activity_id"]
    assert retrieved_activity.user_id == test_mining_activity_data["user_id"]
    assert retrieved_activity.activity_type == test_mining_activity_data["activity_type"]
    assert retrieved_activity.start_time == test_mining_activity_data["start_time"]
    assert retrieved_activity.end_time == test_mining_activity_data["end_time"]
    assert retrieved_activity.status == test_mining_activity_data["status"]
    assert retrieved_activity.device_type == test_mining_activity_data["device_type"]
    assert retrieved_activity.hash_rate == test_mining_activity_data["hash_rate"]
    assert retrieved_activity.power_usage == test_mining_activity_data["power_usage"]
    assert retrieved_activity.efficiency_score == test_mining_activity_data["efficiency_score"]
    assert retrieved_activity.performance_metrics == test_mining_activity_data["performance_metrics"]

def test_create_staking_activity(db_session: Session, test_user_data: Dict[str, Any], test_staking_activity_data: Dict[str, Any]):
    """Test creating a staking activity"""
    # Create user
    user = User(**test_user_data)
    db_session.add(user)
    db_session.commit()

    # Create activity
    activity = Activity(**test_staking_activity_data)
    db_session.add(activity)
    db_session.commit()

    # Retrieve activity
    retrieved_activity = db_session.query(Activity).filter_by(activity_id=test_staking_activity_data["activity_id"]).first()

    # Check activity data
    assert retrieved_activity is not None
    assert retrieved_activity.activity_id == test_staking_activity_data["activity_id"]
    assert retrieved_activity.user_id == test_staking_activity_data["user_id"]
    assert retrieved_activity.activity_type == test_staking_activity_data["activity_type"]
    assert retrieved_activity.start_time == test_staking_activity_data["start_time"]
    assert retrieved_activity.end_time == test_staking_activity_data["end_time"]
    assert retrieved_activity.status == test_staking_activity_data["status"]
    assert retrieved_activity.stake_amount == test_staking_activity_data["stake_amount"]
    assert retrieved_activity.validator_id == test_staking_activity_data["validator_id"]
    assert retrieved_activity.uptime == test_staking_activity_data["uptime"]
    assert retrieved_activity.performance_metrics == test_staking_activity_data["performance_metrics"]

def test_create_trading_activity(db_session: Session, test_user_data: Dict[str, Any], test_trading_activity_data: Dict[str, Any]):
    """Test creating a trading activity"""
    # Create user
    user = User(**test_user_data)
    db_session.add(user)
    db_session.commit()

    # Create activity
    activity = Activity(**test_trading_activity_data)
    db_session.add(activity)
    db_session.commit()

    # Retrieve activity
    retrieved_activity = db_session.query(Activity).filter_by(activity_id=test_trading_activity_data["activity_id"]).first()

    # Check activity data
    assert retrieved_activity is not None
    assert retrieved_activity.activity_id == test_trading_activity_data["activity_id"]
    assert retrieved_activity.user_id == test_trading_activity_data["user_id"]
    assert retrieved_activity.activity_type == test_trading_activity_data["activity_type"]
    assert retrieved_activity.start_time == test_trading_activity_data["start_time"]
    assert retrieved_activity.end_time == test_trading_activity_data["end_time"]
    assert retrieved_activity.status == test_trading_activity_data["status"]
    assert retrieved_activity.trading_pairs == test_trading_activity_data["trading_pairs"]
    assert retrieved_activity.position_size == test_trading_activity_data["position_size"]
    assert retrieved_activity.leverage == test_trading_activity_data["leverage"]
    assert retrieved_activity.performance_metrics == test_trading_activity_data["performance_metrics"]

def test_create_mining_reward(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any]):
    """Test creating a mining reward"""
    # Create user and activity
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Create reward
    reward = Reward(**test_mining_reward_data)
    db_session.add(reward)
    db_session.commit()

    # Retrieve reward
    retrieved_reward = db_session.query(Reward).filter_by(reward_id=test_mining_reward_data["reward_id"]).first()

    # Check reward data
    assert retrieved_reward is not None
    assert retrieved_reward.reward_id == test_mining_reward_data["reward_id"]
    assert retrieved_reward.user_id == test_mining_reward_data["user_id"]
    assert retrieved_reward.activity_id == test_mining_reward_data["activity_id"]
    assert retrieved_reward.reward_type == test_mining_reward_data["reward_type"]
    assert retrieved_reward.amount == test_mining_reward_data["amount"]
    assert retrieved_reward.currency == test_mining_reward_data["currency"]
    assert retrieved_reward.timestamp == test_mining_reward_data["timestamp"]
    assert retrieved_reward.status == test_mining_reward_data["status"]
    assert retrieved_reward.transaction_hash == test_mining_reward_data["transaction_hash"]
    assert retrieved_reward.metadata == test_mining_reward_data["metadata"]

def test_create_staking_reward(db_session: Session, test_user_data: Dict[str, Any], test_staking_activity_data: Dict[str, Any], test_staking_reward_data: Dict[str, Any]):
    """Test creating a staking reward"""
    # Create user and activity
    user = User(**test_user_data)
    activity = Activity(**test_staking_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Create reward
    reward = Reward(**test_staking_reward_data)
    db_session.add(reward)
    db_session.commit()

    # Retrieve reward
    retrieved_reward = db_session.query(Reward).filter_by(reward_id=test_staking_reward_data["reward_id"]).first()

    # Check reward data
    assert retrieved_reward is not None
    assert retrieved_reward.reward_id == test_staking_reward_data["reward_id"]
    assert retrieved_reward.user_id == test_staking_reward_data["user_id"]
    assert retrieved_reward.activity_id == test_staking_reward_data["activity_id"]
    assert retrieved_reward.reward_type == test_staking_reward_data["reward_type"]
    assert retrieved_reward.amount == test_staking_reward_data["amount"]
    assert retrieved_reward.currency == test_staking_reward_data["currency"]
    assert retrieved_reward.timestamp == test_staking_reward_data["timestamp"]
    assert retrieved_reward.status == test_staking_reward_data["status"]
    assert retrieved_reward.transaction_hash == test_staking_reward_data["transaction_hash"]
    assert retrieved_reward.metadata == test_staking_reward_data["metadata"]

def test_create_trading_reward(db_session: Session, test_user_data: Dict[str, Any], test_trading_activity_data: Dict[str, Any], test_trading_reward_data: Dict[str, Any]):
    """Test creating a trading reward"""
    # Create user and activity
    user = User(**test_user_data)
    activity = Activity(**test_trading_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Create reward
    reward = Reward(**test_trading_reward_data)
    db_session.add(reward)
    db_session.commit()

    # Retrieve reward
    retrieved_reward = db_session.query(Reward).filter_by(reward_id=test_trading_reward_data["reward_id"]).first()

    # Check reward data
    assert retrieved_reward is not None
    assert retrieved_reward.reward_id == test_trading_reward_data["reward_id"]
    assert retrieved_reward.user_id == test_trading_reward_data["user_id"]
    assert retrieved_reward.activity_id == test_trading_reward_data["activity_id"]
    assert retrieved_reward.reward_type == test_trading_reward_data["reward_type"]
    assert retrieved_reward.amount == test_trading_reward_data["amount"]
    assert retrieved_reward.currency == test_trading_reward_data["currency"]
    assert retrieved_reward.timestamp == test_trading_reward_data["timestamp"]
    assert retrieved_reward.status == test_trading_reward_data["status"]
    assert retrieved_reward.transaction_hash == test_trading_reward_data["transaction_hash"]
    assert retrieved_reward.metadata == test_trading_reward_data["metadata"]

def test_create_risk_assessment(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_risk_assessment_data: Dict[str, Any]):
    """Test creating a risk assessment"""
    # Create user and activity
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Create risk assessment
    assessment = RiskAssessment(**test_risk_assessment_data)
    db_session.add(assessment)
    db_session.commit()

    # Retrieve risk assessment
    retrieved_assessment = db_session.query(RiskAssessment).filter_by(assessment_id=test_risk_assessment_data["assessment_id"]).first()

    # Check risk assessment data
    assert retrieved_assessment is not None
    assert retrieved_assessment.assessment_id == test_risk_assessment_data["assessment_id"]
    assert retrieved_assessment.user_id == test_risk_assessment_data["user_id"]
    assert retrieved_assessment.activity_id == test_risk_assessment_data["activity_id"]
    assert retrieved_assessment.activity_type == test_risk_assessment_data["activity_type"]
    assert retrieved_assessment.timestamp == test_risk_assessment_data["timestamp"]
    assert retrieved_assessment.risk_score == test_risk_assessment_data["risk_score"]
    assert retrieved_assessment.risk_level == test_risk_assessment_data["risk_level"]
    assert retrieved_assessment.risk_factors == test_risk_assessment_data["risk_factors"]
    assert retrieved_assessment.mitigation_suggestions == test_risk_assessment_data["mitigation_suggestions"]

def test_user_activities_relationship(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_staking_activity_data: Dict[str, Any]):
    """Test user activities relationship"""
    # Create user
    user = User(**test_user_data)
    db_session.add(user)
    db_session.commit()

    # Create activities
    mining_activity = Activity(**test_mining_activity_data)
    staking_activity = Activity(**test_staking_activity_data)
    db_session.add(mining_activity)
    db_session.add(staking_activity)
    db_session.commit()

    # Retrieve user with activities
    retrieved_user = db_session.query(User).filter_by(user_id=test_user_data["user_id"]).first()

    # Check user activities relationship
    assert len(retrieved_user.activities) == 2
    activity_types = [activity.activity_type for activity in retrieved_user.activities]
    assert ActivityType.MINING in activity_types
    assert ActivityType.STAKING in activity_types

def test_user_rewards_relationship(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any], test_staking_reward_data: Dict[str, Any]):
    """Test user rewards relationship"""
    # Create user and activity
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Create rewards
    mining_reward = Reward(**test_mining_reward_data)
    staking_reward = Reward(**test_staking_reward_data)
    db_session.add(mining_reward)
    db_session.add(staking_reward)
    db_session.commit()

    # Retrieve user with rewards
    retrieved_user = db_session.query(User).filter_by(user_id=test_user_data["user_id"]).first()

    # Check user rewards relationship
    assert len(retrieved_user.rewards) == 2
    reward_types = [reward.reward_type for reward in retrieved_user.rewards]
    assert RewardType.MINING in reward_types
    assert RewardType.STAKING in reward_types

def test_activity_rewards_relationship(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any]):
    """Test activity rewards relationship"""
    # Create user and activity
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Create reward
    reward = Reward(**test_mining_reward_data)
    db_session.add(reward)
    db_session.commit()

    # Retrieve activity with rewards
    retrieved_activity = db_session.query(Activity).filter_by(activity_id=test_mining_activity_data["activity_id"]).first()

    # Check activity rewards relationship
    assert len(retrieved_activity.rewards) == 1
    assert retrieved_activity.rewards[0].reward_id == test_mining_reward_data["reward_id"]
    assert retrieved_activity.rewards[0].reward_type == RewardType.MINING

def test_activity_risk_assessments_relationship(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_risk_assessment_data: Dict[str, Any]):
    """Test activity risk assessments relationship"""
    # Create user and activity
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Create risk assessment
    assessment = RiskAssessment(**test_risk_assessment_data)
    db_session.add(assessment)
    db_session.commit()

    # Retrieve activity with risk assessments
    retrieved_activity = db_session.query(Activity).filter_by(activity_id=test_mining_activity_data["activity_id"]).first()

    # Check activity risk assessments relationship
    assert len(retrieved_activity.risk_assessments) == 1
    assert retrieved_activity.risk_assessments[0].assessment_id == test_risk_assessment_data["assessment_id"]
    assert retrieved_activity.risk_assessments[0].risk_level == RiskLevel.HIGH

def test_user_risk_assessments_relationship(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_risk_assessment_data: Dict[str, Any]):
    """Test user risk assessments relationship"""
    # Create user and activity
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Create risk assessment
    assessment = RiskAssessment(**test_risk_assessment_data)
    db_session.add(assessment)
    db_session.commit()

    # Retrieve user with risk assessments
    retrieved_user = db_session.query(User).filter_by(user_id=test_user_data["user_id"]).first()

    # Check user risk assessments relationship
    assert len(retrieved_user.risk_assessments) == 1
    assert retrieved_user.risk_assessments[0].assessment_id == test_risk_assessment_data["assessment_id"]
    assert retrieved_user.risk_assessments[0].risk_level == RiskLevel.HIGH

def test_user_validation(db_session: Session, test_user_data: Dict[str, Any]):
    """Test user validation"""
    # Create user with invalid email
    invalid_user_data = test_user_data.copy()
    invalid_user_data["email"] = "invalid_email"
    user = User(**invalid_user_data)
    db_session.add(user)

    # Check that validation fails
    with pytest.raises(IntegrityError):
        db_session.commit()

    # Rollback session
    db_session.rollback()

    # Create user with invalid username
    invalid_user_data = test_user_data.copy()
    invalid_user_data["username"] = ""
    user = User(**invalid_user_data)
    db_session.add(user)

    # Check that validation fails
    with pytest.raises(IntegrityError):
        db_session.commit()

def test_activity_validation(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any]):
    """Test activity validation"""
    # Create user
    user = User(**test_user_data)
    db_session.add(user)
    db_session.commit()

    # Create activity with invalid user_id
    invalid_activity_data = test_mining_activity_data.copy()
    invalid_activity_data["user_id"] = "non_existent_user"
    activity = Activity(**invalid_activity_data)
    db_session.add(activity)

    # Check that validation fails
    with pytest.raises(IntegrityError):
        db_session.commit()

    # Rollback session
    db_session.rollback()

    # Create activity with invalid activity_type
    invalid_activity_data = test_mining_activity_data.copy()
    invalid_activity_data["activity_type"] = "invalid_type"
    activity = Activity(**invalid_activity_data)
    db_session.add(activity)

    # Check that validation fails
    with pytest.raises(IntegrityError):
        db_session.commit()

def test_reward_validation(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any]):
    """Test reward validation"""
    # Create user and activity
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Create reward with invalid user_id
    invalid_reward_data = test_mining_reward_data.copy()
    invalid_reward_data["user_id"] = "non_existent_user"
    reward = Reward(**invalid_reward_data)
    db_session.add(reward)

    # Check that validation fails
    with pytest.raises(IntegrityError):
        db_session.commit()

    # Rollback session
    db_session.rollback()

    # Create reward with invalid activity_id
    invalid_reward_data = test_mining_reward_data.copy()
    invalid_reward_data["activity_id"] = "non_existent_activity"
    reward = Reward(**invalid_reward_data)
    db_session.add(reward)

    # Check that validation fails
    with pytest.raises(IntegrityError):
        db_session.commit()

    # Rollback session
    db_session.rollback()

    # Create reward with invalid reward_type
    invalid_reward_data = test_mining_reward_data.copy()
    invalid_reward_data["reward_type"] = "invalid_type"
    reward = Reward(**invalid_reward_data)
    db_session.add(reward)

    # Check that validation fails
    with pytest.raises(IntegrityError):
        db_session.commit()

def test_risk_assessment_validation(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_risk_assessment_data: Dict[str, Any]):
    """Test risk assessment validation"""
    # Create user and activity
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Create risk assessment with invalid user_id
    invalid_assessment_data = test_risk_assessment_data.copy()
    invalid_assessment_data["user_id"] = "non_existent_user"
    assessment = RiskAssessment(**invalid_assessment_data)
    db_session.add(assessment)

    # Check that validation fails
    with pytest.raises(IntegrityError):
        db_session.commit()

    # Rollback session
    db_session.rollback()

    # Create risk assessment with invalid activity_id
    invalid_assessment_data = test_risk_assessment_data.copy()
    invalid_assessment_data["activity_id"] = "non_existent_activity"
    assessment = RiskAssessment(**invalid_assessment_data)
    db_session.add(assessment)

    # Check that validation fails
    with pytest.raises(IntegrityError):
        db_session.commit()

    # Rollback session
    db_session.rollback()

    # Create risk assessment with invalid activity_type
    invalid_assessment_data = test_risk_assessment_data.copy()
    invalid_assessment_data["activity_type"] = "invalid_type"
    assessment = RiskAssessment(**invalid_assessment_data)
    db_session.add(assessment)

    # Check that validation fails
    with pytest.raises(IntegrityError):
        db_session.commit()

def test_user_unique_constraint(db_session: Session, test_user_data: Dict[str, Any]):
    """Test user unique constraint"""
    # Create user
    user = User(**test_user_data)
    db_session.add(user)
    db_session.commit()

    # Create user with same email
    duplicate_user_data = test_user_data.copy()
    duplicate_user_data["user_id"] = "duplicate_user"
    duplicate_user_data["username"] = "duplicateuser"
    duplicate_user = User(**duplicate_user_data)
    db_session.add(duplicate_user)

    # Check that unique constraint fails
    with pytest.raises(IntegrityError):
        db_session.commit()

    # Rollback session
    db_session.rollback()

    # Create user with same username
    duplicate_user_data = test_user_data.copy()
    duplicate_user_data["user_id"] = "duplicate_user"
    duplicate_user_data["email"] = "duplicate@example.com"
    duplicate_user = User(**duplicate_user_data)
    db_session.add(duplicate_user)

    # Check that unique constraint fails
    with pytest.raises(IntegrityError):
        db_session.commit()

def test_activity_unique_constraint(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any]):
    """Test activity unique constraint"""
    # Create user
    user = User(**test_user_data)
    db_session.add(user)
    db_session.commit()

    # Create activity
    activity = Activity(**test_mining_activity_data)
    db_session.add(activity)
    db_session.commit()

    # Create activity with same activity_id
    duplicate_activity_data = test_mining_activity_data.copy()
    duplicate_activity = Activity(**duplicate_activity_data)
    db_session.add(duplicate_activity)

    # Check that unique constraint fails
    with pytest.raises(IntegrityError):
        db_session.commit()

def test_reward_unique_constraint(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any]):
    """Test reward unique constraint"""
    # Create user and activity
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Create reward
    reward = Reward(**test_mining_reward_data)
    db_session.add(reward)
    db_session.commit()

    # Create reward with same reward_id
    duplicate_reward_data = test_mining_reward_data.copy()
    duplicate_reward = Reward(**duplicate_reward_data)
    db_session.add(duplicate_reward)

    # Check that unique constraint fails
    with pytest.raises(IntegrityError):
        db_session.commit()

def test_risk_assessment_unique_constraint(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_risk_assessment_data: Dict[str, Any]):
    """Test risk assessment unique constraint"""
    # Create user and activity
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.commit()

    # Create risk assessment
    assessment = RiskAssessment(**test_risk_assessment_data)
    db_session.add(assessment)
    db_session.commit()

    # Create risk assessment with same assessment_id
    duplicate_assessment_data = test_risk_assessment_data.copy()
    duplicate_assessment = RiskAssessment(**duplicate_assessment_data)
    db_session.add(duplicate_assessment)

    # Check that unique constraint fails
    with pytest.raises(IntegrityError):
        db_session.commit()

def test_cascade_delete_user(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any], test_risk_assessment_data: Dict[str, Any]):
    """Test cascade delete user"""
    # Create user, activity, reward, and risk assessment
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    reward = Reward(**test_mining_reward_data)
    assessment = RiskAssessment(**test_risk_assessment_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.add(reward)
    db_session.add(assessment)
    db_session.commit()

    # Delete user
    db_session.delete(user)
    db_session.commit()

    # Check that user is deleted
    assert db_session.query(User).filter_by(user_id=test_user_data["user_id"]).first() is None

    # Check that activity is deleted
    assert db_session.query(Activity).filter_by(activity_id=test_mining_activity_data["activity_id"]).first() is None

    # Check that reward is deleted
    assert db_session.query(Reward).filter_by(reward_id=test_mining_reward_data["reward_id"]).first() is None

    # Check that risk assessment is deleted
    assert db_session.query(RiskAssessment).filter_by(assessment_id=test_risk_assessment_data["assessment_id"]).first() is None

def test_cascade_delete_activity(db_session: Session, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any], test_risk_assessment_data: Dict[str, Any]):
    """Test cascade delete activity"""
    # Create user, activity, reward, and risk assessment
    user = User(**test_user_data)
    activity = Activity(**test_mining_activity_data)
    reward = Reward(**test_mining_reward_data)
    assessment = RiskAssessment(**test_risk_assessment_data)
    db_session.add(user)
    db_session.add(activity)
    db_session.add(reward)
    db_session.add(assessment)
    db_session.commit()

    # Delete activity
    db_session.delete(activity)
    db_session.commit()

    # Check that activity is deleted
    assert db_session.query(Activity).filter_by(activity_id=test_mining_activity_data["activity_id"]).first() is None

    # Check that reward is deleted
    assert db_session.query(Reward).filter_by(reward_id=test_mining_reward_data["reward_id"]).first() is None

    # Check that risk assessment is deleted
    assert db_session.query(RiskAssessment).filter_by(assessment_id=test_risk_assessment_data["assessment_id"]).first() is None

    # Check that user is not deleted
    assert db_session.query(User).filter_by(user_id=test_user_data["user_id"]).first() is not None 