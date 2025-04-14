import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
from decimal import Decimal

from core.rewards import (
    calculate_mining_reward,
    calculate_staking_reward,
    calculate_trading_reward,
    validate_reward,
    distribute_reward,
    get_reward_status,
    update_reward_status,
    aggregate_rewards,
    RewardType,
    RewardStatus
)
from database.models import User, Activity, Reward
from database.exceptions import RewardError

@pytest.fixture
def test_mining_data() -> Dict[str, Any]:
    """Create test mining data"""
    return {
        "hash_rate": 100.0,  # MH/s
        "power_usage": 1500.0,  # Watts
        "duration": 3600,  # seconds
        "difficulty": 2.5,
        "network_hash_rate": 1000.0,  # MH/s
        "block_reward": 6.25,  # BTC
        "timestamp": datetime.utcnow(),
        "device_type": "GPU",
        "efficiency_score": 0.85
    }

@pytest.fixture
def test_staking_data() -> Dict[str, Any]:
    """Create test staking data"""
    return {
        "amount": 1000.0,  # Token amount
        "duration": 30 * 24 * 3600,  # 30 days in seconds
        "apy": 0.12,  # 12% APY
        "validator_performance": 0.98,  # 98% uptime
        "network_stake": 100000.0,  # Total network stake
        "timestamp": datetime.utcnow(),
        "token_price": 2.5,  # USD
        "slashing_events": 0
    }

@pytest.fixture
def test_trading_data() -> Dict[str, Any]:
    """Create test trading data"""
    return {
        "volume": 50000.0,  # USD
        "fee_tier": 0.001,  # 0.1% fee
        "maker_ratio": 0.6,  # 60% maker orders
        "profit_loss": 1500.0,  # USD
        "timestamp": datetime.utcnow(),
        "market_volatility": 0.25,
        "trading_pair": "BTC/USD",
        "strategy_performance": 0.75
    }

def test_calculate_mining_reward(test_mining_data: Dict[str, Any]):
    """Test mining reward calculation"""
    # Calculate reward
    reward = calculate_mining_reward(
        hash_rate=test_mining_data["hash_rate"],
        power_usage=test_mining_data["power_usage"],
        duration=test_mining_data["duration"],
        difficulty=test_mining_data["difficulty"],
        network_hash_rate=test_mining_data["network_hash_rate"],
        block_reward=test_mining_data["block_reward"],
        efficiency_score=test_mining_data["efficiency_score"]
    )
    
    # Validate reward
    assert isinstance(reward, Decimal)
    assert reward > 0
    
    # Verify reward calculation
    expected_share = (test_mining_data["hash_rate"] / test_mining_data["network_hash_rate"])
    expected_blocks = (test_mining_data["duration"] / 600)  # 600 seconds per block
    expected_reward = Decimal(str(expected_share * expected_blocks * test_mining_data["block_reward"] * test_mining_data["efficiency_score"]))
    assert abs(reward - expected_reward) < Decimal('0.00001')

def test_calculate_staking_reward(test_staking_data: Dict[str, Any]):
    """Test staking reward calculation"""
    # Calculate reward
    reward = calculate_staking_reward(
        amount=test_staking_data["amount"],
        duration=test_staking_data["duration"],
        apy=test_staking_data["apy"],
        validator_performance=test_staking_data["validator_performance"],
        network_stake=test_staking_data["network_stake"]
    )
    
    # Validate reward
    assert isinstance(reward, Decimal)
    assert reward > 0
    
    # Verify reward calculation
    expected_base_reward = Decimal(str(test_staking_data["amount"] * test_staking_data["apy"] * (test_staking_data["duration"] / (365 * 24 * 3600))))
    expected_reward = expected_base_reward * Decimal(str(test_staking_data["validator_performance"]))
    assert abs(reward - expected_reward) < Decimal('0.00001')

def test_calculate_trading_reward(test_trading_data: Dict[str, Any]):
    """Test trading reward calculation"""
    # Calculate reward
    reward = calculate_trading_reward(
        volume=test_trading_data["volume"],
        fee_tier=test_trading_data["fee_tier"],
        maker_ratio=test_trading_data["maker_ratio"],
        profit_loss=test_trading_data["profit_loss"],
        market_volatility=test_trading_data["market_volatility"],
        strategy_performance=test_trading_data["strategy_performance"]
    )
    
    # Validate reward
    assert isinstance(reward, Decimal)
    assert reward > 0
    
    # Verify reward calculation
    base_fee_reward = Decimal(str(test_trading_data["volume"] * test_trading_data["fee_tier"]))
    maker_bonus = base_fee_reward * Decimal(str(test_trading_data["maker_ratio"] * 0.2))  # 20% bonus for maker orders
    performance_multiplier = Decimal(str(1 + (test_trading_data["strategy_performance"] - 0.5)))  # Performance bonus
    expected_reward = (base_fee_reward + maker_bonus) * performance_multiplier
    assert abs(reward - expected_reward) < Decimal('0.00001')

def test_validate_reward():
    """Test reward validation"""
    # Valid reward
    valid_reward = {
        "amount": Decimal("1.23456"),
        "type": RewardType.MINING,
        "timestamp": datetime.utcnow(),
        "status": RewardStatus.PENDING
    }
    assert validate_reward(valid_reward) is True
    
    # Invalid amount
    invalid_amount = valid_reward.copy()
    invalid_amount["amount"] = Decimal("-1.0")
    with pytest.raises(RewardError) as excinfo:
        validate_reward(invalid_amount)
    assert "Invalid reward amount" in str(excinfo.value)
    
    # Invalid type
    invalid_type = valid_reward.copy()
    invalid_type["type"] = "INVALID"
    with pytest.raises(RewardError) as excinfo:
        validate_reward(invalid_type)
    assert "Invalid reward type" in str(excinfo.value)
    
    # Invalid timestamp
    invalid_timestamp = valid_reward.copy()
    invalid_timestamp["timestamp"] = datetime.utcnow() + timedelta(days=1)
    with pytest.raises(RewardError) as excinfo:
        validate_reward(invalid_timestamp)
    assert "Invalid reward timestamp" in str(excinfo.value)

def test_distribute_reward(db_session):
    """Test reward distribution"""
    # Create test user
    user = User(
        user_id="test_user",
        username="testuser",
        email="test@example.com"
    )
    db_session.add(user)
    
    # Create test activity
    activity = Activity(
        activity_id="test_activity",
        user_id="test_user",
        activity_type="mining",
        start_time=datetime.utcnow() - timedelta(hours=1),
        end_time=datetime.utcnow()
    )
    db_session.add(activity)
    db_session.commit()
    
    # Distribute reward
    reward_data = {
        "amount": Decimal("1.23456"),
        "type": RewardType.MINING,
        "timestamp": datetime.utcnow(),
        "status": RewardStatus.PENDING,
        "user_id": "test_user",
        "activity_id": "test_activity"
    }
    
    reward_id = distribute_reward(reward_data, db_session)
    
    # Verify reward was distributed
    reward = db_session.query(Reward).filter_by(reward_id=reward_id).first()
    assert reward is not None
    assert reward.amount == reward_data["amount"]
    assert reward.type == reward_data["type"]
    assert reward.status == RewardStatus.PENDING
    assert reward.user_id == reward_data["user_id"]
    assert reward.activity_id == reward_data["activity_id"]

def test_get_reward_status(db_session):
    """Test getting reward status"""
    # Create test reward
    reward = Reward(
        reward_id="test_reward",
        amount=Decimal("1.23456"),
        type=RewardType.MINING,
        timestamp=datetime.utcnow(),
        status=RewardStatus.PENDING,
        user_id="test_user",
        activity_id="test_activity"
    )
    db_session.add(reward)
    db_session.commit()
    
    # Get reward status
    status = get_reward_status("test_reward", db_session)
    assert status == RewardStatus.PENDING

def test_update_reward_status(db_session):
    """Test updating reward status"""
    # Create test reward
    reward = Reward(
        reward_id="test_reward",
        amount=Decimal("1.23456"),
        type=RewardType.MINING,
        timestamp=datetime.utcnow(),
        status=RewardStatus.PENDING,
        user_id="test_user",
        activity_id="test_activity"
    )
    db_session.add(reward)
    db_session.commit()
    
    # Update reward status
    update_reward_status("test_reward", RewardStatus.COMPLETED, db_session)
    
    # Verify status was updated
    reward = db_session.query(Reward).filter_by(reward_id="test_reward").first()
    assert reward.status == RewardStatus.COMPLETED

def test_aggregate_rewards(db_session):
    """Test reward aggregation"""
    # Create test rewards
    rewards = [
        Reward(
            reward_id=f"test_reward_{i}",
            amount=Decimal(str(i + 1)),
            type=RewardType.MINING,
            timestamp=datetime.utcnow(),
            status=RewardStatus.COMPLETED,
            user_id="test_user",
            activity_id=f"test_activity_{i}"
        )
        for i in range(3)
    ]
    db_session.add_all(rewards)
    db_session.commit()
    
    # Aggregate rewards
    aggregated = aggregate_rewards(
        user_id="test_user",
        start_time=datetime.utcnow() - timedelta(days=1),
        end_time=datetime.utcnow(),
        reward_type=RewardType.MINING,
        db_session=db_session
    )
    
    # Verify aggregation
    assert aggregated["total_amount"] == Decimal("6")  # 1 + 2 + 3
    assert aggregated["count"] == 3
    assert aggregated["average"] == Decimal("2")

def test_reward_calculation_edge_cases():
    """Test reward calculation edge cases"""
    # Zero hash rate
    assert calculate_mining_reward(
        hash_rate=0,
        power_usage=1500.0,
        duration=3600,
        difficulty=2.5,
        network_hash_rate=1000.0,
        block_reward=6.25,
        efficiency_score=0.85
    ) == Decimal("0")
    
    # Zero stake amount
    assert calculate_staking_reward(
        amount=0,
        duration=30 * 24 * 3600,
        apy=0.12,
        validator_performance=0.98,
        network_stake=100000.0
    ) == Decimal("0")
    
    # Zero trading volume
    assert calculate_trading_reward(
        volume=0,
        fee_tier=0.001,
        maker_ratio=0.6,
        profit_loss=1500.0,
        market_volatility=0.25,
        strategy_performance=0.75
    ) == Decimal("0")

def test_reward_distribution_errors(db_session):
    """Test reward distribution error cases"""
    # Invalid user
    with pytest.raises(RewardError) as excinfo:
        distribute_reward({
            "amount": Decimal("1.23456"),
            "type": RewardType.MINING,
            "timestamp": datetime.utcnow(),
            "status": RewardStatus.PENDING,
            "user_id": "invalid_user",
            "activity_id": "test_activity"
        }, db_session)
    assert "User not found" in str(excinfo.value)
    
    # Invalid activity
    with pytest.raises(RewardError) as excinfo:
        distribute_reward({
            "amount": Decimal("1.23456"),
            "type": RewardType.MINING,
            "timestamp": datetime.utcnow(),
            "status": RewardStatus.PENDING,
            "user_id": "test_user",
            "activity_id": "invalid_activity"
        }, db_session)
    assert "Activity not found" in str(excinfo.value)

def test_reward_status_transitions(db_session):
    """Test reward status transitions"""
    # Create test reward
    reward = Reward(
        reward_id="test_reward",
        amount=Decimal("1.23456"),
        type=RewardType.MINING,
        timestamp=datetime.utcnow(),
        status=RewardStatus.PENDING,
        user_id="test_user",
        activity_id="test_activity"
    )
    db_session.add(reward)
    db_session.commit()
    
    # Valid transitions
    valid_transitions = [
        (RewardStatus.PENDING, RewardStatus.PROCESSING),
        (RewardStatus.PROCESSING, RewardStatus.COMPLETED),
        (RewardStatus.PROCESSING, RewardStatus.FAILED),
        (RewardStatus.FAILED, RewardStatus.PENDING)
    ]
    
    for from_status, to_status in valid_transitions:
        reward.status = from_status
        db_session.commit()
        update_reward_status("test_reward", to_status, db_session)
        assert reward.status == to_status
    
    # Invalid transition
    reward.status = RewardStatus.COMPLETED
    db_session.commit()
    with pytest.raises(RewardError) as excinfo:
        update_reward_status("test_reward", RewardStatus.PENDING, db_session)
    assert "Invalid status transition" in str(excinfo.value)

def test_reward_aggregation_filters(db_session):
    """Test reward aggregation with different filters"""
    # Create test rewards with different types and timestamps
    rewards = [
        # Mining rewards
        Reward(
            reward_id="mining_1",
            amount=Decimal("1"),
            type=RewardType.MINING,
            timestamp=datetime.utcnow() - timedelta(days=2),
            status=RewardStatus.COMPLETED,
            user_id="test_user",
            activity_id="activity_1"
        ),
        Reward(
            reward_id="mining_2",
            amount=Decimal("2"),
            type=RewardType.MINING,
            timestamp=datetime.utcnow() - timedelta(days=1),
            status=RewardStatus.COMPLETED,
            user_id="test_user",
            activity_id="activity_2"
        ),
        # Staking rewards
        Reward(
            reward_id="staking_1",
            amount=Decimal("3"),
            type=RewardType.STAKING,
            timestamp=datetime.utcnow() - timedelta(days=2),
            status=RewardStatus.COMPLETED,
            user_id="test_user",
            activity_id="activity_3"
        ),
        Reward(
            reward_id="staking_2",
            amount=Decimal("4"),
            type=RewardType.STAKING,
            timestamp=datetime.utcnow() - timedelta(days=1),
            status=RewardStatus.COMPLETED,
            user_id="test_user",
            activity_id="activity_4"
        )
    ]
    db_session.add_all(rewards)
    db_session.commit()
    
    # Test filtering by type
    mining_rewards = aggregate_rewards(
        user_id="test_user",
        start_time=datetime.utcnow() - timedelta(days=3),
        end_time=datetime.utcnow(),
        reward_type=RewardType.MINING,
        db_session=db_session
    )
    assert mining_rewards["total_amount"] == Decimal("3")  # 1 + 2
    assert mining_rewards["count"] == 2
    
    # Test filtering by time range
    recent_rewards = aggregate_rewards(
        user_id="test_user",
        start_time=datetime.utcnow() - timedelta(days=1, hours=1),
        end_time=datetime.utcnow(),
        db_session=db_session
    )
    assert recent_rewards["total_amount"] == Decimal("6")  # 2 + 4
    assert recent_rewards["count"] == 2
    
    # Test filtering by status
    completed_rewards = aggregate_rewards(
        user_id="test_user",
        start_time=datetime.utcnow() - timedelta(days=3),
        end_time=datetime.utcnow(),
        status=RewardStatus.COMPLETED,
        db_session=db_session
    )
    assert completed_rewards["total_amount"] == Decimal("10")  # 1 + 2 + 3 + 4
    assert completed_rewards["count"] == 4 