import pytest
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, Any, List

from core.reward_engine import RewardEngine
from database.models import ActivityType

@pytest.mark.asyncio
async def test_reward_engine_initialization(reward_engine: RewardEngine):
    """Test reward engine initialization"""
    # Initialize model
    await reward_engine.initialize_model()
    
    # Check status
    status = reward_engine.get_status()
    assert status["is_operational"] is True
    assert status["model_loaded"] is True
    assert "device" in status
    assert "model_name" in status
    assert status["model_name"] == "bert-base-uncased"

@pytest.mark.asyncio
async def test_calculate_mining_rewards(reward_engine: RewardEngine):
    """Test mining rewards calculation"""
    # Initialize model
    await reward_engine.initialize_model()
    
    # Test data
    user_id = "test_user"
    activity_data = {
        "device_type": "gpu",
        "hash_rate": 100.0,
        "power_usage": 450.0,
        "temperature": 75.0,
        "efficiency": 0.85,
        "duration": 3600  # 1 hour
    }
    performance_metrics = {
        "hash_rate": 95.0,
        "power_efficiency": 0.8,
        "temperature": 78.0,
        "overall_efficiency": 0.82
    }
    historical_data = [
        {
            "timestamp": datetime.utcnow() - timedelta(days=i),
            "hash_rate": 90.0 + i * 0.5,
            "power_usage": 440.0 + i * 2.0,
            "temperature": 74.0 + i * 0.2,
            "efficiency": 0.8 + i * 0.01,
            "duration": 3600
        }
        for i in range(30)
    ]
    
    # Calculate rewards
    result = await reward_engine.calculate_rewards(
        user_id=user_id,
        activity_type=ActivityType.MINING,
        activity_data=activity_data,
        performance_metrics=performance_metrics,
        historical_data=historical_data
    )
    
    # Check result structure
    assert "base_reward" in result
    assert "multipliers" in result
    assert "final_reward" in result
    assert "reward_breakdown" in result
    assert "optimization_suggestions" in result
    
    # Check base reward
    assert isinstance(result["base_reward"], float)
    assert result["base_reward"] > 0
    
    # Check multipliers
    multipliers = result["multipliers"]
    assert "efficiency_multiplier" in multipliers
    assert "consistency_multiplier" in multipliers
    assert "loyalty_multiplier" in multipliers
    assert "activity_multiplier" in multipliers
    
    # Check final reward
    assert isinstance(result["final_reward"], float)
    assert result["final_reward"] > 0
    assert result["final_reward"] >= result["base_reward"]
    
    # Check reward breakdown
    breakdown = result["reward_breakdown"]
    assert "base_reward" in breakdown
    assert "efficiency_bonus" in breakdown
    assert "consistency_bonus" in breakdown
    assert "loyalty_bonus" in breakdown
    assert "activity_bonus" in breakdown
    
    # Check optimization suggestions
    suggestions = result["optimization_suggestions"]
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0

@pytest.mark.asyncio
async def test_calculate_staking_rewards(reward_engine: RewardEngine):
    """Test staking rewards calculation"""
    # Initialize model
    await reward_engine.initialize_model()
    
    # Test data
    user_id = "test_user"
    activity_data = {
        "stake_amount": 1000.0,
        "lock_period": 30,
        "rewards_rate": 0.05,
        "duration": 86400  # 1 day
    }
    performance_metrics = {
        "current_rewards": 45.0,
        "uptime": 0.98,
        "network_health": 0.95
    }
    historical_data = [
        {
            "timestamp": datetime.utcnow() - timedelta(days=i),
            "stake_amount": 900.0 + i * 10.0,
            "rewards": 40.0 + i * 0.5,
            "uptime": 0.97 - i * 0.001,
            "network_health": 0.94 - i * 0.002,
            "duration": 86400
        }
        for i in range(30)
    ]
    
    # Calculate rewards
    result = await reward_engine.calculate_rewards(
        user_id=user_id,
        activity_type=ActivityType.STAKING,
        activity_data=activity_data,
        performance_metrics=performance_metrics,
        historical_data=historical_data
    )
    
    # Check result structure
    assert "base_reward" in result
    assert "multipliers" in result
    assert "final_reward" in result
    assert "reward_breakdown" in result
    assert "optimization_suggestions" in result
    
    # Check base reward
    assert isinstance(result["base_reward"], float)
    assert result["base_reward"] > 0
    
    # Check multipliers
    multipliers = result["multipliers"]
    assert "efficiency_multiplier" in multipliers
    assert "consistency_multiplier" in multipliers
    assert "loyalty_multiplier" in multipliers
    assert "activity_multiplier" in multipliers
    
    # Check final reward
    assert isinstance(result["final_reward"], float)
    assert result["final_reward"] > 0
    assert result["final_reward"] >= result["base_reward"]
    
    # Check reward breakdown
    breakdown = result["reward_breakdown"]
    assert "base_reward" in breakdown
    assert "efficiency_bonus" in breakdown
    assert "consistency_bonus" in breakdown
    assert "loyalty_bonus" in breakdown
    assert "activity_bonus" in breakdown

@pytest.mark.asyncio
async def test_calculate_trading_rewards(reward_engine: RewardEngine):
    """Test trading rewards calculation"""
    # Initialize model
    await reward_engine.initialize_model()
    
    # Test data
    user_id = "test_user"
    activity_data = {
        "trading_pairs": ["BTC/USD", "ETH/USD"],
        "position_size": 0.1,
        "leverage": 1.0,
        "duration": 86400  # 1 day
    }
    performance_metrics = {
        "win_rate": 0.65,
        "profit_factor": 1.8,
        "sharpe_ratio": 1.5,
        "max_drawdown": 0.15
    }
    historical_data = [
        {
            "timestamp": datetime.utcnow() - timedelta(days=i),
            "win_rate": 0.62 + i * 0.001,
            "profit_factor": 1.7 + i * 0.01,
            "sharpe_ratio": 1.4 + i * 0.005,
            "max_drawdown": 0.16 - i * 0.001,
            "duration": 86400
        }
        for i in range(30)
    ]
    
    # Calculate rewards
    result = await reward_engine.calculate_rewards(
        user_id=user_id,
        activity_type=ActivityType.TRADING,
        activity_data=activity_data,
        performance_metrics=performance_metrics,
        historical_data=historical_data
    )
    
    # Check result structure
    assert "base_reward" in result
    assert "multipliers" in result
    assert "final_reward" in result
    assert "reward_breakdown" in result
    assert "optimization_suggestions" in result
    
    # Check base reward
    assert isinstance(result["base_reward"], float)
    assert result["base_reward"] > 0
    
    # Check multipliers
    multipliers = result["multipliers"]
    assert "efficiency_multiplier" in multipliers
    assert "consistency_multiplier" in multipliers
    assert "loyalty_multiplier" in multipliers
    assert "activity_multiplier" in multipliers
    
    # Check final reward
    assert isinstance(result["final_reward"], float)
    assert result["final_reward"] > 0
    assert result["final_reward"] >= result["base_reward"]
    
    # Check reward breakdown
    breakdown = result["reward_breakdown"]
    assert "base_reward" in breakdown
    assert "efficiency_bonus" in breakdown
    assert "consistency_bonus" in breakdown
    assert "loyalty_bonus" in breakdown
    assert "activity_bonus" in breakdown

@pytest.mark.asyncio
async def test_validate_input_data(reward_engine: RewardEngine):
    """Test input data validation"""
    # Test valid data
    valid_data = {
        "user_id": "test_user",
        "activity_type": ActivityType.MINING,
        "activity_data": {"metric": 0.8},
        "performance_metrics": {"metric": 0.85}
    }
    assert reward_engine._validate_input_data(valid_data) is True
    
    # Test missing required fields
    invalid_data = {
        "user_id": "test_user",
        "activity_type": ActivityType.MINING
    }
    with pytest.raises(ValueError):
        reward_engine._validate_input_data(invalid_data)
    
    # Test invalid activity type
    invalid_type_data = {
        **valid_data,
        "activity_type": "invalid_type"
    }
    with pytest.raises(ValueError):
        reward_engine._validate_input_data(invalid_type_data)

@pytest.mark.asyncio
async def test_calculate_base_reward(reward_engine: RewardEngine):
    """Test base reward calculation"""
    # Test mining activity
    mining_data = {
        "device_type": "gpu",
        "hash_rate": 100.0,
        "power_usage": 450.0,
        "temperature": 75.0,
        "efficiency": 0.85,
        "duration": 3600  # 1 hour
    }
    mining_metrics = {
        "hash_rate": 95.0,
        "power_efficiency": 0.8,
        "temperature": 78.0,
        "overall_efficiency": 0.82
    }
    
    mining_reward = reward_engine._calculate_base_reward(
        activity_type=ActivityType.MINING,
        activity_data=mining_data,
        performance_metrics=mining_metrics
    )
    
    assert isinstance(mining_reward, float)
    assert mining_reward > 0
    
    # Test staking activity
    staking_data = {
        "stake_amount": 1000.0,
        "lock_period": 30,
        "rewards_rate": 0.05,
        "duration": 86400  # 1 day
    }
    staking_metrics = {
        "current_rewards": 45.0,
        "uptime": 0.98,
        "network_health": 0.95
    }
    
    staking_reward = reward_engine._calculate_base_reward(
        activity_type=ActivityType.STAKING,
        activity_data=staking_data,
        performance_metrics=staking_metrics
    )
    
    assert isinstance(staking_reward, float)
    assert staking_reward > 0
    
    # Test trading activity
    trading_data = {
        "trading_pairs": ["BTC/USD", "ETH/USD"],
        "position_size": 0.1,
        "leverage": 1.0,
        "duration": 86400  # 1 day
    }
    trading_metrics = {
        "win_rate": 0.65,
        "profit_factor": 1.8,
        "sharpe_ratio": 1.5,
        "max_drawdown": 0.15
    }
    
    trading_reward = reward_engine._calculate_base_reward(
        activity_type=ActivityType.TRADING,
        activity_data=trading_data,
        performance_metrics=trading_metrics
    )
    
    assert isinstance(trading_reward, float)
    assert trading_reward > 0

@pytest.mark.asyncio
async def test_calculate_multipliers(reward_engine: RewardEngine):
    """Test multiplier calculation"""
    # Test data
    performance_metrics = {
        "hash_rate": 95.0,
        "power_efficiency": 0.8,
        "temperature": 78.0,
        "overall_efficiency": 0.82
    }
    historical_data = [
        {
            "timestamp": datetime.utcnow() - timedelta(days=i),
            "hash_rate": 90.0 + i * 0.5,
            "power_usage": 440.0 + i * 2.0,
            "temperature": 74.0 + i * 0.2,
            "efficiency": 0.8 + i * 0.01,
            "duration": 3600
        }
        for i in range(30)
    ]
    
    # Calculate multipliers
    multipliers = reward_engine._calculate_multipliers(
        performance_metrics=performance_metrics,
        historical_data=historical_data
    )
    
    # Check multipliers
    assert "efficiency_multiplier" in multipliers
    assert "consistency_multiplier" in multipliers
    assert "loyalty_multiplier" in multipliers
    assert "activity_multiplier" in multipliers
    
    # Check multiplier ranges
    for multiplier in multipliers.values():
        assert 1.0 <= multiplier <= 2.0

@pytest.mark.asyncio
async def test_calculate_loyalty_score(reward_engine: RewardEngine):
    """Test loyalty score calculation"""
    # Test data
    historical_data = [
        {
            "timestamp": datetime.utcnow() - timedelta(days=i),
            "hash_rate": 90.0 + i * 0.5,
            "power_usage": 440.0 + i * 2.0,
            "temperature": 74.0 + i * 0.2,
            "efficiency": 0.8 + i * 0.01,
            "duration": 3600
        }
        for i in range(30)
    ]
    
    # Calculate loyalty score
    loyalty_score = reward_engine._calculate_loyalty_score(historical_data)
    
    # Check loyalty score
    assert isinstance(loyalty_score, float)
    assert 0 <= loyalty_score <= 1

@pytest.mark.asyncio
async def test_generate_reward_breakdown(reward_engine: RewardEngine):
    """Test reward breakdown generation"""
    # Test data
    base_reward = 100.0
    multipliers = {
        "efficiency_multiplier": 1.2,
        "consistency_multiplier": 1.1,
        "loyalty_multiplier": 1.15,
        "activity_multiplier": 1.05
    }
    final_reward = base_reward * multipliers["efficiency_multiplier"] * multipliers["consistency_multiplier"] * multipliers["loyalty_multiplier"] * multipliers["activity_multiplier"]
    
    # Generate breakdown
    breakdown = reward_engine._generate_reward_breakdown(
        base_reward=base_reward,
        multipliers=multipliers,
        final_reward=final_reward
    )
    
    # Check breakdown
    assert "base_reward" in breakdown
    assert "efficiency_bonus" in breakdown
    assert "consistency_bonus" in breakdown
    assert "loyalty_bonus" in breakdown
    assert "activity_bonus" in breakdown
    
    # Check bonus calculations
    assert breakdown["base_reward"] == base_reward
    assert breakdown["efficiency_bonus"] == base_reward * (multipliers["efficiency_multiplier"] - 1)
    assert breakdown["consistency_bonus"] == base_reward * (multipliers["consistency_multiplier"] - 1)
    assert breakdown["loyalty_bonus"] == base_reward * (multipliers["loyalty_multiplier"] - 1)
    assert breakdown["activity_bonus"] == base_reward * (multipliers["activity_multiplier"] - 1)

@pytest.mark.asyncio
async def test_generate_optimization_suggestions(reward_engine: RewardEngine):
    """Test optimization suggestions generation"""
    # Test data
    performance_metrics = {
        "hash_rate": 95.0,
        "power_efficiency": 0.8,
        "temperature": 78.0,
        "overall_efficiency": 0.82
    }
    reward_breakdown = {
        "base_reward": 100.0,
        "efficiency_bonus": 20.0,
        "consistency_bonus": 10.0,
        "loyalty_bonus": 15.0,
        "activity_bonus": 5.0
    }
    
    # Generate suggestions
    suggestions = reward_engine._generate_optimization_suggestions(
        performance_metrics=performance_metrics,
        reward_breakdown=reward_breakdown
    )
    
    # Check suggestions
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0
    
    # Check suggestion content
    for suggestion in suggestions:
        assert "type" in suggestion
        assert "description" in suggestion
        assert "potential_impact" in suggestion
        assert "priority" in suggestion
        assert suggestion["priority"] in ["high", "medium", "low"] 