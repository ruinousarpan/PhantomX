import pytest
from datetime import datetime
import numpy as np
from typing import Dict, Any

from core.optimization_engine import OptimizationEngine
from database.models import ActivityType

@pytest.mark.asyncio
async def test_optimization_engine_initialization(optimization_engine: OptimizationEngine):
    """Test optimization engine initialization"""
    # Initialize model
    await optimization_engine.initialize_model()
    
    # Check status
    status = optimization_engine.get_status()
    assert status["is_operational"] is True
    assert status["model_loaded"] is True
    assert "device" in status
    assert "model_name" in status
    assert status["model_name"] == "bert-base-uncased"

@pytest.mark.asyncio
async def test_optimize_mining_activity(optimization_engine: OptimizationEngine):
    """Test mining activity optimization"""
    # Initialize model
    await optimization_engine.initialize_model()
    
    # Test data
    user_id = "test_user"
    current_config = {
        "device_type": "gpu",
        "hash_rate": 100.0,
        "power_limit": 80,
        "memory_clock": 1000,
        "core_clock": 1500,
        "fan_speed": 70
    }
    performance_data = {
        "hash_rate": 95.0,
        "power_usage": 450.0,
        "temperature": 75.0,
        "efficiency": 0.8
    }
    behavior_data = {
        "session_duration": 3600,
        "focus_score": 0.85,
        "consistency_score": 0.9
    }
    constraints = {
        "max_power": 500.0,
        "max_temperature": 80.0,
        "min_hash_rate": 90.0
    }
    
    # Optimize activity
    result = await optimization_engine.optimize_activity(
        user_id=user_id,
        activity_type=ActivityType.MINING,
        current_config=current_config,
        performance_data=performance_data,
        behavior_data=behavior_data,
        constraints=constraints
    )
    
    # Check result structure
    assert "optimized_config" in result
    assert "expected_improvements" in result
    assert "risk_assessment" in result
    
    # Check optimized config
    optimized_config = result["optimized_config"]
    assert "device_type" in optimized_config
    assert "hash_rate" in optimized_config
    assert "power_limit" in optimized_config
    assert "memory_clock" in optimized_config
    assert "core_clock" in optimized_config
    assert "fan_speed" in optimized_config
    
    # Check expected improvements
    improvements = result["expected_improvements"]
    assert "hash_rate" in improvements
    assert "power_efficiency" in improvements
    assert "temperature" in improvements
    assert "overall_efficiency" in improvements
    
    # Check risk assessment
    risk = result["risk_assessment"]
    assert "risk_score" in risk
    assert "risk_factors" in risk
    assert "mitigation_strategies" in risk

@pytest.mark.asyncio
async def test_optimize_staking_activity(optimization_engine: OptimizationEngine):
    """Test staking activity optimization"""
    # Initialize model
    await optimization_engine.initialize_model()
    
    # Test data
    user_id = "test_user"
    current_config = {
        "stake_amount": 1000.0,
        "lock_period": 30,
        "rewards_rate": 0.05
    }
    performance_data = {
        "current_rewards": 45.0,
        "uptime": 0.98,
        "network_health": 0.95
    }
    behavior_data = {
        "stake_history": [
            {"amount": 800.0, "duration": 25, "rewards": 35.0},
            {"amount": 900.0, "duration": 28, "rewards": 40.0}
        ],
        "consistency_score": 0.9
    }
    constraints = {
        "min_stake": 100.0,
        "max_stake": 5000.0,
        "min_lock_period": 7,
        "max_lock_period": 90
    }
    
    # Optimize activity
    result = await optimization_engine.optimize_activity(
        user_id=user_id,
        activity_type=ActivityType.STAKING,
        current_config=current_config,
        performance_data=performance_data,
        behavior_data=behavior_data,
        constraints=constraints
    )
    
    # Check result structure
    assert "optimized_config" in result
    assert "expected_improvements" in result
    assert "risk_assessment" in result
    
    # Check optimized config
    optimized_config = result["optimized_config"]
    assert "stake_amount" in optimized_config
    assert "lock_period" in optimized_config
    assert "rewards_rate" in optimized_config
    
    # Check constraints are respected
    assert constraints["min_stake"] <= optimized_config["stake_amount"] <= constraints["max_stake"]
    assert constraints["min_lock_period"] <= optimized_config["lock_period"] <= constraints["max_lock_period"]

@pytest.mark.asyncio
async def test_optimize_trading_activity(optimization_engine: OptimizationEngine):
    """Test trading activity optimization"""
    # Initialize model
    await optimization_engine.initialize_model()
    
    # Test data
    user_id = "test_user"
    current_config = {
        "trading_pairs": ["BTC/USD", "ETH/USD"],
        "position_size": 0.1,
        "leverage": 1.0,
        "stop_loss": 0.05,
        "take_profit": 0.1
    }
    performance_data = {
        "win_rate": 0.65,
        "profit_factor": 1.8,
        "sharpe_ratio": 1.5,
        "max_drawdown": 0.15
    }
    behavior_data = {
        "trading_history": [
            {"pair": "BTC/USD", "result": "win", "profit": 0.08},
            {"pair": "ETH/USD", "result": "loss", "loss": 0.03}
        ],
        "risk_tolerance": 0.7
    }
    constraints = {
        "max_leverage": 2.0,
        "max_position_size": 0.2,
        "min_stop_loss": 0.02,
        "max_stop_loss": 0.1
    }
    
    # Optimize activity
    result = await optimization_engine.optimize_activity(
        user_id=user_id,
        activity_type=ActivityType.TRADING,
        current_config=current_config,
        performance_data=performance_data,
        behavior_data=behavior_data,
        constraints=constraints
    )
    
    # Check result structure
    assert "optimized_config" in result
    assert "expected_improvements" in result
    assert "risk_assessment" in result
    
    # Check optimized config
    optimized_config = result["optimized_config"]
    assert "trading_pairs" in optimized_config
    assert "position_size" in optimized_config
    assert "leverage" in optimized_config
    assert "stop_loss" in optimized_config
    assert "take_profit" in optimized_config
    
    # Check constraints are respected
    assert optimized_config["leverage"] <= constraints["max_leverage"]
    assert optimized_config["position_size"] <= constraints["max_position_size"]
    assert constraints["min_stop_loss"] <= optimized_config["stop_loss"] <= constraints["max_stop_loss"]

@pytest.mark.asyncio
async def test_validate_input_data(optimization_engine: OptimizationEngine):
    """Test input data validation"""
    # Test valid data
    valid_data = {
        "user_id": "test_user",
        "activity_type": ActivityType.MINING,
        "current_config": {"key": "value"},
        "performance_data": {"metric": 0.8},
        "behavior_data": {"score": 0.9},
        "constraints": {"limit": 100}
    }
    assert optimization_engine._validate_input_data(valid_data) is True
    
    # Test missing required fields
    invalid_data = {
        "user_id": "test_user",
        "activity_type": ActivityType.MINING
    }
    with pytest.raises(ValueError):
        optimization_engine._validate_input_data(invalid_data)
    
    # Test invalid activity type
    invalid_type_data = {
        **valid_data,
        "activity_type": "invalid_type"
    }
    with pytest.raises(ValueError):
        optimization_engine._validate_input_data(invalid_type_data)

@pytest.mark.asyncio
async def test_calculate_expected_improvements(optimization_engine: OptimizationEngine):
    """Test expected improvements calculation"""
    # Test data
    current_metrics = {
        "efficiency": 0.8,
        "hash_rate": 95.0,
        "power_usage": 450.0,
        "temperature": 75.0
    }
    optimized_metrics = {
        "efficiency": 0.85,
        "hash_rate": 100.0,
        "power_usage": 420.0,
        "temperature": 70.0
    }
    
    # Calculate improvements
    improvements = optimization_engine._calculate_expected_improvements(
        current_metrics=current_metrics,
        optimized_metrics=optimized_metrics
    )
    
    # Check improvements
    assert "efficiency" in improvements
    assert "hash_rate" in improvements
    assert "power_usage" in improvements
    assert "temperature" in improvements
    
    # Check improvement calculations
    assert improvements["efficiency"] == 0.0625  # (0.85 - 0.8) / 0.8
    assert improvements["hash_rate"] == 0.0526  # (100 - 95) / 95
    assert improvements["power_usage"] == 0.0667  # (420 - 450) / 450
    assert improvements["temperature"] == 0.0667  # (70 - 75) / 75

@pytest.mark.asyncio
async def test_generate_optimization_recommendations(optimization_engine: OptimizationEngine):
    """Test optimization recommendations generation"""
    # Test data
    optimization_result = {
        "optimized_config": {
            "device_type": "gpu",
            "hash_rate": 100.0,
            "power_limit": 80
        },
        "expected_improvements": {
            "efficiency": 0.0625,
            "hash_rate": 0.0526,
            "power_usage": 0.0667
        },
        "risk_assessment": {
            "risk_score": 0.3,
            "risk_factors": ["temperature", "power_usage"],
            "mitigation_strategies": ["increase_fan_speed", "reduce_power_limit"]
        }
    }
    
    # Generate recommendations
    recommendations = optimization_engine._generate_recommendations(optimization_result)
    
    # Check recommendations
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    
    # Check recommendation content
    for recommendation in recommendations:
        assert "type" in recommendation
        assert "description" in recommendation
        assert "priority" in recommendation
        assert recommendation["priority"] in ["high", "medium", "low"] 