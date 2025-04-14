import pytest
from datetime import datetime
import numpy as np
from typing import Dict, Any

from core.neural_mining import NeuralMiningEngine
from database.models import ActivityType

@pytest.mark.asyncio
async def test_neural_mining_engine_initialization(neural_mining_engine: NeuralMiningEngine):
    """Test neural mining engine initialization"""
    # Initialize model
    await neural_mining_engine.initialize_model()
    
    # Check status
    status = neural_mining_engine.get_status()
    assert status["is_operational"] is True
    assert status["model_loaded"] is True
    assert "device" in status
    assert "model_name" in status
    assert status["model_name"] == "bert-base-uncased"

@pytest.mark.asyncio
async def test_analyze_mining_session(neural_mining_engine: NeuralMiningEngine):
    """Test mining session analysis"""
    # Initialize model
    await neural_mining_engine.initialize_model()
    
    # Test data
    user_id = "test_user"
    device_type = "gpu"
    focus_score = 0.85
    duration = 3600  # 1 hour
    timestamp = datetime.utcnow()
    
    # Analyze session
    result = await neural_mining_engine.analyze_mining_session(
        user_id=user_id,
        device_type=device_type,
        focus_score=focus_score,
        duration=duration,
        timestamp=timestamp
    )
    
    # Check result structure
    assert "efficiency_score" in result
    assert "risk_score" in result
    assert "recommendations" in result
    assert "potential_rewards" in result
    
    # Check efficiency score
    assert 0 <= result["efficiency_score"] <= 1
    
    # Check risk score
    assert 0 <= result["risk_score"] <= 1
    
    # Check recommendations
    assert isinstance(result["recommendations"], list)
    assert len(result["recommendations"]) > 0
    
    # Check potential rewards
    assert isinstance(result["potential_rewards"], float)
    assert result["potential_rewards"] >= 0

@pytest.mark.asyncio
async def test_optimize_mining_configuration(neural_mining_engine: NeuralMiningEngine):
    """Test mining configuration optimization"""
    # Initialize model
    await neural_mining_engine.initialize_model()
    
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
    constraints = {
        "max_power": 500.0,
        "max_temperature": 80.0,
        "min_hash_rate": 90.0
    }
    
    # Optimize configuration
    result = await neural_mining_engine.optimize_mining(
        user_id=user_id,
        current_config=current_config,
        performance_data=performance_data,
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
async def test_calculate_base_efficiency(neural_mining_engine: NeuralMiningEngine):
    """Test base efficiency calculation"""
    # Test different device types
    device_types = ["cpu", "gpu", "asic"]
    focus_scores = [0.5, 0.75, 0.9]
    
    for device_type in device_types:
        for focus_score in focus_scores:
            efficiency = neural_mining_engine._calculate_base_efficiency(
                device_type=device_type,
                focus_score=focus_score
            )
            
            # Check efficiency range
            assert 0 <= efficiency <= 1
            
            # Check device type impact
            if device_type == "asic":
                assert efficiency >= 0.8  # ASICs should have high base efficiency
            elif device_type == "gpu":
                assert 0.5 <= efficiency <= 0.9  # GPUs should have medium-high efficiency
            else:  # CPU
                assert 0.3 <= efficiency <= 0.7  # CPUs should have lower efficiency
            
            # Check focus score impact
            assert efficiency <= focus_score  # Efficiency should not exceed focus score

@pytest.mark.asyncio
async def test_analyze_user_behavior(neural_mining_engine: NeuralMiningEngine):
    """Test user behavior analysis"""
    # Test data
    user_id = "test_user"
    device_type = "gpu"
    focus_score = 0.85
    duration = 3600  # 1 hour
    
    # Analyze behavior
    result = neural_mining_engine._analyze_user_behavior(
        user_id=user_id,
        device_type=device_type,
        focus_score=focus_score,
        duration=duration
    )
    
    # Check result structure
    assert "consistency_score" in result
    assert "optimal_duration" in result
    assert "duration_score" in result
    assert "focus_trend" in result
    
    # Check consistency score
    assert 0 <= result["consistency_score"] <= 1
    
    # Check optimal duration
    assert result["optimal_duration"] == 3600  # 1 hour is optimal
    
    # Check duration score
    assert 0 <= result["duration_score"] <= 1
    
    # Check focus trend
    assert result["focus_trend"] in ["improving", "stable", "declining"]

@pytest.mark.asyncio
async def test_generate_recommendations(neural_mining_engine: NeuralMiningEngine):
    """Test recommendation generation"""
    # Test data
    analysis_result = {
        "efficiency_score": 0.75,
        "risk_score": 0.3,
        "duration_score": 0.8,
        "consistency_score": 0.7,
        "focus_trend": "stable"
    }
    
    # Generate recommendations
    recommendations = neural_mining_engine._generate_recommendations(analysis_result)
    
    # Check recommendations
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    
    # Check recommendation content
    for recommendation in recommendations:
        assert "type" in recommendation
        assert "description" in recommendation
        assert "priority" in recommendation
        assert recommendation["priority"] in ["high", "medium", "low"]

@pytest.mark.asyncio
async def test_calculate_potential_rewards(neural_mining_engine: NeuralMiningEngine):
    """Test potential rewards calculation"""
    # Test data
    base_efficiency = 0.8
    duration_score = 0.9
    consistency_score = 0.85
    device_type = "gpu"
    duration = 3600  # 1 hour
    
    # Calculate rewards
    rewards = neural_mining_engine._calculate_potential_rewards(
        base_efficiency=base_efficiency,
        duration_score=duration_score,
        consistency_score=consistency_score,
        device_type=device_type,
        duration=duration
    )
    
    # Check rewards
    assert isinstance(rewards, float)
    assert rewards >= 0
    
    # Check device type impact
    gpu_rewards = rewards
    cpu_rewards = neural_mining_engine._calculate_potential_rewards(
        base_efficiency=base_efficiency,
        duration_score=duration_score,
        consistency_score=consistency_score,
        device_type="cpu",
        duration=duration
    )
    assert gpu_rewards > cpu_rewards  # GPU should have higher rewards than CPU 