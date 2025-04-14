import pytest
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, Any, List

from core.prediction_engine import PredictionEngine
from database.models import ActivityType

@pytest.mark.asyncio
async def test_prediction_engine_initialization(prediction_engine: PredictionEngine):
    """Test prediction engine initialization"""
    # Initialize model
    await prediction_engine.initialize_model()
    
    # Check status
    status = prediction_engine.get_status()
    assert status["is_operational"] is True
    assert status["model_loaded"] is True
    assert "device" in status
    assert "model_name" in status
    assert status["model_name"] == "bert-base-uncased"

@pytest.mark.asyncio
async def test_predict_mining_activity(prediction_engine: PredictionEngine):
    """Test mining activity prediction"""
    # Initialize model
    await prediction_engine.initialize_model()
    
    # Test data
    user_id = "test_user"
    historical_data = [
        {
            "timestamp": datetime.utcnow() - timedelta(days=i),
            "hash_rate": 95.0 + i * 0.5,
            "power_usage": 450.0 + i * 2.0,
            "temperature": 75.0 + i * 0.2,
            "efficiency": 0.8 + i * 0.01
        }
        for i in range(30)
    ]
    current_state = {
        "device_type": "gpu",
        "hash_rate": 110.0,
        "power_usage": 480.0,
        "temperature": 78.0,
        "efficiency": 0.85
    }
    
    # Predict activity
    result = await prediction_engine.predict_activity(
        user_id=user_id,
        activity_type=ActivityType.MINING,
        historical_data=historical_data,
        current_state=current_state,
        prediction_horizon="short_term"
    )
    
    # Check result structure
    assert "predictions" in result
    assert "confidence_scores" in result
    assert "trend_analysis" in result
    assert "recommendations" in result
    
    # Check predictions
    predictions = result["predictions"]
    assert "hash_rate" in predictions
    assert "power_usage" in predictions
    assert "temperature" in predictions
    assert "efficiency" in predictions
    
    # Check confidence scores
    confidence_scores = result["confidence_scores"]
    assert "hash_rate" in confidence_scores
    assert "power_usage" in confidence_scores
    assert "temperature" in confidence_scores
    assert "efficiency" in confidence_scores
    
    # Check trend analysis
    trend_analysis = result["trend_analysis"]
    assert "hash_rate_trend" in trend_analysis
    assert "power_usage_trend" in trend_analysis
    assert "temperature_trend" in trend_analysis
    assert "efficiency_trend" in trend_analysis
    
    # Check recommendations
    recommendations = result["recommendations"]
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0

@pytest.mark.asyncio
async def test_predict_staking_activity(prediction_engine: PredictionEngine):
    """Test staking activity prediction"""
    # Initialize model
    await prediction_engine.initialize_model()
    
    # Test data
    user_id = "test_user"
    historical_data = [
        {
            "timestamp": datetime.utcnow() - timedelta(days=i),
            "stake_amount": 1000.0 + i * 10.0,
            "rewards": 45.0 + i * 0.5,
            "uptime": 0.98 - i * 0.001,
            "network_health": 0.95 - i * 0.002
        }
        for i in range(30)
    ]
    current_state = {
        "stake_amount": 1300.0,
        "rewards": 60.0,
        "uptime": 0.97,
        "network_health": 0.93
    }
    
    # Predict activity
    result = await prediction_engine.predict_activity(
        user_id=user_id,
        activity_type=ActivityType.STAKING,
        historical_data=historical_data,
        current_state=current_state,
        prediction_horizon="medium_term"
    )
    
    # Check result structure
    assert "predictions" in result
    assert "confidence_scores" in result
    assert "trend_analysis" in result
    assert "recommendations" in result
    
    # Check predictions
    predictions = result["predictions"]
    assert "stake_amount" in predictions
    assert "rewards" in predictions
    assert "uptime" in predictions
    assert "network_health" in predictions
    
    # Check confidence scores
    confidence_scores = result["confidence_scores"]
    assert "stake_amount" in confidence_scores
    assert "rewards" in confidence_scores
    assert "uptime" in confidence_scores
    assert "network_health" in confidence_scores
    
    # Check trend analysis
    trend_analysis = result["trend_analysis"]
    assert "stake_amount_trend" in trend_analysis
    assert "rewards_trend" in trend_analysis
    assert "uptime_trend" in trend_analysis
    assert "network_health_trend" in trend_analysis

@pytest.mark.asyncio
async def test_predict_trading_activity(prediction_engine: PredictionEngine):
    """Test trading activity prediction"""
    # Initialize model
    await prediction_engine.initialize_model()
    
    # Test data
    user_id = "test_user"
    historical_data = [
        {
            "timestamp": datetime.utcnow() - timedelta(days=i),
            "win_rate": 0.65 + i * 0.001,
            "profit_factor": 1.8 + i * 0.01,
            "sharpe_ratio": 1.5 + i * 0.005,
            "max_drawdown": 0.15 - i * 0.001
        }
        for i in range(30)
    ]
    current_state = {
        "win_rate": 0.68,
        "profit_factor": 2.1,
        "sharpe_ratio": 1.65,
        "max_drawdown": 0.12
    }
    
    # Predict activity
    result = await prediction_engine.predict_activity(
        user_id=user_id,
        activity_type=ActivityType.TRADING,
        historical_data=historical_data,
        current_state=current_state,
        prediction_horizon="long_term"
    )
    
    # Check result structure
    assert "predictions" in result
    assert "confidence_scores" in result
    assert "trend_analysis" in result
    assert "recommendations" in result
    
    # Check predictions
    predictions = result["predictions"]
    assert "win_rate" in predictions
    assert "profit_factor" in predictions
    assert "sharpe_ratio" in predictions
    assert "max_drawdown" in predictions
    
    # Check confidence scores
    confidence_scores = result["confidence_scores"]
    assert "win_rate" in confidence_scores
    assert "profit_factor" in confidence_scores
    assert "sharpe_ratio" in confidence_scores
    assert "max_drawdown" in confidence_scores
    
    # Check trend analysis
    trend_analysis = result["trend_analysis"]
    assert "win_rate_trend" in trend_analysis
    assert "profit_factor_trend" in trend_analysis
    assert "sharpe_ratio_trend" in trend_analysis
    assert "max_drawdown_trend" in trend_analysis

@pytest.mark.asyncio
async def test_validate_input_data(prediction_engine: PredictionEngine):
    """Test input data validation"""
    # Test valid data
    valid_data = {
        "user_id": "test_user",
        "activity_type": ActivityType.MINING,
        "historical_data": [{"metric": 0.8}],
        "current_state": {"metric": 0.85},
        "prediction_horizon": "short_term"
    }
    assert prediction_engine._validate_input_data(valid_data) is True
    
    # Test missing required fields
    invalid_data = {
        "user_id": "test_user",
        "activity_type": ActivityType.MINING
    }
    with pytest.raises(ValueError):
        prediction_engine._validate_input_data(invalid_data)
    
    # Test invalid activity type
    invalid_type_data = {
        **valid_data,
        "activity_type": "invalid_type"
    }
    with pytest.raises(ValueError):
        prediction_engine._validate_input_data(invalid_type_data)
    
    # Test invalid prediction horizon
    invalid_horizon_data = {
        **valid_data,
        "prediction_horizon": "invalid_horizon"
    }
    with pytest.raises(ValueError):
        prediction_engine._validate_input_data(invalid_horizon_data)

@pytest.mark.asyncio
async def test_prepare_features(prediction_engine: PredictionEngine):
    """Test feature preparation"""
    # Test data
    historical_data = [
        {
            "timestamp": datetime.utcnow() - timedelta(days=i),
            "hash_rate": 95.0 + i * 0.5,
            "power_usage": 450.0 + i * 2.0,
            "temperature": 75.0 + i * 0.2,
            "efficiency": 0.8 + i * 0.01
        }
        for i in range(30)
    ]
    current_state = {
        "device_type": "gpu",
        "hash_rate": 110.0,
        "power_usage": 480.0,
        "temperature": 78.0,
        "efficiency": 0.85
    }
    
    # Prepare features
    features = prediction_engine._prepare_features(
        historical_data=historical_data,
        current_state=current_state
    )
    
    # Check features
    assert isinstance(features, np.ndarray)
    assert features.shape[0] > 0
    assert features.shape[1] > 0

@pytest.mark.asyncio
async def test_calculate_confidence_scores(prediction_engine: PredictionEngine):
    """Test confidence score calculation"""
    # Test data
    predictions = {
        "hash_rate": 115.0,
        "power_usage": 490.0,
        "temperature": 80.0,
        "efficiency": 0.87
    }
    historical_data = [
        {
            "timestamp": datetime.utcnow() - timedelta(days=i),
            "hash_rate": 95.0 + i * 0.5,
            "power_usage": 450.0 + i * 2.0,
            "temperature": 75.0 + i * 0.2,
            "efficiency": 0.8 + i * 0.01
        }
        for i in range(30)
    ]
    
    # Calculate confidence scores
    confidence_scores = prediction_engine._calculate_confidence_scores(
        predictions=predictions,
        historical_data=historical_data
    )
    
    # Check confidence scores
    assert "hash_rate" in confidence_scores
    assert "power_usage" in confidence_scores
    assert "temperature" in confidence_scores
    assert "efficiency" in confidence_scores
    
    # Check score ranges
    for score in confidence_scores.values():
        assert 0 <= score <= 1

@pytest.mark.asyncio
async def test_analyze_trends(prediction_engine: PredictionEngine):
    """Test trend analysis"""
    # Test data
    historical_data = [
        {
            "timestamp": datetime.utcnow() - timedelta(days=i),
            "hash_rate": 95.0 + i * 0.5,
            "power_usage": 450.0 + i * 2.0,
            "temperature": 75.0 + i * 0.2,
            "efficiency": 0.8 + i * 0.01
        }
        for i in range(30)
    ]
    
    # Analyze trends
    trends = prediction_engine._analyze_trends(historical_data)
    
    # Check trends
    assert "hash_rate_trend" in trends
    assert "power_usage_trend" in trends
    assert "temperature_trend" in trends
    assert "efficiency_trend" in trends
    
    # Check trend values
    for trend in trends.values():
        assert trend in ["increasing", "stable", "decreasing"]

@pytest.mark.asyncio
async def test_generate_recommendations(prediction_engine: PredictionEngine):
    """Test recommendation generation"""
    # Test data
    predictions = {
        "hash_rate": 115.0,
        "power_usage": 490.0,
        "temperature": 80.0,
        "efficiency": 0.87
    }
    confidence_scores = {
        "hash_rate": 0.85,
        "power_usage": 0.82,
        "temperature": 0.78,
        "efficiency": 0.80
    }
    trends = {
        "hash_rate_trend": "increasing",
        "power_usage_trend": "stable",
        "temperature_trend": "increasing",
        "efficiency_trend": "stable"
    }
    
    # Generate recommendations
    recommendations = prediction_engine._generate_recommendations(
        predictions=predictions,
        confidence_scores=confidence_scores,
        trends=trends
    )
    
    # Check recommendations
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    
    # Check recommendation content
    for recommendation in recommendations:
        assert "type" in recommendation
        assert "description" in recommendation
        assert "priority" in recommendation
        assert recommendation["priority"] in ["high", "medium", "low"] 