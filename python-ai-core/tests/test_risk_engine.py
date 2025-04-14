import pytest
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, Any, List

from core.risk_engine import RiskEngine
from database.models import ActivityType

@pytest.mark.asyncio
async def test_risk_engine_initialization(risk_engine: RiskEngine):
    """Test risk engine initialization"""
    # Initialize model
    await risk_engine.initialize_model()
    
    # Check status
    status = risk_engine.get_status()
    assert status["is_operational"] is True
    assert status["model_loaded"] is True
    assert "device" in status
    assert "model_name" in status
    assert status["model_name"] == "bert-base-uncased"

@pytest.mark.asyncio
async def test_assess_mining_risk(risk_engine: RiskEngine):
    """Test mining risk assessment"""
    # Initialize model
    await risk_engine.initialize_model()
    
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
    
    # Assess risk
    result = await risk_engine.assess_risk(
        user_id=user_id,
        activity_type=ActivityType.MINING,
        activity_data=activity_data,
        performance_metrics=performance_metrics,
        historical_data=historical_data
    )
    
    # Check result structure
    assert "risk_score" in result
    assert "risk_factors" in result
    assert "risk_breakdown" in result
    assert "mitigation_strategies" in result
    assert "impact_analysis" in result
    
    # Check risk score
    assert isinstance(result["risk_score"], float)
    assert 0 <= result["risk_score"] <= 1
    
    # Check risk factors
    risk_factors = result["risk_factors"]
    assert isinstance(risk_factors, list)
    assert len(risk_factors) > 0
    
    # Check risk breakdown
    risk_breakdown = result["risk_breakdown"]
    assert "hardware_risk" in risk_breakdown
    assert "power_risk" in risk_breakdown
    assert "temperature_risk" in risk_breakdown
    assert "efficiency_risk" in risk_breakdown
    
    # Check mitigation strategies
    mitigation_strategies = result["mitigation_strategies"]
    assert isinstance(mitigation_strategies, list)
    assert len(mitigation_strategies) > 0
    
    # Check impact analysis
    impact_analysis = result["impact_analysis"]
    assert "performance_impact" in impact_analysis
    assert "financial_impact" in impact_analysis
    assert "reliability_impact" in impact_analysis

@pytest.mark.asyncio
async def test_assess_staking_risk(risk_engine: RiskEngine):
    """Test staking risk assessment"""
    # Initialize model
    await risk_engine.initialize_model()
    
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
    
    # Assess risk
    result = await risk_engine.assess_risk(
        user_id=user_id,
        activity_type=ActivityType.STAKING,
        activity_data=activity_data,
        performance_metrics=performance_metrics,
        historical_data=historical_data
    )
    
    # Check result structure
    assert "risk_score" in result
    assert "risk_factors" in result
    assert "risk_breakdown" in result
    assert "mitigation_strategies" in result
    assert "impact_analysis" in result
    
    # Check risk score
    assert isinstance(result["risk_score"], float)
    assert 0 <= result["risk_score"] <= 1
    
    # Check risk factors
    risk_factors = result["risk_factors"]
    assert isinstance(risk_factors, list)
    assert len(risk_factors) > 0
    
    # Check risk breakdown
    risk_breakdown = result["risk_breakdown"]
    assert "market_risk" in risk_breakdown
    assert "network_risk" in risk_breakdown
    assert "liquidity_risk" in risk_breakdown
    assert "rewards_risk" in risk_breakdown
    
    # Check mitigation strategies
    mitigation_strategies = result["mitigation_strategies"]
    assert isinstance(mitigation_strategies, list)
    assert len(mitigation_strategies) > 0
    
    # Check impact analysis
    impact_analysis = result["impact_analysis"]
    assert "performance_impact" in impact_analysis
    assert "financial_impact" in impact_analysis
    assert "reliability_impact" in impact_analysis

@pytest.mark.asyncio
async def test_assess_trading_risk(risk_engine: RiskEngine):
    """Test trading risk assessment"""
    # Initialize model
    await risk_engine.initialize_model()
    
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
    
    # Assess risk
    result = await risk_engine.assess_risk(
        user_id=user_id,
        activity_type=ActivityType.TRADING,
        activity_data=activity_data,
        performance_metrics=performance_metrics,
        historical_data=historical_data
    )
    
    # Check result structure
    assert "risk_score" in result
    assert "risk_factors" in result
    assert "risk_breakdown" in result
    assert "mitigation_strategies" in result
    assert "impact_analysis" in result
    
    # Check risk score
    assert isinstance(result["risk_score"], float)
    assert 0 <= result["risk_score"] <= 1
    
    # Check risk factors
    risk_factors = result["risk_factors"]
    assert isinstance(risk_factors, list)
    assert len(risk_factors) > 0
    
    # Check risk breakdown
    risk_breakdown = result["risk_breakdown"]
    assert "market_risk" in risk_breakdown
    assert "volatility_risk" in risk_breakdown
    assert "liquidity_risk" in risk_breakdown
    assert "leverage_risk" in risk_breakdown
    
    # Check mitigation strategies
    mitigation_strategies = result["mitigation_strategies"]
    assert isinstance(mitigation_strategies, list)
    assert len(mitigation_strategies) > 0
    
    # Check impact analysis
    impact_analysis = result["impact_analysis"]
    assert "performance_impact" in impact_analysis
    assert "financial_impact" in impact_analysis
    assert "reliability_impact" in impact_analysis

@pytest.mark.asyncio
async def test_validate_input_data(risk_engine: RiskEngine):
    """Test input data validation"""
    # Test valid data
    valid_data = {
        "user_id": "test_user",
        "activity_type": ActivityType.MINING,
        "activity_data": {"metric": 0.8},
        "performance_metrics": {"metric": 0.85}
    }
    assert risk_engine._validate_input_data(valid_data) is True
    
    # Test missing required fields
    invalid_data = {
        "user_id": "test_user",
        "activity_type": ActivityType.MINING
    }
    with pytest.raises(ValueError):
        risk_engine._validate_input_data(invalid_data)
    
    # Test invalid activity type
    invalid_type_data = {
        **valid_data,
        "activity_type": "invalid_type"
    }
    with pytest.raises(ValueError):
        risk_engine._validate_input_data(invalid_type_data)

@pytest.mark.asyncio
async def test_calculate_risk_score(risk_engine: RiskEngine):
    """Test risk score calculation"""
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
    
    mining_risk = risk_engine._calculate_risk_score(
        activity_type=ActivityType.MINING,
        activity_data=mining_data,
        performance_metrics=mining_metrics
    )
    
    assert isinstance(mining_risk, float)
    assert 0 <= mining_risk <= 1
    
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
    
    staking_risk = risk_engine._calculate_risk_score(
        activity_type=ActivityType.STAKING,
        activity_data=staking_data,
        performance_metrics=staking_metrics
    )
    
    assert isinstance(staking_risk, float)
    assert 0 <= staking_risk <= 1
    
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
    
    trading_risk = risk_engine._calculate_risk_score(
        activity_type=ActivityType.TRADING,
        activity_data=trading_data,
        performance_metrics=trading_metrics
    )
    
    assert isinstance(trading_risk, float)
    assert 0 <= trading_risk <= 1

@pytest.mark.asyncio
async def test_determine_risk_level(risk_engine: RiskEngine):
    """Test risk level determination"""
    # Test different risk scores
    risk_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for score in risk_scores:
        risk_level = risk_engine._determine_risk_level(score)
        
        # Check risk level
        assert risk_level in ["low", "medium", "high", "critical"]
        
        # Check risk level assignment
        if score < 0.3:
            assert risk_level == "low"
        elif score < 0.5:
            assert risk_level == "medium"
        elif score < 0.7:
            assert risk_level == "high"
        else:
            assert risk_level == "critical"

@pytest.mark.asyncio
async def test_generate_risk_breakdown(risk_engine: RiskEngine):
    """Test risk breakdown generation"""
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
    
    mining_breakdown = risk_engine._generate_risk_breakdown(
        activity_type=ActivityType.MINING,
        activity_data=mining_data,
        performance_metrics=mining_metrics
    )
    
    # Check mining breakdown
    assert "hardware_risk" in mining_breakdown
    assert "power_risk" in mining_breakdown
    assert "temperature_risk" in mining_breakdown
    assert "efficiency_risk" in mining_breakdown
    
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
    
    staking_breakdown = risk_engine._generate_risk_breakdown(
        activity_type=ActivityType.STAKING,
        activity_data=staking_data,
        performance_metrics=staking_metrics
    )
    
    # Check staking breakdown
    assert "market_risk" in staking_breakdown
    assert "network_risk" in staking_breakdown
    assert "liquidity_risk" in staking_breakdown
    assert "rewards_risk" in staking_breakdown
    
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
    
    trading_breakdown = risk_engine._generate_risk_breakdown(
        activity_type=ActivityType.TRADING,
        activity_data=trading_data,
        performance_metrics=trading_metrics
    )
    
    # Check trading breakdown
    assert "market_risk" in trading_breakdown
    assert "volatility_risk" in trading_breakdown
    assert "liquidity_risk" in trading_breakdown
    assert "leverage_risk" in trading_breakdown

@pytest.mark.asyncio
async def test_generate_mitigation_strategies(risk_engine: RiskEngine):
    """Test mitigation strategies generation"""
    # Test data
    risk_breakdown = {
        "hardware_risk": 0.7,
        "power_risk": 0.6,
        "temperature_risk": 0.8,
        "efficiency_risk": 0.5
    }
    risk_level = "high"
    
    # Generate strategies
    strategies = risk_engine._generate_mitigation_strategies(
        risk_breakdown=risk_breakdown,
        risk_level=risk_level
    )
    
    # Check strategies
    assert isinstance(strategies, list)
    assert len(strategies) > 0
    
    # Check strategy content
    for strategy in strategies:
        assert "type" in strategy
        assert "description" in strategy
        assert "effectiveness" in strategy
        assert "priority" in strategy
        assert strategy["priority"] in ["high", "medium", "low"]
        assert 0 <= strategy["effectiveness"] <= 1

@pytest.mark.asyncio
async def test_analyze_impact(risk_engine: RiskEngine):
    """Test impact analysis"""
    # Test data
    risk_breakdown = {
        "hardware_risk": 0.7,
        "power_risk": 0.6,
        "temperature_risk": 0.8,
        "efficiency_risk": 0.5
    }
    activity_data = {
        "device_type": "gpu",
        "hash_rate": 100.0,
        "power_usage": 450.0,
        "temperature": 75.0,
        "efficiency": 0.85,
        "duration": 3600  # 1 hour
    }
    
    # Analyze impact
    impact = risk_engine._analyze_impact(
        risk_breakdown=risk_breakdown,
        activity_data=activity_data
    )
    
    # Check impact
    assert "performance_impact" in impact
    assert "financial_impact" in impact
    assert "reliability_impact" in impact
    
    # Check impact scores
    assert 0 <= impact["performance_impact"] <= 1
    assert 0 <= impact["financial_impact"] <= 1
    assert 0 <= impact["reliability_impact"] <= 1 