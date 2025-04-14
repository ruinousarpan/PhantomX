import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
from decimal import Decimal

from core.risk import (
    calculate_mining_risk,
    calculate_staking_risk,
    calculate_trading_risk,
    validate_risk_assessment,
    monitor_risk_levels,
    get_risk_thresholds,
    update_risk_thresholds,
    aggregate_risk_metrics,
    RiskLevel,
    RiskType,
    RiskFactor
)
from database.models import User, Activity, Risk
from database.exceptions import RiskError

@pytest.fixture
def test_mining_metrics() -> Dict[str, Any]:
    """Create test mining metrics"""
    return {
        "hash_rate_stability": 0.95,  # 95% stable
        "power_efficiency": 0.85,  # 85% efficient
        "temperature_variance": 0.1,  # 10% variance
        "hardware_age": 180,  # days
        "maintenance_score": 0.9,  # 90% maintenance
        "failure_rate": 0.02,  # 2% failure rate
        "network_difficulty_trend": 0.15,  # 15% increase
        "profitability_margin": 0.25,  # 25% margin
        "historical_performance": {
            "uptime": 0.98,
            "avg_hash_rate": 95.0,
            "avg_power_usage": 1450.0,
            "avg_temperature": 75.0
        }
    }

@pytest.fixture
def test_staking_metrics() -> Dict[str, Any]:
    """Create test staking metrics"""
    return {
        "validator_uptime": 0.99,  # 99% uptime
        "slashing_history": 0,  # No slashing events
        "network_participation": 0.85,  # 85% participation
        "token_volatility": 0.2,  # 20% volatility
        "stake_concentration": 0.05,  # 5% of total stake
        "protocol_changes": 2,  # Number of recent changes
        "validator_performance": 0.95,  # 95% performance
        "reward_consistency": 0.9,  # 90% consistent
        "historical_metrics": {
            "avg_uptime": 0.98,
            "missed_blocks": 5,
            "reward_rate": 0.12,
            "peer_count": 50
        }
    }

@pytest.fixture
def test_trading_metrics() -> Dict[str, Any]:
    """Create test trading metrics"""
    return {
        "position_size": 50000.0,  # USD
        "leverage_ratio": 2.0,  # 2x leverage
        "market_volatility": 0.3,  # 30% volatility
        "liquidity_score": 0.8,  # 80% liquidity
        "correlation_factor": 0.4,  # 40% correlation
        "drawdown_risk": 0.15,  # 15% drawdown
        "concentration_risk": 0.25,  # 25% concentration
        "execution_quality": 0.9,  # 90% quality
        "historical_data": {
            "win_rate": 0.65,
            "avg_profit_loss": 500.0,
            "max_drawdown": 5000.0,
            "sharpe_ratio": 1.5
        }
    }

def test_calculate_mining_risk(test_mining_metrics: Dict[str, Any]):
    """Test mining risk calculation"""
    # Calculate risk
    risk_assessment = calculate_mining_risk(
        hash_rate_stability=test_mining_metrics["hash_rate_stability"],
        power_efficiency=test_mining_metrics["power_efficiency"],
        temperature_variance=test_mining_metrics["temperature_variance"],
        hardware_age=test_mining_metrics["hardware_age"],
        maintenance_score=test_mining_metrics["maintenance_score"],
        failure_rate=test_mining_metrics["failure_rate"],
        network_difficulty_trend=test_mining_metrics["network_difficulty_trend"],
        profitability_margin=test_mining_metrics["profitability_margin"],
        historical_performance=test_mining_metrics["historical_performance"]
    )
    
    # Validate risk assessment
    assert isinstance(risk_assessment, Dict)
    assert "risk_score" in risk_assessment
    assert "risk_level" in risk_assessment
    assert "risk_factors" in risk_assessment
    
    # Verify risk score is within bounds
    assert 0 <= risk_assessment["risk_score"] <= 1
    
    # Verify risk level is valid
    assert risk_assessment["risk_level"] in RiskLevel
    
    # Verify risk factors
    assert all(factor in RiskFactor for factor in risk_assessment["risk_factors"])
    
    # Verify specific risk factors
    if test_mining_metrics["temperature_variance"] > 0.15:
        assert RiskFactor.HIGH_TEMPERATURE in risk_assessment["risk_factors"]
    if test_mining_metrics["hardware_age"] > 365:
        assert RiskFactor.HARDWARE_AGING in risk_assessment["risk_factors"]

def test_calculate_staking_risk(test_staking_metrics: Dict[str, Any]):
    """Test staking risk calculation"""
    # Calculate risk
    risk_assessment = calculate_staking_risk(
        validator_uptime=test_staking_metrics["validator_uptime"],
        slashing_history=test_staking_metrics["slashing_history"],
        network_participation=test_staking_metrics["network_participation"],
        token_volatility=test_staking_metrics["token_volatility"],
        stake_concentration=test_staking_metrics["stake_concentration"],
        protocol_changes=test_staking_metrics["protocol_changes"],
        validator_performance=test_staking_metrics["validator_performance"],
        reward_consistency=test_staking_metrics["reward_consistency"],
        historical_metrics=test_staking_metrics["historical_metrics"]
    )
    
    # Validate risk assessment
    assert isinstance(risk_assessment, Dict)
    assert "risk_score" in risk_assessment
    assert "risk_level" in risk_assessment
    assert "risk_factors" in risk_assessment
    
    # Verify risk score is within bounds
    assert 0 <= risk_assessment["risk_score"] <= 1
    
    # Verify risk level is valid
    assert risk_assessment["risk_level"] in RiskLevel
    
    # Verify risk factors
    assert all(factor in RiskFactor for factor in risk_assessment["risk_factors"])
    
    # Verify specific risk factors
    if test_staking_metrics["validator_uptime"] < 0.95:
        assert RiskFactor.LOW_UPTIME in risk_assessment["risk_factors"]
    if test_staking_metrics["slashing_history"] > 0:
        assert RiskFactor.SLASHING_HISTORY in risk_assessment["risk_factors"]

def test_calculate_trading_risk(test_trading_metrics: Dict[str, Any]):
    """Test trading risk calculation"""
    # Calculate risk
    risk_assessment = calculate_trading_risk(
        position_size=test_trading_metrics["position_size"],
        leverage_ratio=test_trading_metrics["leverage_ratio"],
        market_volatility=test_trading_metrics["market_volatility"],
        liquidity_score=test_trading_metrics["liquidity_score"],
        correlation_factor=test_trading_metrics["correlation_factor"],
        drawdown_risk=test_trading_metrics["drawdown_risk"],
        concentration_risk=test_trading_metrics["concentration_risk"],
        execution_quality=test_trading_metrics["execution_quality"],
        historical_data=test_trading_metrics["historical_data"]
    )
    
    # Validate risk assessment
    assert isinstance(risk_assessment, Dict)
    assert "risk_score" in risk_assessment
    assert "risk_level" in risk_assessment
    assert "risk_factors" in risk_assessment
    
    # Verify risk score is within bounds
    assert 0 <= risk_assessment["risk_score"] <= 1
    
    # Verify risk level is valid
    assert risk_assessment["risk_level"] in RiskLevel
    
    # Verify risk factors
    assert all(factor in RiskFactor for factor in risk_assessment["risk_factors"])
    
    # Verify specific risk factors
    if test_trading_metrics["leverage_ratio"] > 3.0:
        assert RiskFactor.HIGH_LEVERAGE in risk_assessment["risk_factors"]
    if test_trading_metrics["market_volatility"] > 0.25:
        assert RiskFactor.HIGH_VOLATILITY in risk_assessment["risk_factors"]

def test_validate_risk_assessment():
    """Test risk assessment validation"""
    # Valid risk assessment
    valid_assessment = {
        "risk_score": 0.75,
        "risk_level": RiskLevel.HIGH,
        "risk_factors": [RiskFactor.HIGH_LEVERAGE, RiskFactor.HIGH_VOLATILITY],
        "timestamp": datetime.utcnow(),
        "type": RiskType.TRADING
    }
    assert validate_risk_assessment(valid_assessment) is True
    
    # Invalid risk score
    invalid_score = valid_assessment.copy()
    invalid_score["risk_score"] = 1.5
    with pytest.raises(RiskError) as excinfo:
        validate_risk_assessment(invalid_score)
    assert "Invalid risk score" in str(excinfo.value)
    
    # Invalid risk level
    invalid_level = valid_assessment.copy()
    invalid_level["risk_level"] = "EXTREME"
    with pytest.raises(RiskError) as excinfo:
        validate_risk_assessment(invalid_level)
    assert "Invalid risk level" in str(excinfo.value)
    
    # Invalid risk factors
    invalid_factors = valid_assessment.copy()
    invalid_factors["risk_factors"] = ["UNKNOWN_FACTOR"]
    with pytest.raises(RiskError) as excinfo:
        validate_risk_assessment(invalid_factors)
    assert "Invalid risk factors" in str(excinfo.value)

def test_monitor_risk_levels(db_session):
    """Test risk level monitoring"""
    # Create test user
    user = User(
        user_id="test_user",
        username="testuser",
        email="test@example.com"
    )
    db_session.add(user)
    
    # Create test activities with risk assessments
    activities = []
    risk_assessments = []
    
    # Mining activity
    mining_activity = Activity(
        activity_id="mining_activity",
        user_id="test_user",
        activity_type="mining",
        start_time=datetime.utcnow() - timedelta(hours=1),
        end_time=datetime.utcnow()
    )
    activities.append(mining_activity)
    
    mining_risk = Risk(
        risk_id="mining_risk",
        activity_id="mining_activity",
        risk_score=0.75,
        risk_level=RiskLevel.HIGH,
        risk_factors=[RiskFactor.HIGH_TEMPERATURE, RiskFactor.HARDWARE_AGING],
        timestamp=datetime.utcnow(),
        type=RiskType.MINING
    )
    risk_assessments.append(mining_risk)
    
    # Staking activity
    staking_activity = Activity(
        activity_id="staking_activity",
        user_id="test_user",
        activity_type="staking",
        start_time=datetime.utcnow() - timedelta(hours=1),
        end_time=datetime.utcnow()
    )
    activities.append(staking_activity)
    
    staking_risk = Risk(
        risk_id="staking_risk",
        activity_id="staking_activity",
        risk_score=0.3,
        risk_level=RiskLevel.LOW,
        risk_factors=[],
        timestamp=datetime.utcnow(),
        type=RiskType.STAKING
    )
    risk_assessments.append(staking_risk)
    
    # Add to database
    db_session.add_all(activities)
    db_session.add_all(risk_assessments)
    db_session.commit()
    
    # Monitor risk levels
    risk_alerts = monitor_risk_levels(
        user_id="test_user",
        start_time=datetime.utcnow() - timedelta(days=1),
        end_time=datetime.utcnow(),
        db_session=db_session
    )
    
    # Verify alerts
    assert len(risk_alerts) > 0
    assert any(alert["risk_level"] == RiskLevel.HIGH for alert in risk_alerts)
    assert any(alert["activity_id"] == "mining_activity" for alert in risk_alerts)

def test_risk_thresholds():
    """Test risk threshold management"""
    # Get default thresholds
    default_thresholds = get_risk_thresholds()
    
    # Verify default thresholds
    assert "mining" in default_thresholds
    assert "staking" in default_thresholds
    assert "trading" in default_thresholds
    
    # Update thresholds
    new_thresholds = {
        "mining": {
            "temperature": 80.0,
            "power_usage": 2000.0,
            "hash_rate_min": 50.0
        },
        "staking": {
            "min_uptime": 0.98,
            "max_slashing": 1,
            "min_participation": 0.9
        },
        "trading": {
            "max_leverage": 2.5,
            "max_position_size": 100000.0,
            "max_drawdown": 0.2
        }
    }
    
    update_risk_thresholds(new_thresholds)
    
    # Verify updated thresholds
    updated_thresholds = get_risk_thresholds()
    assert updated_thresholds == new_thresholds

def test_aggregate_risk_metrics(db_session):
    """Test risk metrics aggregation"""
    # Create test risk assessments
    risk_assessments = [
        Risk(
            risk_id=f"risk_{i}",
            activity_id=f"activity_{i}",
            risk_score=0.2 * (i + 1),  # 0.2, 0.4, 0.6, 0.8, 1.0
            risk_level=RiskLevel.LOW if i < 2 else RiskLevel.MEDIUM if i < 4 else RiskLevel.HIGH,
            risk_factors=[RiskFactor.HIGH_LEVERAGE] if i > 2 else [],
            timestamp=datetime.utcnow() - timedelta(days=i),
            type=RiskType.TRADING
        )
        for i in range(5)
    ]
    
    db_session.add_all(risk_assessments)
    db_session.commit()
    
    # Aggregate metrics
    metrics = aggregate_risk_metrics(
        start_time=datetime.utcnow() - timedelta(days=5),
        end_time=datetime.utcnow(),
        risk_type=RiskType.TRADING,
        db_session=db_session
    )
    
    # Verify metrics
    assert "average_risk_score" in metrics
    assert "risk_level_distribution" in metrics
    assert "common_risk_factors" in metrics
    assert "trend" in metrics
    
    # Verify calculations
    assert abs(metrics["average_risk_score"] - 0.6) < 0.01  # Average of 0.2, 0.4, 0.6, 0.8, 1.0
    assert metrics["risk_level_distribution"][RiskLevel.LOW] == 2
    assert metrics["risk_level_distribution"][RiskLevel.MEDIUM] == 2
    assert metrics["risk_level_distribution"][RiskLevel.HIGH] == 1
    assert RiskFactor.HIGH_LEVERAGE in metrics["common_risk_factors"]

def test_risk_calculation_edge_cases():
    """Test risk calculation edge cases"""
    # Zero values
    zero_mining_risk = calculate_mining_risk(
        hash_rate_stability=0,
        power_efficiency=0,
        temperature_variance=0,
        hardware_age=0,
        maintenance_score=0,
        failure_rate=0,
        network_difficulty_trend=0,
        profitability_margin=0,
        historical_performance={
            "uptime": 0,
            "avg_hash_rate": 0,
            "avg_power_usage": 0,
            "avg_temperature": 0
        }
    )
    assert zero_mining_risk["risk_level"] == RiskLevel.HIGH
    
    # Maximum values
    max_staking_risk = calculate_staking_risk(
        validator_uptime=1,
        slashing_history=10,
        network_participation=1,
        token_volatility=1,
        stake_concentration=1,
        protocol_changes=10,
        validator_performance=1,
        reward_consistency=1,
        historical_metrics={
            "avg_uptime": 1,
            "missed_blocks": 1000,
            "reward_rate": 1,
            "peer_count": 1000
        }
    )
    assert max_staking_risk["risk_level"] == RiskLevel.HIGH
    
    # Boundary values
    boundary_trading_risk = calculate_trading_risk(
        position_size=1000000.0,  # Very large position
        leverage_ratio=10.0,  # High leverage
        market_volatility=0.5,  # High volatility
        liquidity_score=0.1,  # Low liquidity
        correlation_factor=0.9,  # High correlation
        drawdown_risk=0.5,  # High drawdown
        concentration_risk=0.8,  # High concentration
        execution_quality=0.2,  # Poor execution
        historical_data={
            "win_rate": 0.3,
            "avg_profit_loss": -1000.0,
            "max_drawdown": 50000.0,
            "sharpe_ratio": -0.5
        }
    )
    assert boundary_trading_risk["risk_level"] == RiskLevel.HIGH
    assert len(boundary_trading_risk["risk_factors"]) > 3

def test_risk_monitoring_alerts(db_session):
    """Test risk monitoring alerts"""
    # Create test user
    user = User(
        user_id="test_user",
        username="testuser",
        email="test@example.com"
    )
    db_session.add(user)
    
    # Create activities with increasing risk levels
    activities = []
    risk_assessments = []
    
    for i in range(5):
        # Create activity
        activity = Activity(
            activity_id=f"activity_{i}",
            user_id="test_user",
            activity_type="trading",
            start_time=datetime.utcnow() - timedelta(hours=5-i),
            end_time=datetime.utcnow() - timedelta(hours=4-i)
        )
        activities.append(activity)
        
        # Create risk assessment with increasing risk
        risk = Risk(
            risk_id=f"risk_{i}",
            activity_id=f"activity_{i}",
            risk_score=0.2 * (i + 1),
            risk_level=RiskLevel.LOW if i < 2 else RiskLevel.MEDIUM if i < 4 else RiskLevel.HIGH,
            risk_factors=[RiskFactor.HIGH_LEVERAGE] if i > 2 else [],
            timestamp=datetime.utcnow() - timedelta(hours=5-i),
            type=RiskType.TRADING
        )
        risk_assessments.append(risk)
    
    db_session.add_all(activities)
    db_session.add_all(risk_assessments)
    db_session.commit()
    
    # Monitor risk levels
    alerts = monitor_risk_levels(
        user_id="test_user",
        start_time=datetime.utcnow() - timedelta(hours=6),
        end_time=datetime.utcnow(),
        db_session=db_session
    )
    
    # Verify alerts
    assert len(alerts) > 0
    assert any(alert["alert_type"] == "RISK_INCREASE" for alert in alerts)
    assert any(alert["risk_level"] == RiskLevel.HIGH for alert in alerts)
    
    # Verify alert details
    high_risk_alerts = [alert for alert in alerts if alert["risk_level"] == RiskLevel.HIGH]
    assert len(high_risk_alerts) > 0
    assert all("mitigation_steps" in alert for alert in high_risk_alerts)
    assert all("priority" in alert for alert in high_risk_alerts)

def test_risk_threshold_validation():
    """Test risk threshold validation"""
    # Invalid mining thresholds
    with pytest.raises(RiskError) as excinfo:
        update_risk_thresholds({
            "mining": {
                "temperature": -10.0,  # Invalid temperature
                "power_usage": -500.0,  # Invalid power usage
                "hash_rate_min": -50.0  # Invalid hash rate
            }
        })
    assert "Invalid threshold values" in str(excinfo.value)
    
    # Invalid staking thresholds
    with pytest.raises(RiskError) as excinfo:
        update_risk_thresholds({
            "staking": {
                "min_uptime": 1.5,  # Invalid uptime (> 1)
                "max_slashing": -1,  # Invalid slashing count
                "min_participation": 2.0  # Invalid participation rate
            }
        })
    assert "Invalid threshold values" in str(excinfo.value)
    
    # Invalid trading thresholds
    with pytest.raises(RiskError) as excinfo:
        update_risk_thresholds({
            "trading": {
                "max_leverage": 0,  # Invalid leverage
                "max_position_size": -1000.0,  # Invalid position size
                "max_drawdown": 1.5  # Invalid drawdown (> 1)
            }
        })
    assert "Invalid threshold values" in str(excinfo.value) 