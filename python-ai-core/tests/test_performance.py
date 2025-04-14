import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
from decimal import Decimal

from core.performance import (
    calculate_mining_performance,
    calculate_staking_performance,
    calculate_trading_performance,
    validate_performance_metrics,
    monitor_performance_levels,
    get_performance_thresholds,
    update_performance_thresholds,
    aggregate_performance_metrics,
    PerformanceLevel,
    PerformanceType,
    PerformanceFactor
)
from database.models import User, Activity, Performance
from database.exceptions import PerformanceError

@pytest.fixture
def test_mining_metrics() -> Dict[str, Any]:
    """Create test mining metrics"""
    return {
        "hash_rate": 95.0,  # MH/s
        "power_usage": 1450.0,  # Watts
        "temperature": 75.0,  # Celsius
        "uptime": 0.98,  # 98% uptime
        "efficiency": 0.85,  # 85% efficiency
        "block_rewards": 0.5,  # BTC
        "network_difficulty": 45.0,  # TH
        "profitability": 0.25,  # 25% profit margin
        "historical_data": {
            "avg_hash_rate": 92.0,
            "avg_power_usage": 1400.0,
            "avg_temperature": 72.0,
            "total_blocks": 150
        }
    }

@pytest.fixture
def test_staking_metrics() -> Dict[str, Any]:
    """Create test staking metrics"""
    return {
        "validator_uptime": 0.99,  # 99% uptime
        "missed_blocks": 5,  # blocks
        "reward_rate": 0.12,  # 12% APY
        "peer_count": 50,  # peers
        "network_participation": 0.85,  # 85% participation
        "slashing_events": 0,  # no slashing
        "stake_amount": 1000.0,  # ETH
        "validator_count": 100,  # total validators
        "historical_data": {
            "avg_uptime": 0.98,
            "total_rewards": 120.0,
            "avg_peer_count": 48,
            "participation_rate": 0.87
        }
    }

@pytest.fixture
def test_trading_metrics() -> Dict[str, Any]:
    """Create test trading metrics"""
    return {
        "position_size": 50000.0,  # USD
        "leverage_ratio": 2.0,  # 2x leverage
        "win_rate": 0.65,  # 65% win rate
        "profit_loss": 500.0,  # USD
        "drawdown": 0.15,  # 15% drawdown
        "sharpe_ratio": 1.5,  # risk-adjusted return
        "volume": 100000.0,  # USD
        "execution_quality": 0.9,  # 90% quality
        "historical_data": {
            "total_trades": 100,
            "avg_profit_loss": 450.0,
            "max_drawdown": 5000.0,
            "win_loss_ratio": 1.8
        }
    }

def test_calculate_mining_performance(test_mining_metrics: Dict[str, Any]):
    """Test mining performance calculation"""
    # Calculate performance
    performance_metrics = calculate_mining_performance(
        hash_rate=test_mining_metrics["hash_rate"],
        power_usage=test_mining_metrics["power_usage"],
        temperature=test_mining_metrics["temperature"],
        uptime=test_mining_metrics["uptime"],
        efficiency=test_mining_metrics["efficiency"],
        block_rewards=test_mining_metrics["block_rewards"],
        network_difficulty=test_mining_metrics["network_difficulty"],
        profitability=test_mining_metrics["profitability"],
        historical_data=test_mining_metrics["historical_data"]
    )
    
    # Validate performance metrics
    assert isinstance(performance_metrics, Dict)
    assert "performance_score" in performance_metrics
    assert "performance_level" in performance_metrics
    assert "performance_factors" in performance_metrics
    
    # Verify performance score is within bounds
    assert 0 <= performance_metrics["performance_score"] <= 1
    
    # Verify performance level is valid
    assert performance_metrics["performance_level"] in PerformanceLevel
    
    # Verify performance factors
    assert all(factor in PerformanceFactor for factor in performance_metrics["performance_factors"])
    
    # Verify specific performance factors
    if test_mining_metrics["efficiency"] > 0.8:
        assert PerformanceFactor.HIGH_EFFICIENCY in performance_metrics["performance_factors"]
    if test_mining_metrics["uptime"] > 0.95:
        assert PerformanceFactor.HIGH_UPTIME in performance_metrics["performance_factors"]

def test_calculate_staking_performance(test_staking_metrics: Dict[str, Any]):
    """Test staking performance calculation"""
    # Calculate performance
    performance_metrics = calculate_staking_performance(
        validator_uptime=test_staking_metrics["validator_uptime"],
        missed_blocks=test_staking_metrics["missed_blocks"],
        reward_rate=test_staking_metrics["reward_rate"],
        peer_count=test_staking_metrics["peer_count"],
        network_participation=test_staking_metrics["network_participation"],
        slashing_events=test_staking_metrics["slashing_events"],
        stake_amount=test_staking_metrics["stake_amount"],
        validator_count=test_staking_metrics["validator_count"],
        historical_data=test_staking_metrics["historical_data"]
    )
    
    # Validate performance metrics
    assert isinstance(performance_metrics, Dict)
    assert "performance_score" in performance_metrics
    assert "performance_level" in performance_metrics
    assert "performance_factors" in performance_metrics
    
    # Verify performance score is within bounds
    assert 0 <= performance_metrics["performance_score"] <= 1
    
    # Verify performance level is valid
    assert performance_metrics["performance_level"] in PerformanceLevel
    
    # Verify performance factors
    assert all(factor in PerformanceFactor for factor in performance_metrics["performance_factors"])
    
    # Verify specific performance factors
    if test_staking_metrics["validator_uptime"] > 0.95:
        assert PerformanceFactor.HIGH_UPTIME in performance_metrics["performance_factors"]
    if test_staking_metrics["reward_rate"] > 0.1:
        assert PerformanceFactor.HIGH_REWARDS in performance_metrics["performance_factors"]

def test_calculate_trading_performance(test_trading_metrics: Dict[str, Any]):
    """Test trading performance calculation"""
    # Calculate performance
    performance_metrics = calculate_trading_performance(
        position_size=test_trading_metrics["position_size"],
        leverage_ratio=test_trading_metrics["leverage_ratio"],
        win_rate=test_trading_metrics["win_rate"],
        profit_loss=test_trading_metrics["profit_loss"],
        drawdown=test_trading_metrics["drawdown"],
        sharpe_ratio=test_trading_metrics["sharpe_ratio"],
        volume=test_trading_metrics["volume"],
        execution_quality=test_trading_metrics["execution_quality"],
        historical_data=test_trading_metrics["historical_data"]
    )
    
    # Validate performance metrics
    assert isinstance(performance_metrics, Dict)
    assert "performance_score" in performance_metrics
    assert "performance_level" in performance_metrics
    assert "performance_factors" in performance_metrics
    
    # Verify performance score is within bounds
    assert 0 <= performance_metrics["performance_score"] <= 1
    
    # Verify performance level is valid
    assert performance_metrics["performance_level"] in PerformanceLevel
    
    # Verify performance factors
    assert all(factor in PerformanceFactor for factor in performance_metrics["performance_factors"])
    
    # Verify specific performance factors
    if test_trading_metrics["win_rate"] > 0.6:
        assert PerformanceFactor.HIGH_WIN_RATE in performance_metrics["performance_factors"]
    if test_trading_metrics["sharpe_ratio"] > 1.0:
        assert PerformanceFactor.GOOD_RISK_ADJUSTED_RETURN in performance_metrics["performance_factors"]

def test_validate_performance_metrics():
    """Test performance metrics validation"""
    # Valid performance metrics
    valid_metrics = {
        "performance_score": 0.75,
        "performance_level": PerformanceLevel.HIGH,
        "performance_factors": [PerformanceFactor.HIGH_WIN_RATE, PerformanceFactor.GOOD_RISK_ADJUSTED_RETURN],
        "timestamp": datetime.utcnow(),
        "type": PerformanceType.TRADING
    }
    assert validate_performance_metrics(valid_metrics) is True
    
    # Invalid performance score
    invalid_score = valid_metrics.copy()
    invalid_score["performance_score"] = 1.5
    with pytest.raises(PerformanceError) as excinfo:
        validate_performance_metrics(invalid_score)
    assert "Invalid performance score" in str(excinfo.value)
    
    # Invalid performance level
    invalid_level = valid_metrics.copy()
    invalid_level["performance_level"] = "EXTREME"
    with pytest.raises(PerformanceError) as excinfo:
        validate_performance_metrics(invalid_level)
    assert "Invalid performance level" in str(excinfo.value)
    
    # Invalid performance factors
    invalid_factors = valid_metrics.copy()
    invalid_factors["performance_factors"] = ["UNKNOWN_FACTOR"]
    with pytest.raises(PerformanceError) as excinfo:
        validate_performance_metrics(invalid_factors)
    assert "Invalid performance factors" in str(excinfo.value)

def test_monitor_performance_levels(db_session):
    """Test performance level monitoring"""
    # Create test user
    user = User(
        user_id="test_user",
        username="testuser",
        email="test@example.com"
    )
    db_session.add(user)
    
    # Create test activities with performance metrics
    activities = []
    performance_metrics = []
    
    # Mining activity
    mining_activity = Activity(
        activity_id="mining_activity",
        user_id="test_user",
        activity_type="mining",
        start_time=datetime.utcnow() - timedelta(hours=1),
        end_time=datetime.utcnow()
    )
    activities.append(mining_activity)
    
    mining_performance = Performance(
        performance_id="mining_performance",
        activity_id="mining_activity",
        performance_score=0.85,
        performance_level=PerformanceLevel.HIGH,
        performance_factors=[PerformanceFactor.HIGH_EFFICIENCY, PerformanceFactor.HIGH_UPTIME],
        timestamp=datetime.utcnow(),
        type=PerformanceType.MINING
    )
    performance_metrics.append(mining_performance)
    
    # Staking activity
    staking_activity = Activity(
        activity_id="staking_activity",
        user_id="test_user",
        activity_type="staking",
        start_time=datetime.utcnow() - timedelta(hours=1),
        end_time=datetime.utcnow()
    )
    activities.append(staking_activity)
    
    staking_performance = Performance(
        performance_id="staking_performance",
        activity_id="staking_activity",
        performance_score=0.65,
        performance_level=PerformanceLevel.MEDIUM,
        performance_factors=[PerformanceFactor.HIGH_UPTIME],
        timestamp=datetime.utcnow(),
        type=PerformanceType.STAKING
    )
    performance_metrics.append(staking_performance)
    
    # Add to database
    db_session.add_all(activities)
    db_session.add_all(performance_metrics)
    db_session.commit()
    
    # Monitor performance levels
    performance_alerts = monitor_performance_levels(
        user_id="test_user",
        start_time=datetime.utcnow() - timedelta(days=1),
        end_time=datetime.utcnow(),
        db_session=db_session
    )
    
    # Verify alerts
    assert len(performance_alerts) > 0
    assert any(alert["performance_level"] == PerformanceLevel.HIGH for alert in performance_alerts)
    assert any(alert["activity_id"] == "mining_activity" for alert in performance_alerts)

def test_performance_thresholds():
    """Test performance threshold management"""
    # Get default thresholds
    default_thresholds = get_performance_thresholds()
    
    # Verify default thresholds
    assert "mining" in default_thresholds
    assert "staking" in default_thresholds
    assert "trading" in default_thresholds
    
    # Update thresholds
    new_thresholds = {
        "mining": {
            "min_hash_rate": 50.0,
            "max_power_usage": 2000.0,
            "max_temperature": 80.0,
            "min_uptime": 0.95
        },
        "staking": {
            "min_uptime": 0.98,
            "max_missed_blocks": 10,
            "min_reward_rate": 0.1,
            "min_peer_count": 30
        },
        "trading": {
            "min_win_rate": 0.5,
            "max_drawdown": 0.2,
            "min_sharpe_ratio": 1.0,
            "min_execution_quality": 0.8
        }
    }
    
    update_performance_thresholds(new_thresholds)
    
    # Verify updated thresholds
    updated_thresholds = get_performance_thresholds()
    assert updated_thresholds == new_thresholds

def test_aggregate_performance_metrics(db_session):
    """Test performance metrics aggregation"""
    # Create test performance metrics
    performance_metrics = [
        Performance(
            performance_id=f"performance_{i}",
            activity_id=f"activity_{i}",
            performance_score=0.2 * (i + 1),  # 0.2, 0.4, 0.6, 0.8, 1.0
            performance_level=PerformanceLevel.LOW if i < 2 else PerformanceLevel.MEDIUM if i < 4 else PerformanceLevel.HIGH,
            performance_factors=[PerformanceFactor.HIGH_WIN_RATE] if i > 2 else [],
            timestamp=datetime.utcnow() - timedelta(days=i),
            type=PerformanceType.TRADING
        )
        for i in range(5)
    ]
    
    db_session.add_all(performance_metrics)
    db_session.commit()
    
    # Aggregate metrics
    metrics = aggregate_performance_metrics(
        start_time=datetime.utcnow() - timedelta(days=5),
        end_time=datetime.utcnow(),
        performance_type=PerformanceType.TRADING,
        db_session=db_session
    )
    
    # Verify metrics
    assert "average_performance_score" in metrics
    assert "performance_level_distribution" in metrics
    assert "common_performance_factors" in metrics
    assert "trend" in metrics
    
    # Verify calculations
    assert abs(metrics["average_performance_score"] - 0.6) < 0.01  # Average of 0.2, 0.4, 0.6, 0.8, 1.0
    assert metrics["performance_level_distribution"][PerformanceLevel.LOW] == 2
    assert metrics["performance_level_distribution"][PerformanceLevel.MEDIUM] == 2
    assert metrics["performance_level_distribution"][PerformanceLevel.HIGH] == 1
    assert PerformanceFactor.HIGH_WIN_RATE in metrics["common_performance_factors"]

def test_performance_calculation_edge_cases():
    """Test performance calculation edge cases"""
    # Zero values
    zero_mining_performance = calculate_mining_performance(
        hash_rate=0,
        power_usage=0,
        temperature=0,
        uptime=0,
        efficiency=0,
        block_rewards=0,
        network_difficulty=0,
        profitability=0,
        historical_data={
            "avg_hash_rate": 0,
            "avg_power_usage": 0,
            "avg_temperature": 0,
            "total_blocks": 0
        }
    )
    assert zero_mining_performance["performance_level"] == PerformanceLevel.LOW
    
    # Maximum values
    max_staking_performance = calculate_staking_performance(
        validator_uptime=1,
        missed_blocks=0,
        reward_rate=1,
        peer_count=1000,
        network_participation=1,
        slashing_events=0,
        stake_amount=1000000.0,
        validator_count=1000,
        historical_data={
            "avg_uptime": 1,
            "total_rewards": 1000.0,
            "avg_peer_count": 1000,
            "participation_rate": 1
        }
    )
    assert max_staking_performance["performance_level"] == PerformanceLevel.HIGH
    
    # Boundary values
    boundary_trading_performance = calculate_trading_performance(
        position_size=1000000.0,  # Very large position
        leverage_ratio=10.0,  # High leverage
        win_rate=0.9,  # High win rate
        profit_loss=10000.0,  # High profit
        drawdown=0.5,  # High drawdown
        sharpe_ratio=2.0,  # Good risk-adjusted return
        volume=1000000.0,  # High volume
        execution_quality=0.95,  # High execution quality
        historical_data={
            "total_trades": 1000,
            "avg_profit_loss": 5000.0,
            "max_drawdown": 50000.0,
            "win_loss_ratio": 2.5
        }
    )
    assert boundary_trading_performance["performance_level"] == PerformanceLevel.HIGH
    assert len(boundary_trading_performance["performance_factors"]) > 3

def test_performance_monitoring_alerts(db_session):
    """Test performance monitoring alerts"""
    # Create test user
    user = User(
        user_id="test_user",
        username="testuser",
        email="test@example.com"
    )
    db_session.add(user)
    
    # Create activities with increasing performance levels
    activities = []
    performance_metrics = []
    
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
        
        # Create performance metric with increasing performance
        performance = Performance(
            performance_id=f"performance_{i}",
            activity_id=f"activity_{i}",
            performance_score=0.2 * (i + 1),
            performance_level=PerformanceLevel.LOW if i < 2 else PerformanceLevel.MEDIUM if i < 4 else PerformanceLevel.HIGH,
            performance_factors=[PerformanceFactor.HIGH_WIN_RATE] if i > 2 else [],
            timestamp=datetime.utcnow() - timedelta(hours=5-i),
            type=PerformanceType.TRADING
        )
        performance_metrics.append(performance)
    
    db_session.add_all(activities)
    db_session.add_all(performance_metrics)
    db_session.commit()
    
    # Monitor performance levels
    alerts = monitor_performance_levels(
        user_id="test_user",
        start_time=datetime.utcnow() - timedelta(hours=6),
        end_time=datetime.utcnow(),
        db_session=db_session
    )
    
    # Verify alerts
    assert len(alerts) > 0
    assert any(alert["alert_type"] == "PERFORMANCE_INCREASE" for alert in alerts)
    assert any(alert["performance_level"] == PerformanceLevel.HIGH for alert in alerts)
    
    # Verify alert details
    high_performance_alerts = [alert for alert in alerts if alert["performance_level"] == PerformanceLevel.HIGH]
    assert len(high_performance_alerts) > 0
    assert all("recommendations" in alert for alert in high_performance_alerts)
    assert all("priority" in alert for alert in high_performance_alerts)

def test_performance_threshold_validation():
    """Test performance threshold validation"""
    # Invalid mining thresholds
    with pytest.raises(PerformanceError) as excinfo:
        update_performance_thresholds({
            "mining": {
                "min_hash_rate": -50.0,  # Invalid hash rate
                "max_power_usage": -500.0,  # Invalid power usage
                "max_temperature": -10.0,  # Invalid temperature
                "min_uptime": -0.1  # Invalid uptime
            }
        })
    assert "Invalid threshold values" in str(excinfo.value)
    
    # Invalid staking thresholds
    with pytest.raises(PerformanceError) as excinfo:
        update_performance_thresholds({
            "staking": {
                "min_uptime": 1.5,  # Invalid uptime (> 1)
                "max_missed_blocks": -10,  # Invalid missed blocks
                "min_reward_rate": -0.1,  # Invalid reward rate
                "min_peer_count": -30  # Invalid peer count
            }
        })
    assert "Invalid threshold values" in str(excinfo.value)
    
    # Invalid trading thresholds
    with pytest.raises(PerformanceError) as excinfo:
        update_performance_thresholds({
            "trading": {
                "min_win_rate": -0.5,  # Invalid win rate
                "max_drawdown": 1.5,  # Invalid drawdown (> 1)
                "min_sharpe_ratio": -1.0,  # Invalid Sharpe ratio
                "min_execution_quality": 1.5  # Invalid execution quality
            }
        })
    assert "Invalid threshold values" in str(excinfo.value) 