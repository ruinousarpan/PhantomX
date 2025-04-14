import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from decimal import Decimal

from core.data_processing import (
    transform_mining_data,
    transform_staking_data,
    transform_trading_data,
    aggregate_performance_data,
    aggregate_risk_data,
    aggregate_reward_data,
    analyze_trends,
    detect_anomalies,
    calculate_correlations,
    DataProcessingError
)
from database.models import User, DataProcessingResult
from database.exceptions import DataProcessingError as DBDataProcessingError

@pytest.fixture
def test_user(db_session):
    """Create test user"""
    user = User(
        user_id="test_user",
        username="testuser",
        email="test@example.com",
        first_name="Test",
        last_name="User",
        role="USER",
        status="ACTIVE",
        email_verified=True,
        created_at=datetime.utcnow(),
        last_login=datetime.utcnow()
    )
    db_session.add(user)
    db_session.commit()
    return user

@pytest.fixture
def test_mining_data():
    """Create test mining data"""
    return pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="H"),
        "hash_rate": np.random.uniform(80, 100, 100),
        "power_usage": np.random.uniform(1200, 1500, 100),
        "temperature": np.random.uniform(60, 80, 100),
        "uptime": np.random.uniform(0.95, 1.0, 100),
        "efficiency": np.random.uniform(0.8, 0.9, 100),
        "block_rewards": np.random.uniform(0.4, 0.6, 100),
        "network_difficulty": np.random.uniform(40, 50, 100),
        "profitability": np.random.uniform(0.2, 0.3, 100)
    })

@pytest.fixture
def test_staking_data():
    """Create test staking data"""
    return pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="H"),
        "validator_uptime": np.random.uniform(0.95, 1.0, 100),
        "missed_blocks": np.random.randint(0, 5, 100),
        "reward_rate": np.random.uniform(0.1, 0.15, 100),
        "peer_count": np.random.randint(40, 60, 100),
        "network_participation": np.random.uniform(0.8, 0.9, 100),
        "slashing_events": np.random.randint(0, 2, 100),
        "stake_amount": np.random.uniform(900, 1100, 100),
        "validator_count": np.random.randint(90, 110, 100)
    })

@pytest.fixture
def test_trading_data():
    """Create test trading data"""
    return pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="H"),
        "position_size": np.random.uniform(40000, 60000, 100),
        "leverage_ratio": np.random.uniform(1.5, 2.5, 100),
        "win_rate": np.random.uniform(0.5, 0.7, 100),
        "profit_loss": np.random.uniform(400, 600, 100),
        "drawdown": np.random.uniform(0.1, 0.2, 100),
        "sharpe_ratio": np.random.uniform(1.2, 1.8, 100),
        "volume": np.random.uniform(90000, 110000, 100),
        "execution_quality": np.random.uniform(0.85, 0.95, 100)
    })

@pytest.fixture
def test_performance_data():
    """Create test performance data"""
    return pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="H"),
        "mining_performance": np.random.uniform(0.8, 0.9, 100),
        "staking_performance": np.random.uniform(0.85, 0.95, 100),
        "trading_performance": np.random.uniform(0.7, 0.8, 100),
        "overall_performance": np.random.uniform(0.8, 0.9, 100)
    })

@pytest.fixture
def test_risk_data():
    """Create test risk data"""
    return pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="H"),
        "mining_risk": np.random.uniform(0.2, 0.3, 100),
        "staking_risk": np.random.uniform(0.1, 0.2, 100),
        "trading_risk": np.random.uniform(0.3, 0.4, 100),
        "overall_risk": np.random.uniform(0.2, 0.3, 100)
    })

@pytest.fixture
def test_reward_data():
    """Create test reward data"""
    return pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="H"),
        "mining_rewards": np.random.uniform(0.4, 0.6, 100),
        "staking_rewards": np.random.uniform(0.1, 0.15, 100),
        "trading_rewards": np.random.uniform(0.05, 0.1, 100),
        "overall_rewards": np.random.uniform(0.2, 0.25, 100)
    })

def test_transform_mining_data(db_session, test_user, test_mining_data):
    """Test mining data transformation"""
    # Transform mining data
    result = transform_mining_data(
        user_id=test_user.user_id,
        data=test_mining_data,
        db_session=db_session
    )
    
    # Verify transformation result
    assert isinstance(result, pd.DataFrame)
    assert "timestamp" in result.columns
    assert "hash_rate_normalized" in result.columns
    assert "power_usage_normalized" in result.columns
    assert "temperature_normalized" in result.columns
    assert "uptime_normalized" in result.columns
    assert "efficiency_normalized" in result.columns
    assert "block_rewards_normalized" in result.columns
    assert "network_difficulty_normalized" in result.columns
    assert "profitability_normalized" in result.columns
    
    # Verify normalization
    assert result["hash_rate_normalized"].min() >= 0
    assert result["hash_rate_normalized"].max() <= 1
    assert result["power_usage_normalized"].min() >= 0
    assert result["power_usage_normalized"].max() <= 1
    assert result["temperature_normalized"].min() >= 0
    assert result["temperature_normalized"].max() <= 1
    assert result["uptime_normalized"].min() >= 0
    assert result["uptime_normalized"].max() <= 1
    assert result["efficiency_normalized"].min() >= 0
    assert result["efficiency_normalized"].max() <= 1
    assert result["block_rewards_normalized"].min() >= 0
    assert result["block_rewards_normalized"].max() <= 1
    assert result["network_difficulty_normalized"].min() >= 0
    assert result["network_difficulty_normalized"].max() <= 1
    assert result["profitability_normalized"].min() >= 0
    assert result["profitability_normalized"].max() <= 1
    
    # Verify database entry
    db_result = db_session.query(DataProcessingResult).filter_by(
        user_id=test_user.user_id,
        processing_type="MINING_DATA_TRANSFORMATION"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_transform_staking_data(db_session, test_user, test_staking_data):
    """Test staking data transformation"""
    # Transform staking data
    result = transform_staking_data(
        user_id=test_user.user_id,
        data=test_staking_data,
        db_session=db_session
    )
    
    # Verify transformation result
    assert isinstance(result, pd.DataFrame)
    assert "timestamp" in result.columns
    assert "validator_uptime_normalized" in result.columns
    assert "missed_blocks_normalized" in result.columns
    assert "reward_rate_normalized" in result.columns
    assert "peer_count_normalized" in result.columns
    assert "network_participation_normalized" in result.columns
    assert "slashing_events_normalized" in result.columns
    assert "stake_amount_normalized" in result.columns
    assert "validator_count_normalized" in result.columns
    
    # Verify normalization
    assert result["validator_uptime_normalized"].min() >= 0
    assert result["validator_uptime_normalized"].max() <= 1
    assert result["missed_blocks_normalized"].min() >= 0
    assert result["missed_blocks_normalized"].max() <= 1
    assert result["reward_rate_normalized"].min() >= 0
    assert result["reward_rate_normalized"].max() <= 1
    assert result["peer_count_normalized"].min() >= 0
    assert result["peer_count_normalized"].max() <= 1
    assert result["network_participation_normalized"].min() >= 0
    assert result["network_participation_normalized"].max() <= 1
    assert result["slashing_events_normalized"].min() >= 0
    assert result["slashing_events_normalized"].max() <= 1
    assert result["stake_amount_normalized"].min() >= 0
    assert result["stake_amount_normalized"].max() <= 1
    assert result["validator_count_normalized"].min() >= 0
    assert result["validator_count_normalized"].max() <= 1
    
    # Verify database entry
    db_result = db_session.query(DataProcessingResult).filter_by(
        user_id=test_user.user_id,
        processing_type="STAKING_DATA_TRANSFORMATION"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_transform_trading_data(db_session, test_user, test_trading_data):
    """Test trading data transformation"""
    # Transform trading data
    result = transform_trading_data(
        user_id=test_user.user_id,
        data=test_trading_data,
        db_session=db_session
    )
    
    # Verify transformation result
    assert isinstance(result, pd.DataFrame)
    assert "timestamp" in result.columns
    assert "position_size_normalized" in result.columns
    assert "leverage_ratio_normalized" in result.columns
    assert "win_rate_normalized" in result.columns
    assert "profit_loss_normalized" in result.columns
    assert "drawdown_normalized" in result.columns
    assert "sharpe_ratio_normalized" in result.columns
    assert "volume_normalized" in result.columns
    assert "execution_quality_normalized" in result.columns
    
    # Verify normalization
    assert result["position_size_normalized"].min() >= 0
    assert result["position_size_normalized"].max() <= 1
    assert result["leverage_ratio_normalized"].min() >= 0
    assert result["leverage_ratio_normalized"].max() <= 1
    assert result["win_rate_normalized"].min() >= 0
    assert result["win_rate_normalized"].max() <= 1
    assert result["profit_loss_normalized"].min() >= 0
    assert result["profit_loss_normalized"].max() <= 1
    assert result["drawdown_normalized"].min() >= 0
    assert result["drawdown_normalized"].max() <= 1
    assert result["sharpe_ratio_normalized"].min() >= 0
    assert result["sharpe_ratio_normalized"].max() <= 1
    assert result["volume_normalized"].min() >= 0
    assert result["volume_normalized"].max() <= 1
    assert result["execution_quality_normalized"].min() >= 0
    assert result["execution_quality_normalized"].max() <= 1
    
    # Verify database entry
    db_result = db_session.query(DataProcessingResult).filter_by(
        user_id=test_user.user_id,
        processing_type="TRADING_DATA_TRANSFORMATION"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_aggregate_performance_data(db_session, test_user, test_performance_data):
    """Test performance data aggregation"""
    # Aggregate performance data
    result = aggregate_performance_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        db_session=db_session
    )
    
    # Verify aggregation result
    assert isinstance(result, Dict)
    assert "mining_performance" in result
    assert "staking_performance" in result
    assert "trading_performance" in result
    assert "overall_performance" in result
    
    # Verify aggregation details
    assert "mean" in result["mining_performance"]
    assert "std" in result["mining_performance"]
    assert "min" in result["mining_performance"]
    assert "max" in result["mining_performance"]
    assert "trend" in result["mining_performance"]
    
    assert "mean" in result["staking_performance"]
    assert "std" in result["staking_performance"]
    assert "min" in result["staking_performance"]
    assert "max" in result["staking_performance"]
    assert "trend" in result["staking_performance"]
    
    assert "mean" in result["trading_performance"]
    assert "std" in result["trading_performance"]
    assert "min" in result["trading_performance"]
    assert "max" in result["trading_performance"]
    assert "trend" in result["trading_performance"]
    
    assert "mean" in result["overall_performance"]
    assert "std" in result["overall_performance"]
    assert "min" in result["overall_performance"]
    assert "max" in result["overall_performance"]
    assert "trend" in result["overall_performance"]
    
    # Verify database entry
    db_result = db_session.query(DataProcessingResult).filter_by(
        user_id=test_user.user_id,
        processing_type="PERFORMANCE_DATA_AGGREGATION"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_aggregate_risk_data(db_session, test_user, test_risk_data):
    """Test risk data aggregation"""
    # Aggregate risk data
    result = aggregate_risk_data(
        user_id=test_user.user_id,
        data=test_risk_data,
        db_session=db_session
    )
    
    # Verify aggregation result
    assert isinstance(result, Dict)
    assert "mining_risk" in result
    assert "staking_risk" in result
    assert "trading_risk" in result
    assert "overall_risk" in result
    
    # Verify aggregation details
    assert "mean" in result["mining_risk"]
    assert "std" in result["mining_risk"]
    assert "min" in result["mining_risk"]
    assert "max" in result["mining_risk"]
    assert "trend" in result["mining_risk"]
    
    assert "mean" in result["staking_risk"]
    assert "std" in result["staking_risk"]
    assert "min" in result["staking_risk"]
    assert "max" in result["staking_risk"]
    assert "trend" in result["staking_risk"]
    
    assert "mean" in result["trading_risk"]
    assert "std" in result["trading_risk"]
    assert "min" in result["trading_risk"]
    assert "max" in result["trading_risk"]
    assert "trend" in result["trading_risk"]
    
    assert "mean" in result["overall_risk"]
    assert "std" in result["overall_risk"]
    assert "min" in result["overall_risk"]
    assert "max" in result["overall_risk"]
    assert "trend" in result["overall_risk"]
    
    # Verify database entry
    db_result = db_session.query(DataProcessingResult).filter_by(
        user_id=test_user.user_id,
        processing_type="RISK_DATA_AGGREGATION"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_aggregate_reward_data(db_session, test_user, test_reward_data):
    """Test reward data aggregation"""
    # Aggregate reward data
    result = aggregate_reward_data(
        user_id=test_user.user_id,
        data=test_reward_data,
        db_session=db_session
    )
    
    # Verify aggregation result
    assert isinstance(result, Dict)
    assert "mining_rewards" in result
    assert "staking_rewards" in result
    assert "trading_rewards" in result
    assert "overall_rewards" in result
    
    # Verify aggregation details
    assert "mean" in result["mining_rewards"]
    assert "std" in result["mining_rewards"]
    assert "min" in result["mining_rewards"]
    assert "max" in result["mining_rewards"]
    assert "trend" in result["mining_rewards"]
    
    assert "mean" in result["staking_rewards"]
    assert "std" in result["staking_rewards"]
    assert "min" in result["staking_rewards"]
    assert "max" in result["staking_rewards"]
    assert "trend" in result["staking_rewards"]
    
    assert "mean" in result["trading_rewards"]
    assert "std" in result["trading_rewards"]
    assert "min" in result["trading_rewards"]
    assert "max" in result["trading_rewards"]
    assert "trend" in result["trading_rewards"]
    
    assert "mean" in result["overall_rewards"]
    assert "std" in result["overall_rewards"]
    assert "min" in result["overall_rewards"]
    assert "max" in result["overall_rewards"]
    assert "trend" in result["overall_rewards"]
    
    # Verify database entry
    db_result = db_session.query(DataProcessingResult).filter_by(
        user_id=test_user.user_id,
        processing_type="REWARD_DATA_AGGREGATION"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_analyze_trends(db_session, test_user, test_performance_data, test_risk_data, test_reward_data):
    """Test trend analysis"""
    # Analyze trends
    result = analyze_trends(
        user_id=test_user.user_id,
        performance_data=test_performance_data,
        risk_data=test_risk_data,
        reward_data=test_reward_data,
        db_session=db_session
    )
    
    # Verify trend analysis result
    assert isinstance(result, Dict)
    assert "performance_trends" in result
    assert "risk_trends" in result
    assert "reward_trends" in result
    
    # Verify performance trends
    assert "mining_performance" in result["performance_trends"]
    assert "staking_performance" in result["performance_trends"]
    assert "trading_performance" in result["performance_trends"]
    assert "overall_performance" in result["performance_trends"]
    
    assert "trend" in result["performance_trends"]["mining_performance"]
    assert "slope" in result["performance_trends"]["mining_performance"]
    assert "r2" in result["performance_trends"]["mining_performance"]
    
    assert "trend" in result["performance_trends"]["staking_performance"]
    assert "slope" in result["performance_trends"]["staking_performance"]
    assert "r2" in result["performance_trends"]["staking_performance"]
    
    assert "trend" in result["performance_trends"]["trading_performance"]
    assert "slope" in result["performance_trends"]["trading_performance"]
    assert "r2" in result["performance_trends"]["trading_performance"]
    
    assert "trend" in result["performance_trends"]["overall_performance"]
    assert "slope" in result["performance_trends"]["overall_performance"]
    assert "r2" in result["performance_trends"]["overall_performance"]
    
    # Verify risk trends
    assert "mining_risk" in result["risk_trends"]
    assert "staking_risk" in result["risk_trends"]
    assert "trading_risk" in result["risk_trends"]
    assert "overall_risk" in result["risk_trends"]
    
    assert "trend" in result["risk_trends"]["mining_risk"]
    assert "slope" in result["risk_trends"]["mining_risk"]
    assert "r2" in result["risk_trends"]["mining_risk"]
    
    assert "trend" in result["risk_trends"]["staking_risk"]
    assert "slope" in result["risk_trends"]["staking_risk"]
    assert "r2" in result["risk_trends"]["staking_risk"]
    
    assert "trend" in result["risk_trends"]["trading_risk"]
    assert "slope" in result["risk_trends"]["trading_risk"]
    assert "r2" in result["risk_trends"]["trading_risk"]
    
    assert "trend" in result["risk_trends"]["overall_risk"]
    assert "slope" in result["risk_trends"]["overall_risk"]
    assert "r2" in result["risk_trends"]["overall_risk"]
    
    # Verify reward trends
    assert "mining_rewards" in result["reward_trends"]
    assert "staking_rewards" in result["reward_trends"]
    assert "trading_rewards" in result["reward_trends"]
    assert "overall_rewards" in result["reward_trends"]
    
    assert "trend" in result["reward_trends"]["mining_rewards"]
    assert "slope" in result["reward_trends"]["mining_rewards"]
    assert "r2" in result["reward_trends"]["mining_rewards"]
    
    assert "trend" in result["reward_trends"]["staking_rewards"]
    assert "slope" in result["reward_trends"]["staking_rewards"]
    assert "r2" in result["reward_trends"]["staking_rewards"]
    
    assert "trend" in result["reward_trends"]["trading_rewards"]
    assert "slope" in result["reward_trends"]["trading_rewards"]
    assert "r2" in result["reward_trends"]["trading_rewards"]
    
    assert "trend" in result["reward_trends"]["overall_rewards"]
    assert "slope" in result["reward_trends"]["overall_rewards"]
    assert "r2" in result["reward_trends"]["overall_rewards"]
    
    # Verify database entry
    db_result = db_session.query(DataProcessingResult).filter_by(
        user_id=test_user.user_id,
        processing_type="TREND_ANALYSIS"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_detect_anomalies(db_session, test_user, test_mining_data, test_staking_data, test_trading_data):
    """Test anomaly detection"""
    # Detect anomalies
    result = detect_anomalies(
        user_id=test_user.user_id,
        mining_data=test_mining_data,
        staking_data=test_staking_data,
        trading_data=test_trading_data,
        db_session=db_session
    )
    
    # Verify anomaly detection result
    assert isinstance(result, Dict)
    assert "mining_anomalies" in result
    assert "staking_anomalies" in result
    assert "trading_anomalies" in result
    
    # Verify mining anomalies
    assert "hash_rate" in result["mining_anomalies"]
    assert "power_usage" in result["mining_anomalies"]
    assert "temperature" in result["mining_anomalies"]
    assert "uptime" in result["mining_anomalies"]
    assert "efficiency" in result["mining_anomalies"]
    assert "block_rewards" in result["mining_anomalies"]
    assert "network_difficulty" in result["mining_anomalies"]
    assert "profitability" in result["mining_anomalies"]
    
    assert "anomaly_indices" in result["mining_anomalies"]["hash_rate"]
    assert "anomaly_scores" in result["mining_anomalies"]["hash_rate"]
    assert "threshold" in result["mining_anomalies"]["hash_rate"]
    
    # Verify staking anomalies
    assert "validator_uptime" in result["staking_anomalies"]
    assert "missed_blocks" in result["staking_anomalies"]
    assert "reward_rate" in result["staking_anomalies"]
    assert "peer_count" in result["staking_anomalies"]
    assert "network_participation" in result["staking_anomalies"]
    assert "slashing_events" in result["staking_anomalies"]
    assert "stake_amount" in result["staking_anomalies"]
    assert "validator_count" in result["staking_anomalies"]
    
    assert "anomaly_indices" in result["staking_anomalies"]["validator_uptime"]
    assert "anomaly_scores" in result["staking_anomalies"]["validator_uptime"]
    assert "threshold" in result["staking_anomalies"]["validator_uptime"]
    
    # Verify trading anomalies
    assert "position_size" in result["trading_anomalies"]
    assert "leverage_ratio" in result["trading_anomalies"]
    assert "win_rate" in result["trading_anomalies"]
    assert "profit_loss" in result["trading_anomalies"]
    assert "drawdown" in result["trading_anomalies"]
    assert "sharpe_ratio" in result["trading_anomalies"]
    assert "volume" in result["trading_anomalies"]
    assert "execution_quality" in result["trading_anomalies"]
    
    assert "anomaly_indices" in result["trading_anomalies"]["position_size"]
    assert "anomaly_scores" in result["trading_anomalies"]["position_size"]
    assert "threshold" in result["trading_anomalies"]["position_size"]
    
    # Verify database entry
    db_result = db_session.query(DataProcessingResult).filter_by(
        user_id=test_user.user_id,
        processing_type="ANOMALY_DETECTION"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_calculate_correlations(db_session, test_user, test_mining_data, test_staking_data, test_trading_data):
    """Test correlation calculation"""
    # Calculate correlations
    result = calculate_correlations(
        user_id=test_user.user_id,
        mining_data=test_mining_data,
        staking_data=test_staking_data,
        trading_data=test_trading_data,
        db_session=db_session
    )
    
    # Verify correlation calculation result
    assert isinstance(result, Dict)
    assert "mining_correlations" in result
    assert "staking_correlations" in result
    assert "trading_correlations" in result
    assert "cross_correlations" in result
    
    # Verify mining correlations
    assert isinstance(result["mining_correlations"], pd.DataFrame)
    assert result["mining_correlations"].shape == (8, 8)
    assert "hash_rate" in result["mining_correlations"].index
    assert "power_usage" in result["mining_correlations"].index
    assert "temperature" in result["mining_correlations"].index
    assert "uptime" in result["mining_correlations"].index
    assert "efficiency" in result["mining_correlations"].index
    assert "block_rewards" in result["mining_correlations"].index
    assert "network_difficulty" in result["mining_correlations"].index
    assert "profitability" in result["mining_correlations"].index
    
    # Verify staking correlations
    assert isinstance(result["staking_correlations"], pd.DataFrame)
    assert result["staking_correlations"].shape == (8, 8)
    assert "validator_uptime" in result["staking_correlations"].index
    assert "missed_blocks" in result["staking_correlations"].index
    assert "reward_rate" in result["staking_correlations"].index
    assert "peer_count" in result["staking_correlations"].index
    assert "network_participation" in result["staking_correlations"].index
    assert "slashing_events" in result["staking_correlations"].index
    assert "stake_amount" in result["staking_correlations"].index
    assert "validator_count" in result["staking_correlations"].index
    
    # Verify trading correlations
    assert isinstance(result["trading_correlations"], pd.DataFrame)
    assert result["trading_correlations"].shape == (8, 8)
    assert "position_size" in result["trading_correlations"].index
    assert "leverage_ratio" in result["trading_correlations"].index
    assert "win_rate" in result["trading_correlations"].index
    assert "profit_loss" in result["trading_correlations"].index
    assert "drawdown" in result["trading_correlations"].index
    assert "sharpe_ratio" in result["trading_correlations"].index
    assert "volume" in result["trading_correlations"].index
    assert "execution_quality" in result["trading_correlations"].index
    
    # Verify cross correlations
    assert isinstance(result["cross_correlations"], Dict)
    assert "mining_staking" in result["cross_correlations"]
    assert "mining_trading" in result["cross_correlations"]
    assert "staking_trading" in result["cross_correlations"]
    
    assert isinstance(result["cross_correlations"]["mining_staking"], pd.DataFrame)
    assert isinstance(result["cross_correlations"]["mining_trading"], pd.DataFrame)
    assert isinstance(result["cross_correlations"]["staking_trading"], pd.DataFrame)
    
    # Verify database entry
    db_result = db_session.query(DataProcessingResult).filter_by(
        user_id=test_user.user_id,
        processing_type="CORRELATION_CALCULATION"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_data_processing_error_handling():
    """Test data processing error handling"""
    # Invalid user ID
    with pytest.raises(DataProcessingError) as excinfo:
        transform_mining_data(
            user_id=None,
            data=pd.DataFrame(),
            db_session=None
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data type
    with pytest.raises(DataProcessingError) as excinfo:
        transform_staking_data(
            user_id="test_user",
            data="not a DataFrame",
            db_session=None
        )
    assert "Invalid data type" in str(excinfo.value)
    
    # Empty data
    with pytest.raises(DataProcessingError) as excinfo:
        transform_trading_data(
            user_id="test_user",
            data=pd.DataFrame(),
            db_session=None
        )
    assert "Empty data" in str(excinfo.value)
    
    # Missing required columns
    with pytest.raises(DataProcessingError) as excinfo:
        transform_mining_data(
            user_id="test_user",
            data=pd.DataFrame({"column1": [1, 2, 3]}),
            db_session=None
        )
    assert "Missing required columns" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBDataProcessingError) as excinfo:
        aggregate_performance_data(
            user_id="test_user",
            data=pd.DataFrame(),
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 