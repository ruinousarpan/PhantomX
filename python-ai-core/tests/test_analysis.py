import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from decimal import Decimal
import json
import os
import tempfile
import shutil
import io
import base64
import hashlib
import uuid
from scipy import stats

from core.analysis import (
    analyze_data,
    get_analysis_info,
    AnalysisError
)
from database.models import User, AnalysisRecord
from database.exceptions import AnalysisError as DBAnalysisError

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
def test_performance_data():
    """Create test performance data"""
    # Create data with performance metrics
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
    days = dates.date
    
    return pd.DataFrame({
        "timestamp": dates,
        "day": days,
        "mining_performance": np.random.uniform(0.8, 0.9, 100),
        "staking_performance": np.random.uniform(0.85, 0.95, 100),
        "trading_performance": np.random.uniform(0.7, 0.8, 100),
        "overall_performance": np.random.uniform(0.8, 0.9, 100)
    })

@pytest.fixture
def test_risk_data():
    """Create test risk data"""
    # Create data with risk metrics
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
    days = dates.date
    
    return pd.DataFrame({
        "timestamp": dates,
        "day": days,
        "mining_risk": np.random.uniform(0.1, 0.3, 100),
        "staking_risk": np.random.uniform(0.05, 0.15, 100),
        "trading_risk": np.random.uniform(0.2, 0.4, 100),
        "overall_risk": np.random.uniform(0.1, 0.3, 100)
    })

@pytest.fixture
def test_reward_data():
    """Create test reward data"""
    # Create data with reward metrics
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
    days = dates.date
    
    return pd.DataFrame({
        "timestamp": dates,
        "day": days,
        "mining_reward": np.random.uniform(0.5, 1.0, 100),
        "staking_reward": np.random.uniform(0.6, 1.1, 100),
        "trading_reward": np.random.uniform(0.4, 0.9, 100),
        "overall_reward": np.random.uniform(0.5, 1.0, 100)
    })

@pytest.fixture
def test_activity_data():
    """Create test activity data"""
    # Create data with activity metrics
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
    days = dates.date
    
    return pd.DataFrame({
        "timestamp": dates,
        "day": days,
        "mining_activity": np.random.uniform(0.7, 0.9, 100),
        "staking_activity": np.random.uniform(0.8, 0.95, 100),
        "trading_activity": np.random.uniform(0.6, 0.85, 100),
        "overall_activity": np.random.uniform(0.7, 0.9, 100)
    })

@pytest.fixture
def test_analysis_config():
    """Create test analysis configuration"""
    return {
        "analysis_type": "statistical",
        "metrics": {
            "descriptive": ["mean", "std", "min", "max", "quantile"],
            "inferential": ["ttest", "correlation", "regression"],
            "time_series": ["trend", "seasonality", "stationarity"]
        },
        "params": {
            "quantile": [0.25, 0.5, 0.75],
            "confidence_level": 0.95,
            "window_size": 24  # hours
        }
    }

def test_analyze_performance_data(db_session, test_user, test_performance_data, test_analysis_config):
    """Test analyzing performance data"""
    # Analyze performance data
    result = analyze_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        analysis_config=test_analysis_config,
        db_session=db_session
    )
    
    # Verify analysis result
    assert isinstance(result, Dict)
    assert "analysis_id" in result
    assert "analysis_results" in result
    assert "analysis_details" in result
    
    # Verify analysis metadata
    assert result["analysis_type"] == "statistical"
    assert "descriptive" in result["metrics"]
    assert "inferential" in result["metrics"]
    assert "time_series" in result["metrics"]
    
    # Verify descriptive statistics
    descriptive = result["analysis_results"]["descriptive"]
    for metric in ["mining_performance", "staking_performance", "trading_performance", "overall_performance"]:
        assert metric in descriptive
        assert "mean" in descriptive[metric]
        assert "std" in descriptive[metric]
        assert "min" in descriptive[metric]
        assert "max" in descriptive[metric]
        assert "quantiles" in descriptive[metric]
    
    # Verify inferential statistics
    inferential = result["analysis_results"]["inferential"]
    assert "correlation_matrix" in inferential
    assert "ttest_results" in inferential
    assert "regression_results" in inferential
    
    # Verify time series analysis
    time_series = result["analysis_results"]["time_series"]
    for metric in ["mining_performance", "staking_performance", "trading_performance", "overall_performance"]:
        assert metric in time_series
        assert "trend" in time_series[metric]
        assert "seasonality" in time_series[metric]
        assert "stationarity" in time_series[metric]
    
    # Verify database entry
    db_record = db_session.query(AnalysisRecord).filter_by(
        user_id=test_user.user_id,
        analysis_id=result["analysis_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_analyze_risk_data(db_session, test_user, test_risk_data):
    """Test analyzing risk data"""
    # Create analysis config for risk data
    risk_config = {
        "analysis_type": "risk",
        "metrics": {
            "var": ["historical", "parametric", "monte_carlo"],
            "volatility": ["historical", "ewma", "garch"],
            "tail_risk": ["expected_shortfall", "max_drawdown"]
        },
        "params": {
            "confidence_level": 0.99,
            "time_horizon": 24,  # hours
            "simulation_runs": 1000
        }
    }
    
    # Analyze risk data
    result = analyze_data(
        user_id=test_user.user_id,
        data=test_risk_data,
        analysis_config=risk_config,
        db_session=db_session
    )
    
    # Verify analysis result
    assert isinstance(result, Dict)
    assert "analysis_id" in result
    assert "analysis_results" in result
    assert "analysis_details" in result
    
    # Verify analysis metadata
    assert result["analysis_type"] == "risk"
    assert "var" in result["metrics"]
    assert "volatility" in result["metrics"]
    assert "tail_risk" in result["metrics"]
    
    # Verify VaR analysis
    var_results = result["analysis_results"]["var"]
    for metric in ["mining_risk", "staking_risk", "trading_risk", "overall_risk"]:
        assert metric in var_results
        assert "historical" in var_results[metric]
        assert "parametric" in var_results[metric]
        assert "monte_carlo" in var_results[metric]
    
    # Verify volatility analysis
    volatility_results = result["analysis_results"]["volatility"]
    for metric in ["mining_risk", "staking_risk", "trading_risk", "overall_risk"]:
        assert metric in volatility_results
        assert "historical" in volatility_results[metric]
        assert "ewma" in volatility_results[metric]
        assert "garch" in volatility_results[metric]
    
    # Verify tail risk analysis
    tail_risk_results = result["analysis_results"]["tail_risk"]
    for metric in ["mining_risk", "staking_risk", "trading_risk", "overall_risk"]:
        assert metric in tail_risk_results
        assert "expected_shortfall" in tail_risk_results[metric]
        assert "max_drawdown" in tail_risk_results[metric]
    
    # Verify database entry
    db_record = db_session.query(AnalysisRecord).filter_by(
        user_id=test_user.user_id,
        analysis_id=result["analysis_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_analyze_reward_data(db_session, test_user, test_reward_data):
    """Test analyzing reward data"""
    # Create analysis config for reward data
    reward_config = {
        "analysis_type": "reward",
        "metrics": {
            "performance": ["sharpe_ratio", "sortino_ratio", "information_ratio"],
            "attribution": ["factor_analysis", "attribution_analysis"],
            "efficiency": ["reward_to_risk_ratio", "capture_ratios"]
        },
        "params": {
            "risk_free_rate": 0.02,
            "benchmark_returns": np.random.uniform(0.4, 0.8, 100),
            "factor_model": "custom"
        }
    }
    
    # Analyze reward data
    result = analyze_data(
        user_id=test_user.user_id,
        data=test_reward_data,
        analysis_config=reward_config,
        db_session=db_session
    )
    
    # Verify analysis result
    assert isinstance(result, Dict)
    assert "analysis_id" in result
    assert "analysis_results" in result
    assert "analysis_details" in result
    
    # Verify analysis metadata
    assert result["analysis_type"] == "reward"
    assert "performance" in result["metrics"]
    assert "attribution" in result["metrics"]
    assert "efficiency" in result["metrics"]
    
    # Verify performance metrics
    performance_results = result["analysis_results"]["performance"]
    for metric in ["mining_reward", "staking_reward", "trading_reward", "overall_reward"]:
        assert metric in performance_results
        assert "sharpe_ratio" in performance_results[metric]
        assert "sortino_ratio" in performance_results[metric]
        assert "information_ratio" in performance_results[metric]
    
    # Verify attribution analysis
    attribution_results = result["analysis_results"]["attribution"]
    for metric in ["mining_reward", "staking_reward", "trading_reward", "overall_reward"]:
        assert metric in attribution_results
        assert "factor_analysis" in attribution_results[metric]
        assert "attribution_analysis" in attribution_results[metric]
    
    # Verify efficiency metrics
    efficiency_results = result["analysis_results"]["efficiency"]
    for metric in ["mining_reward", "staking_reward", "trading_reward", "overall_reward"]:
        assert metric in efficiency_results
        assert "reward_to_risk_ratio" in efficiency_results[metric]
        assert "capture_ratios" in efficiency_results[metric]
    
    # Verify database entry
    db_record = db_session.query(AnalysisRecord).filter_by(
        user_id=test_user.user_id,
        analysis_id=result["analysis_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_analyze_activity_data(db_session, test_user, test_activity_data):
    """Test analyzing activity data"""
    # Create analysis config for activity data
    activity_config = {
        "analysis_type": "activity",
        "metrics": {
            "patterns": ["daily", "weekly", "monthly"],
            "engagement": ["active_hours", "peak_times", "consistency"],
            "correlation": ["cross_activity", "temporal"]
        },
        "params": {
            "min_active_threshold": 0.5,
            "time_zone": "UTC",
            "rolling_window": 24  # hours
        }
    }
    
    # Analyze activity data
    result = analyze_data(
        user_id=test_user.user_id,
        data=test_activity_data,
        analysis_config=activity_config,
        db_session=db_session
    )
    
    # Verify analysis result
    assert isinstance(result, Dict)
    assert "analysis_id" in result
    assert "analysis_results" in result
    assert "analysis_details" in result
    
    # Verify analysis metadata
    assert result["analysis_type"] == "activity"
    assert "patterns" in result["metrics"]
    assert "engagement" in result["metrics"]
    assert "correlation" in result["metrics"]
    
    # Verify pattern analysis
    pattern_results = result["analysis_results"]["patterns"]
    for metric in ["mining_activity", "staking_activity", "trading_activity", "overall_activity"]:
        assert metric in pattern_results
        assert "daily" in pattern_results[metric]
        assert "weekly" in pattern_results[metric]
        assert "monthly" in pattern_results[metric]
    
    # Verify engagement analysis
    engagement_results = result["analysis_results"]["engagement"]
    for metric in ["mining_activity", "staking_activity", "trading_activity", "overall_activity"]:
        assert metric in engagement_results
        assert "active_hours" in engagement_results[metric]
        assert "peak_times" in engagement_results[metric]
        assert "consistency" in engagement_results[metric]
    
    # Verify correlation analysis
    correlation_results = result["analysis_results"]["correlation"]
    assert "cross_activity" in correlation_results
    assert "temporal" in correlation_results
    
    # Verify database entry
    db_record = db_session.query(AnalysisRecord).filter_by(
        user_id=test_user.user_id,
        analysis_id=result["analysis_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_analyze_with_custom_metrics(db_session, test_user, test_performance_data):
    """Test analyzing with custom metrics"""
    # Create analysis config with custom metrics
    custom_config = {
        "analysis_type": "custom",
        "metrics": {
            "custom_metrics": {
                "weighted_average": "lambda x: np.average(x, weights=range(len(x)))",
                "rolling_zscore": "lambda x: (x - x.rolling(window=24).mean()) / x.rolling(window=24).std()",
                "momentum": "lambda x: x.diff(periods=24) / x.shift(periods=24)"
            }
        },
        "params": {
            "window_size": 24,
            "min_periods": 12
        }
    }
    
    # Analyze performance data
    result = analyze_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        analysis_config=custom_config,
        db_session=db_session
    )
    
    # Verify analysis result
    assert isinstance(result, Dict)
    assert "analysis_id" in result
    assert "analysis_results" in result
    assert "analysis_details" in result
    
    # Verify analysis metadata
    assert result["analysis_type"] == "custom"
    assert "custom_metrics" in result["metrics"]
    
    # Verify custom metrics results
    custom_results = result["analysis_results"]["custom_metrics"]
    for metric in ["mining_performance", "staking_performance", "trading_performance", "overall_performance"]:
        assert metric in custom_results
        assert "weighted_average" in custom_results[metric]
        assert "rolling_zscore" in custom_results[metric]
        assert "momentum" in custom_results[metric]
    
    # Verify database entry
    db_record = db_session.query(AnalysisRecord).filter_by(
        user_id=test_user.user_id,
        analysis_id=result["analysis_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_analyze_with_multiple_datasets(db_session, test_user, test_performance_data, test_risk_data):
    """Test analyzing with multiple datasets"""
    # Create analysis config for multiple datasets
    multi_dataset_config = {
        "analysis_type": "multi_dataset",
        "metrics": {
            "correlation": ["cross_correlation", "granger_causality"],
            "cointegration": ["engle_granger", "johansen"],
            "risk_adjusted": ["information_ratio", "treynor_ratio"]
        },
        "params": {
            "lag_order": 24,
            "test_type": "c",
            "alpha": 0.05
        }
    }
    
    # Combine datasets
    combined_data = pd.merge(
        test_performance_data,
        test_risk_data,
        on=["timestamp", "day"],
        suffixes=("_performance", "_risk")
    )
    
    # Analyze multiple datasets
    result = analyze_data(
        user_id=test_user.user_id,
        data=combined_data,
        analysis_config=multi_dataset_config,
        db_session=db_session
    )
    
    # Verify analysis result
    assert isinstance(result, Dict)
    assert "analysis_id" in result
    assert "analysis_results" in result
    assert "analysis_details" in result
    
    # Verify analysis metadata
    assert result["analysis_type"] == "multi_dataset"
    assert "correlation" in result["metrics"]
    assert "cointegration" in result["metrics"]
    assert "risk_adjusted" in result["metrics"]
    
    # Verify correlation analysis
    correlation_results = result["analysis_results"]["correlation"]
    assert "cross_correlation" in correlation_results
    assert "granger_causality" in correlation_results
    
    # Verify cointegration analysis
    cointegration_results = result["analysis_results"]["cointegration"]
    assert "engle_granger" in cointegration_results
    assert "johansen" in cointegration_results
    
    # Verify risk-adjusted metrics
    risk_adjusted_results = result["analysis_results"]["risk_adjusted"]
    assert "information_ratio" in risk_adjusted_results
    assert "treynor_ratio" in risk_adjusted_results
    
    # Verify database entry
    db_record = db_session.query(AnalysisRecord).filter_by(
        user_id=test_user.user_id,
        analysis_id=result["analysis_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_get_analysis_info(db_session, test_user, test_performance_data, test_analysis_config):
    """Test analysis info retrieval"""
    # First, analyze performance data
    analysis_result = analyze_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        analysis_config=test_analysis_config,
        db_session=db_session
    )
    
    analysis_id = analysis_result["analysis_id"]
    
    # Get analysis info
    result = get_analysis_info(
        user_id=test_user.user_id,
        analysis_id=analysis_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "analysis_id" in result
    assert result["analysis_id"] == analysis_id
    
    # Verify analysis metadata
    assert result["analysis_type"] == "statistical"
    assert "descriptive" in result["metrics"]
    assert "inferential" in result["metrics"]
    assert "time_series" in result["metrics"]
    
    # Verify analysis details
    assert "analysis_details" in result
    assert isinstance(result["analysis_details"], Dict)
    assert "timestamp" in result["analysis_details"]
    assert "analysis_results" in result["analysis_details"]
    assert "computation_time" in result["analysis_details"]
    assert "parameters" in result["analysis_details"]
    
    # Verify database entry
    db_record = db_session.query(AnalysisRecord).filter_by(
        user_id=test_user.user_id,
        analysis_id=analysis_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_analysis_error_handling(db_session, test_user):
    """Test analysis error handling"""
    # Invalid user ID
    with pytest.raises(AnalysisError) as excinfo:
        analyze_data(
            user_id=None,
            data=pd.DataFrame(),
            analysis_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(AnalysisError) as excinfo:
        analyze_data(
            user_id=test_user.user_id,
            data=None,
            analysis_config={},
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Invalid analysis type
    with pytest.raises(AnalysisError) as excinfo:
        analyze_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            analysis_config={"analysis_type": "invalid_type"},
            db_session=db_session
        )
    assert "Invalid analysis type" in str(excinfo.value)
    
    # Invalid metrics
    with pytest.raises(AnalysisError) as excinfo:
        analyze_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            analysis_config={"analysis_type": "statistical", "metrics": {"invalid_metric": ["mean"]}},
            db_session=db_session
        )
    assert "Invalid metrics" in str(excinfo.value)
    
    # Invalid parameters
    with pytest.raises(AnalysisError) as excinfo:
        analyze_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            analysis_config={"analysis_type": "statistical", "params": {"invalid_param": 0}},
            db_session=db_session
        )
    assert "Invalid parameters" in str(excinfo.value)
    
    # Invalid analysis ID
    with pytest.raises(AnalysisError) as excinfo:
        get_analysis_info(
            user_id=test_user.user_id,
            analysis_id="invalid_analysis_id",
            db_session=db_session
        )
    assert "Invalid analysis ID" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBAnalysisError) as excinfo:
        analyze_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            analysis_config={"analysis_type": "statistical"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 