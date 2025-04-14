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

from core.aggregation import (
    aggregate_data,
    get_aggregation_info,
    AggregationError
)
from database.models import User, AggregationRecord
from database.exceptions import AggregationError as DBAggregationError

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
def test_aggregation_config():
    """Create test aggregation configuration"""
    return {
        "aggregation_type": "time",
        "group_by": ["day"],
        "metrics": {
            "mining_performance": ["mean", "min", "max"],
            "staking_performance": ["mean", "min", "max"],
            "trading_performance": ["mean", "min", "max"],
            "overall_performance": ["mean", "min", "max"]
        }
    }

def test_aggregate_performance_data(db_session, test_user, test_performance_data, test_aggregation_config):
    """Test aggregating performance data"""
    # Aggregate performance data
    result = aggregate_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        aggregation_config=test_aggregation_config,
        db_session=db_session
    )
    
    # Verify aggregation result
    assert isinstance(result, Dict)
    assert "aggregation_id" in result
    assert "aggregated_data" in result
    assert "aggregation_details" in result
    
    # Verify aggregation metadata
    assert result["aggregation_type"] == "time"
    assert result["group_by"] == ["day"]
    assert result["metrics"] == {
        "mining_performance": ["mean", "min", "max"],
        "staking_performance": ["mean", "min", "max"],
        "trading_performance": ["mean", "min", "max"],
        "overall_performance": ["mean", "min", "max"]
    }
    
    # Verify aggregation details
    assert isinstance(result["aggregation_details"], Dict)
    assert "timestamp" in result["aggregation_details"]
    assert "aggregation_results" in result["aggregation_details"]
    assert "group_counts" in result["aggregation_details"]
    assert "group_stats" in result["aggregation_details"]
    
    # Verify aggregation results
    assert isinstance(result["aggregation_details"]["aggregation_results"], Dict)
    for metric in ["mining_performance", "staking_performance", "trading_performance", "overall_performance"]:
        for agg in ["mean", "min", "max"]:
            assert f"{metric}_{agg}" in result["aggregated_data"].columns
    
    # Verify aggregated data
    assert isinstance(result["aggregated_data"], pd.DataFrame)
    assert "day" in result["aggregated_data"].columns
    assert len(result["aggregated_data"]) < len(test_performance_data)
    
    # Verify database entry
    db_record = db_session.query(AggregationRecord).filter_by(
        user_id=test_user.user_id,
        aggregation_id=result["aggregation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_aggregate_risk_data(db_session, test_user, test_risk_data):
    """Test aggregating risk data"""
    # Create aggregation config for risk data
    risk_config = {
        "aggregation_type": "time",
        "group_by": ["day"],
        "metrics": {
            "mining_risk": ["mean", "std", "quantile"],
            "staking_risk": ["mean", "std", "quantile"],
            "trading_risk": ["mean", "std", "quantile"],
            "overall_risk": ["mean", "std", "quantile"]
        },
        "params": {
            "quantile": [0.25, 0.5, 0.75]
        }
    }
    
    # Aggregate risk data
    result = aggregate_data(
        user_id=test_user.user_id,
        data=test_risk_data,
        aggregation_config=risk_config,
        db_session=db_session
    )
    
    # Verify aggregation result
    assert isinstance(result, Dict)
    assert "aggregation_id" in result
    assert "aggregated_data" in result
    assert "aggregation_details" in result
    
    # Verify aggregation metadata
    assert result["aggregation_type"] == "time"
    assert result["group_by"] == ["day"]
    assert result["metrics"] == {
        "mining_risk": ["mean", "std", "quantile"],
        "staking_risk": ["mean", "std", "quantile"],
        "trading_risk": ["mean", "std", "quantile"],
        "overall_risk": ["mean", "std", "quantile"]
    }
    
    # Verify aggregation details
    assert isinstance(result["aggregation_details"], Dict)
    assert "timestamp" in result["aggregation_details"]
    assert "aggregation_results" in result["aggregation_details"]
    assert "group_counts" in result["aggregation_details"]
    assert "group_stats" in result["aggregation_details"]
    
    # Verify aggregation results
    assert isinstance(result["aggregation_details"]["aggregation_results"], Dict)
    for metric in ["mining_risk", "staking_risk", "trading_risk", "overall_risk"]:
        for agg in ["mean", "std"]:
            assert f"{metric}_{agg}" in result["aggregated_data"].columns
        for q in [0.25, 0.5, 0.75]:
            assert f"{metric}_quantile_{q}" in result["aggregated_data"].columns
    
    # Verify aggregated data
    assert isinstance(result["aggregated_data"], pd.DataFrame)
    assert "day" in result["aggregated_data"].columns
    assert len(result["aggregated_data"]) < len(test_risk_data)
    
    # Verify database entry
    db_record = db_session.query(AggregationRecord).filter_by(
        user_id=test_user.user_id,
        aggregation_id=result["aggregation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_aggregate_reward_data(db_session, test_user, test_reward_data):
    """Test aggregating reward data"""
    # Create aggregation config for reward data
    reward_config = {
        "aggregation_type": "time",
        "group_by": ["day"],
        "metrics": {
            "mining_reward": ["sum", "count", "mean"],
            "staking_reward": ["sum", "count", "mean"],
            "trading_reward": ["sum", "count", "mean"],
            "overall_reward": ["sum", "count", "mean"]
        }
    }
    
    # Aggregate reward data
    result = aggregate_data(
        user_id=test_user.user_id,
        data=test_reward_data,
        aggregation_config=reward_config,
        db_session=db_session
    )
    
    # Verify aggregation result
    assert isinstance(result, Dict)
    assert "aggregation_id" in result
    assert "aggregated_data" in result
    assert "aggregation_details" in result
    
    # Verify aggregation metadata
    assert result["aggregation_type"] == "time"
    assert result["group_by"] == ["day"]
    assert result["metrics"] == {
        "mining_reward": ["sum", "count", "mean"],
        "staking_reward": ["sum", "count", "mean"],
        "trading_reward": ["sum", "count", "mean"],
        "overall_reward": ["sum", "count", "mean"]
    }
    
    # Verify aggregation details
    assert isinstance(result["aggregation_details"], Dict)
    assert "timestamp" in result["aggregation_details"]
    assert "aggregation_results" in result["aggregation_details"]
    assert "group_counts" in result["aggregation_details"]
    assert "group_stats" in result["aggregation_details"]
    
    # Verify aggregation results
    assert isinstance(result["aggregation_details"]["aggregation_results"], Dict)
    for metric in ["mining_reward", "staking_reward", "trading_reward", "overall_reward"]:
        for agg in ["sum", "count", "mean"]:
            assert f"{metric}_{agg}" in result["aggregated_data"].columns
    
    # Verify aggregated data
    assert isinstance(result["aggregated_data"], pd.DataFrame)
    assert "day" in result["aggregated_data"].columns
    assert len(result["aggregated_data"]) < len(test_reward_data)
    
    # Verify database entry
    db_record = db_session.query(AggregationRecord).filter_by(
        user_id=test_user.user_id,
        aggregation_id=result["aggregation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_aggregate_activity_data(db_session, test_user, test_activity_data):
    """Test aggregating activity data"""
    # Create aggregation config for activity data
    activity_config = {
        "aggregation_type": "time",
        "group_by": ["day"],
        "metrics": {
            "mining_activity": ["mean", "sum", "count"],
            "staking_activity": ["mean", "sum", "count"],
            "trading_activity": ["mean", "sum", "count"],
            "overall_activity": ["mean", "sum", "count"]
        }
    }
    
    # Aggregate activity data
    result = aggregate_data(
        user_id=test_user.user_id,
        data=test_activity_data,
        aggregation_config=activity_config,
        db_session=db_session
    )
    
    # Verify aggregation result
    assert isinstance(result, Dict)
    assert "aggregation_id" in result
    assert "aggregated_data" in result
    assert "aggregation_details" in result
    
    # Verify aggregation metadata
    assert result["aggregation_type"] == "time"
    assert result["group_by"] == ["day"]
    assert result["metrics"] == {
        "mining_activity": ["mean", "sum", "count"],
        "staking_activity": ["mean", "sum", "count"],
        "trading_activity": ["mean", "sum", "count"],
        "overall_activity": ["mean", "sum", "count"]
    }
    
    # Verify aggregation details
    assert isinstance(result["aggregation_details"], Dict)
    assert "timestamp" in result["aggregation_details"]
    assert "aggregation_results" in result["aggregation_details"]
    assert "group_counts" in result["aggregation_details"]
    assert "group_stats" in result["aggregation_details"]
    
    # Verify aggregation results
    assert isinstance(result["aggregation_details"]["aggregation_results"], Dict)
    for metric in ["mining_activity", "staking_activity", "trading_activity", "overall_activity"]:
        for agg in ["mean", "sum", "count"]:
            assert f"{metric}_{agg}" in result["aggregated_data"].columns
    
    # Verify aggregated data
    assert isinstance(result["aggregated_data"], pd.DataFrame)
    assert "day" in result["aggregated_data"].columns
    assert len(result["aggregated_data"]) < len(test_activity_data)
    
    # Verify database entry
    db_record = db_session.query(AggregationRecord).filter_by(
        user_id=test_user.user_id,
        aggregation_id=result["aggregation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_aggregate_with_custom_function(db_session, test_user, test_performance_data):
    """Test aggregating with custom function"""
    # Create aggregation config with custom function
    custom_config = {
        "aggregation_type": "custom",
        "group_by": ["day"],
        "metrics": {
            "mining_performance": ["custom"],
            "staking_performance": ["custom"],
            "trading_performance": ["custom"],
            "overall_performance": ["custom"]
        },
        "functions": {
            "custom": "lambda x: np.percentile(x, [25, 50, 75])"
        }
    }
    
    # Aggregate performance data
    result = aggregate_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        aggregation_config=custom_config,
        db_session=db_session
    )
    
    # Verify aggregation result
    assert isinstance(result, Dict)
    assert "aggregation_id" in result
    assert "aggregated_data" in result
    assert "aggregation_details" in result
    
    # Verify aggregation metadata
    assert result["aggregation_type"] == "custom"
    assert result["group_by"] == ["day"]
    assert result["metrics"] == {
        "mining_performance": ["custom"],
        "staking_performance": ["custom"],
        "trading_performance": ["custom"],
        "overall_performance": ["custom"]
    }
    
    # Verify aggregation details
    assert isinstance(result["aggregation_details"], Dict)
    assert "timestamp" in result["aggregation_details"]
    assert "aggregation_results" in result["aggregation_details"]
    assert "group_counts" in result["aggregation_details"]
    assert "group_stats" in result["aggregation_details"]
    
    # Verify aggregation results
    assert isinstance(result["aggregation_details"]["aggregation_results"], Dict)
    for metric in ["mining_performance", "staking_performance", "trading_performance", "overall_performance"]:
        for percentile in [25, 50, 75]:
            assert f"{metric}_percentile_{percentile}" in result["aggregated_data"].columns
    
    # Verify aggregated data
    assert isinstance(result["aggregated_data"], pd.DataFrame)
    assert "day" in result["aggregated_data"].columns
    assert len(result["aggregated_data"]) < len(test_performance_data)
    
    # Verify database entry
    db_record = db_session.query(AggregationRecord).filter_by(
        user_id=test_user.user_id,
        aggregation_id=result["aggregation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_aggregate_with_multiple_groups(db_session, test_user, test_performance_data):
    """Test aggregating with multiple groups"""
    # Add additional grouping column
    test_performance_data["category"] = np.random.choice(["A", "B", "C"], size=len(test_performance_data))
    
    # Create aggregation config with multiple groups
    multi_group_config = {
        "aggregation_type": "time",
        "group_by": ["day", "category"],
        "metrics": {
            "mining_performance": ["mean", "min", "max"],
            "staking_performance": ["mean", "min", "max"],
            "trading_performance": ["mean", "min", "max"],
            "overall_performance": ["mean", "min", "max"]
        }
    }
    
    # Aggregate performance data
    result = aggregate_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        aggregation_config=multi_group_config,
        db_session=db_session
    )
    
    # Verify aggregation result
    assert isinstance(result, Dict)
    assert "aggregation_id" in result
    assert "aggregated_data" in result
    assert "aggregation_details" in result
    
    # Verify aggregation metadata
    assert result["aggregation_type"] == "time"
    assert result["group_by"] == ["day", "category"]
    assert result["metrics"] == {
        "mining_performance": ["mean", "min", "max"],
        "staking_performance": ["mean", "min", "max"],
        "trading_performance": ["mean", "min", "max"],
        "overall_performance": ["mean", "min", "max"]
    }
    
    # Verify aggregation details
    assert isinstance(result["aggregation_details"], Dict)
    assert "timestamp" in result["aggregation_details"]
    assert "aggregation_results" in result["aggregation_details"]
    assert "group_counts" in result["aggregation_details"]
    assert "group_stats" in result["aggregation_details"]
    
    # Verify aggregation results
    assert isinstance(result["aggregation_details"]["aggregation_results"], Dict)
    for metric in ["mining_performance", "staking_performance", "trading_performance", "overall_performance"]:
        for agg in ["mean", "min", "max"]:
            assert f"{metric}_{agg}" in result["aggregated_data"].columns
    
    # Verify aggregated data
    assert isinstance(result["aggregated_data"], pd.DataFrame)
    assert "day" in result["aggregated_data"].columns
    assert "category" in result["aggregated_data"].columns
    assert len(result["aggregated_data"]) < len(test_performance_data)
    
    # Verify database entry
    db_record = db_session.query(AggregationRecord).filter_by(
        user_id=test_user.user_id,
        aggregation_id=result["aggregation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_get_aggregation_info(db_session, test_user, test_performance_data, test_aggregation_config):
    """Test aggregation info retrieval"""
    # First, aggregate performance data
    aggregation_result = aggregate_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        aggregation_config=test_aggregation_config,
        db_session=db_session
    )
    
    aggregation_id = aggregation_result["aggregation_id"]
    
    # Get aggregation info
    result = get_aggregation_info(
        user_id=test_user.user_id,
        aggregation_id=aggregation_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "aggregation_id" in result
    assert result["aggregation_id"] == aggregation_id
    
    # Verify aggregation metadata
    assert result["aggregation_type"] == "time"
    assert result["group_by"] == ["day"]
    assert result["metrics"] == {
        "mining_performance": ["mean", "min", "max"],
        "staking_performance": ["mean", "min", "max"],
        "trading_performance": ["mean", "min", "max"],
        "overall_performance": ["mean", "min", "max"]
    }
    
    # Verify aggregation details
    assert "aggregation_details" in result
    assert isinstance(result["aggregation_details"], Dict)
    assert "timestamp" in result["aggregation_details"]
    assert "aggregation_results" in result["aggregation_details"]
    assert "group_counts" in result["aggregation_details"]
    assert "group_stats" in result["aggregation_details"]
    
    # Verify database entry
    db_record = db_session.query(AggregationRecord).filter_by(
        user_id=test_user.user_id,
        aggregation_id=aggregation_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_aggregation_error_handling(db_session, test_user):
    """Test aggregation error handling"""
    # Invalid user ID
    with pytest.raises(AggregationError) as excinfo:
        aggregate_data(
            user_id=None,
            data=pd.DataFrame(),
            aggregation_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(AggregationError) as excinfo:
        aggregate_data(
            user_id=test_user.user_id,
            data=None,
            aggregation_config={},
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Invalid aggregation type
    with pytest.raises(AggregationError) as excinfo:
        aggregate_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            aggregation_config={"aggregation_type": "invalid_type"},
            db_session=db_session
        )
    assert "Invalid aggregation type" in str(excinfo.value)
    
    # Invalid group by
    with pytest.raises(AggregationError) as excinfo:
        aggregate_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            aggregation_config={"aggregation_type": "time", "group_by": ["invalid_column"]},
            db_session=db_session
        )
    assert "Invalid group by columns" in str(excinfo.value)
    
    # Invalid metrics
    with pytest.raises(AggregationError) as excinfo:
        aggregate_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            aggregation_config={"aggregation_type": "time", "group_by": ["col"], "metrics": {"invalid_metric": ["mean"]}},
            db_session=db_session
        )
    assert "Invalid metrics" in str(excinfo.value)
    
    # Invalid aggregation ID
    with pytest.raises(AggregationError) as excinfo:
        get_aggregation_info(
            user_id=test_user.user_id,
            aggregation_id="invalid_aggregation_id",
            db_session=db_session
        )
    assert "Invalid aggregation ID" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBAggregationError) as excinfo:
        aggregate_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            aggregation_config={"aggregation_type": "time"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 