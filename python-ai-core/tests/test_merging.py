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

from core.merging import (
    merge_data,
    get_merge_info,
    MergeError
)
from database.models import User, MergeRecord
from database.exceptions import MergeError as DBMergeError

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
    # Create data with multiple days to test merging
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
    # Create data with multiple days to test merging
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
    days = dates.date
    
    return pd.DataFrame({
        "timestamp": dates,
        "day": days,
        "mining_risk": np.random.uniform(0.2, 0.3, 100),
        "staking_risk": np.random.uniform(0.1, 0.2, 100),
        "trading_risk": np.random.uniform(0.3, 0.4, 100),
        "overall_risk": np.random.uniform(0.2, 0.3, 100)
    })

@pytest.fixture
def test_reward_data():
    """Create test reward data"""
    # Create data with multiple days to test merging
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
    days = dates.date
    
    return pd.DataFrame({
        "timestamp": dates,
        "day": days,
        "mining_rewards": np.random.uniform(0.4, 0.6, 100),
        "staking_rewards": np.random.uniform(0.1, 0.15, 100),
        "trading_rewards": np.random.uniform(0.05, 0.1, 100),
        "overall_rewards": np.random.uniform(0.2, 0.25, 100)
    })

@pytest.fixture
def test_activity_data():
    """Create test activity data"""
    # Create data with multiple days to test merging
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
    days = dates.date
    
    return pd.DataFrame({
        "timestamp": dates,
        "day": days,
        "mining_activity": np.random.uniform(0.7, 0.9, 100),
        "staking_activity": np.random.uniform(0.8, 0.95, 100),
        "trading_activity": np.random.uniform(0.6, 0.8, 100),
        "overall_activity": np.random.uniform(0.7, 0.85, 100)
    })

@pytest.fixture
def test_analytics_data():
    """Create test analytics data"""
    return {
        "performance": {
            "mining": {"current": 0.85, "trend": "up", "change": 0.05},
            "staking": {"current": 0.9, "trend": "stable", "change": 0.0},
            "trading": {"current": 0.75, "trend": "down", "change": -0.03},
            "overall": {"current": 0.83, "trend": "up", "change": 0.02}
        },
        "risk": {
            "mining": {"current": 0.25, "trend": "down", "change": -0.02},
            "staking": {"current": 0.15, "trend": "stable", "change": 0.0},
            "trading": {"current": 0.35, "trend": "up", "change": 0.03},
            "overall": {"current": 0.25, "trend": "stable", "change": 0.0}
        },
        "reward": {
            "mining": {"current": 0.5, "trend": "up", "change": 0.05},
            "staking": {"current": 0.12, "trend": "stable", "change": 0.0},
            "trading": {"current": 0.08, "trend": "down", "change": -0.02},
            "overall": {"current": 0.23, "trend": "up", "change": 0.03}
        },
        "activity": {
            "mining": {"current": 0.8, "trend": "up", "change": 0.05},
            "staking": {"current": 0.85, "trend": "stable", "change": 0.0},
            "trading": {"current": 0.7, "trend": "down", "change": -0.03},
            "overall": {"current": 0.78, "trend": "up", "change": 0.02}
        }
    }

@pytest.fixture
def test_merge_config():
    """Create test merge configuration"""
    return {
        "merge_type": "inner",
        "on": "day",
        "how": "inner",
        "suffixes": ("_x", "_y"),
        "validate": "1:1"
    }

def test_merge_performance_risk_data(db_session, test_user, test_performance_data, test_risk_data, test_merge_config):
    """Test merging performance and risk data"""
    # Merge performance and risk data
    result = merge_data(
        user_id=test_user.user_id,
        data1=test_performance_data,
        data2=test_risk_data,
        merge_config=test_merge_config,
        db_session=db_session
    )
    
    # Verify merge result
    assert isinstance(result, Dict)
    assert "merge_id" in result
    assert "merged_data" in result
    
    # Verify merge metadata
    assert result["merge_type"] == "inner"
    assert result["on"] == "day"
    assert result["how"] == "inner"
    assert result["suffixes"] == ("_x", "_y")
    assert result["validate"] == "1:1"
    
    # Verify merge details
    assert "merge_details" in result
    assert isinstance(result["merge_details"], Dict)
    assert "timestamp" in result["merge_details"]
    assert "data1_shape" in result["merge_details"]
    assert "data2_shape" in result["merge_details"]
    assert "merged_shape" in result["merge_details"]
    assert "merge_count" in result["merge_details"]
    
    # Verify merged data
    assert isinstance(result["merged_data"], pd.DataFrame)
    assert "day" in result["merged_data"].columns
    assert "mining_performance" in result["merged_data"].columns
    assert "staking_performance" in result["merged_data"].columns
    assert "trading_performance" in result["merged_data"].columns
    assert "overall_performance" in result["merged_data"].columns
    assert "mining_risk" in result["merged_data"].columns
    assert "staking_risk" in result["merged_data"].columns
    assert "trading_risk" in result["merged_data"].columns
    assert "overall_risk" in result["merged_data"].columns
    
    # Verify data merging
    assert len(result["merged_data"]) == len(test_performance_data["day"].unique())
    assert result["merged_data"]["day"].is_monotonic_increasing
    
    # Verify database entry
    db_record = db_session.query(MergeRecord).filter_by(
        user_id=test_user.user_id,
        merge_id=result["merge_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_merge_performance_reward_data(db_session, test_user, test_performance_data, test_reward_data, test_merge_config):
    """Test merging performance and reward data"""
    # Merge performance and reward data
    result = merge_data(
        user_id=test_user.user_id,
        data1=test_performance_data,
        data2=test_reward_data,
        merge_config=test_merge_config,
        db_session=db_session
    )
    
    # Verify merge result
    assert isinstance(result, Dict)
    assert "merge_id" in result
    assert "merged_data" in result
    
    # Verify merge metadata
    assert result["merge_type"] == "inner"
    assert result["on"] == "day"
    assert result["how"] == "inner"
    assert result["suffixes"] == ("_x", "_y")
    assert result["validate"] == "1:1"
    
    # Verify merge details
    assert "merge_details" in result
    assert isinstance(result["merge_details"], Dict)
    assert "timestamp" in result["merge_details"]
    assert "data1_shape" in result["merge_details"]
    assert "data2_shape" in result["merge_details"]
    assert "merged_shape" in result["merge_details"]
    assert "merge_count" in result["merge_details"]
    
    # Verify merged data
    assert isinstance(result["merged_data"], pd.DataFrame)
    assert "day" in result["merged_data"].columns
    assert "mining_performance" in result["merged_data"].columns
    assert "staking_performance" in result["merged_data"].columns
    assert "trading_performance" in result["merged_data"].columns
    assert "overall_performance" in result["merged_data"].columns
    assert "mining_rewards" in result["merged_data"].columns
    assert "staking_rewards" in result["merged_data"].columns
    assert "trading_rewards" in result["merged_data"].columns
    assert "overall_rewards" in result["merged_data"].columns
    
    # Verify data merging
    assert len(result["merged_data"]) == len(test_performance_data["day"].unique())
    assert result["merged_data"]["day"].is_monotonic_increasing
    
    # Verify database entry
    db_record = db_session.query(MergeRecord).filter_by(
        user_id=test_user.user_id,
        merge_id=result["merge_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_merge_performance_activity_data(db_session, test_user, test_performance_data, test_activity_data, test_merge_config):
    """Test merging performance and activity data"""
    # Merge performance and activity data
    result = merge_data(
        user_id=test_user.user_id,
        data1=test_performance_data,
        data2=test_activity_data,
        merge_config=test_merge_config,
        db_session=db_session
    )
    
    # Verify merge result
    assert isinstance(result, Dict)
    assert "merge_id" in result
    assert "merged_data" in result
    
    # Verify merge metadata
    assert result["merge_type"] == "inner"
    assert result["on"] == "day"
    assert result["how"] == "inner"
    assert result["suffixes"] == ("_x", "_y")
    assert result["validate"] == "1:1"
    
    # Verify merge details
    assert "merge_details" in result
    assert isinstance(result["merge_details"], Dict)
    assert "timestamp" in result["merge_details"]
    assert "data1_shape" in result["merge_details"]
    assert "data2_shape" in result["merge_details"]
    assert "merged_shape" in result["merge_details"]
    assert "merge_count" in result["merge_details"]
    
    # Verify merged data
    assert isinstance(result["merged_data"], pd.DataFrame)
    assert "day" in result["merged_data"].columns
    assert "mining_performance" in result["merged_data"].columns
    assert "staking_performance" in result["merged_data"].columns
    assert "trading_performance" in result["merged_data"].columns
    assert "overall_performance" in result["merged_data"].columns
    assert "mining_activity" in result["merged_data"].columns
    assert "staking_activity" in result["merged_data"].columns
    assert "trading_activity" in result["merged_data"].columns
    assert "overall_activity" in result["merged_data"].columns
    
    # Verify data merging
    assert len(result["merged_data"]) == len(test_performance_data["day"].unique())
    assert result["merged_data"]["day"].is_monotonic_increasing
    
    # Verify database entry
    db_record = db_session.query(MergeRecord).filter_by(
        user_id=test_user.user_id,
        merge_id=result["merge_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_merge_all_data(db_session, test_user, test_performance_data, test_risk_data, test_reward_data, test_activity_data, test_merge_config):
    """Test merging all data types"""
    # First merge performance and risk data
    result1 = merge_data(
        user_id=test_user.user_id,
        data1=test_performance_data,
        data2=test_risk_data,
        merge_config=test_merge_config,
        db_session=db_session
    )
    
    # Then merge with reward data
    result2 = merge_data(
        user_id=test_user.user_id,
        data1=result1["merged_data"],
        data2=test_reward_data,
        merge_config=test_merge_config,
        db_session=db_session
    )
    
    # Finally merge with activity data
    result = merge_data(
        user_id=test_user.user_id,
        data1=result2["merged_data"],
        data2=test_activity_data,
        merge_config=test_merge_config,
        db_session=db_session
    )
    
    # Verify merge result
    assert isinstance(result, Dict)
    assert "merge_id" in result
    assert "merged_data" in result
    
    # Verify merge metadata
    assert result["merge_type"] == "inner"
    assert result["on"] == "day"
    assert result["how"] == "inner"
    assert result["suffixes"] == ("_x", "_y")
    assert result["validate"] == "1:1"
    
    # Verify merge details
    assert "merge_details" in result
    assert isinstance(result["merge_details"], Dict)
    assert "timestamp" in result["merge_details"]
    assert "data1_shape" in result["merge_details"]
    assert "data2_shape" in result["merge_details"]
    assert "merged_shape" in result["merge_details"]
    assert "merge_count" in result["merge_details"]
    
    # Verify merged data
    assert isinstance(result["merged_data"], pd.DataFrame)
    assert "day" in result["merged_data"].columns
    assert "mining_performance" in result["merged_data"].columns
    assert "staking_performance" in result["merged_data"].columns
    assert "trading_performance" in result["merged_data"].columns
    assert "overall_performance" in result["merged_data"].columns
    assert "mining_risk" in result["merged_data"].columns
    assert "staking_risk" in result["merged_data"].columns
    assert "trading_risk" in result["merged_data"].columns
    assert "overall_risk" in result["merged_data"].columns
    assert "mining_rewards" in result["merged_data"].columns
    assert "staking_rewards" in result["merged_data"].columns
    assert "trading_rewards" in result["merged_data"].columns
    assert "overall_rewards" in result["merged_data"].columns
    assert "mining_activity" in result["merged_data"].columns
    assert "staking_activity" in result["merged_data"].columns
    assert "trading_activity" in result["merged_data"].columns
    assert "overall_activity" in result["merged_data"].columns
    
    # Verify data merging
    assert len(result["merged_data"]) == len(test_performance_data["day"].unique())
    assert result["merged_data"]["day"].is_monotonic_increasing
    
    # Verify database entry
    db_record = db_session.query(MergeRecord).filter_by(
        user_id=test_user.user_id,
        merge_id=result["merge_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_get_merge_info(db_session, test_user, test_performance_data, test_risk_data, test_merge_config):
    """Test merge info retrieval"""
    # First, merge performance and risk data
    merge_result = merge_data(
        user_id=test_user.user_id,
        data1=test_performance_data,
        data2=test_risk_data,
        merge_config=test_merge_config,
        db_session=db_session
    )
    
    merge_id = merge_result["merge_id"]
    
    # Get merge info
    result = get_merge_info(
        user_id=test_user.user_id,
        merge_id=merge_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "merge_id" in result
    assert result["merge_id"] == merge_id
    
    # Verify merge metadata
    assert result["merge_type"] == "inner"
    assert result["on"] == "day"
    assert result["how"] == "inner"
    assert result["suffixes"] == ("_x", "_y")
    assert result["validate"] == "1:1"
    
    # Verify merge details
    assert "merge_details" in result
    assert isinstance(result["merge_details"], Dict)
    assert "timestamp" in result["merge_details"]
    assert "data1_shape" in result["merge_details"]
    assert "data2_shape" in result["merge_details"]
    assert "merged_shape" in result["merge_details"]
    assert "merge_count" in result["merge_details"]
    
    # Verify database entry
    db_record = db_session.query(MergeRecord).filter_by(
        user_id=test_user.user_id,
        merge_id=merge_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_merge_error_handling(db_session, test_user):
    """Test merge error handling"""
    # Invalid user ID
    with pytest.raises(MergeError) as excinfo:
        merge_data(
            user_id=None,
            data1=pd.DataFrame(),
            data2=pd.DataFrame(),
            merge_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data1
    with pytest.raises(MergeError) as excinfo:
        merge_data(
            user_id=test_user.user_id,
            data1=None,
            data2=pd.DataFrame(),
            merge_config={},
            db_session=db_session
        )
    assert "Invalid data1" in str(excinfo.value)
    
    # Invalid data2
    with pytest.raises(MergeError) as excinfo:
        merge_data(
            user_id=test_user.user_id,
            data1=pd.DataFrame(),
            data2=None,
            merge_config={},
            db_session=db_session
        )
    assert "Invalid data2" in str(excinfo.value)
    
    # Invalid merge on column
    with pytest.raises(MergeError) as excinfo:
        merge_data(
            user_id=test_user.user_id,
            data1=pd.DataFrame({"col1": [1, 2, 3]}),
            data2=pd.DataFrame({"col2": [1, 2, 3]}),
            merge_config={"on": "invalid_column"},
            db_session=db_session
        )
    assert "Invalid merge on column" in str(excinfo.value)
    
    # Invalid merge type
    with pytest.raises(MergeError) as excinfo:
        merge_data(
            user_id=test_user.user_id,
            data1=pd.DataFrame({"col": [1, 2, 3]}),
            data2=pd.DataFrame({"col": [1, 2, 3]}),
            merge_config={"merge_type": "invalid_type"},
            db_session=db_session
        )
    assert "Invalid merge type" in str(excinfo.value)
    
    # Invalid merge ID
    with pytest.raises(MergeError) as excinfo:
        get_merge_info(
            user_id=test_user.user_id,
            merge_id="invalid_merge_id",
            db_session=db_session
        )
    assert "Invalid merge ID" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBMergeError) as excinfo:
        merge_data(
            user_id=test_user.user_id,
            data1=pd.DataFrame({"col": [1, 2, 3]}),
            data2=pd.DataFrame({"col": [1, 2, 3]}),
            merge_config={"on": "col"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 