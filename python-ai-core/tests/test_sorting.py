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

from core.sorting import (
    sort_data,
    get_sort_info,
    SortError
)
from database.models import User, SortRecord
from database.exceptions import SortError as DBSortError

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

@pytest.fixture
def test_activity_data():
    """Create test activity data"""
    return pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="H"),
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
def test_sort_config():
    """Create test sort configuration"""
    return {
        "columns": ["mining_performance", "staking_performance"],
        "ascending": [False, True],
        "na_position": "last",
        "ignore_index": True
    }

def test_sort_performance_data(db_session, test_user, test_performance_data, test_sort_config):
    """Test performance data sorting"""
    # Sort performance data
    result = sort_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        sort_config=test_sort_config,
        db_session=db_session
    )
    
    # Verify sort result
    assert isinstance(result, Dict)
    assert "sort_id" in result
    assert "sorted_data" in result
    
    # Verify sort metadata
    assert result["columns"] == ["mining_performance", "staking_performance"]
    assert result["ascending"] == [False, True]
    assert result["na_position"] == "last"
    assert result["ignore_index"] is True
    
    # Verify sort details
    assert "sort_details" in result
    assert isinstance(result["sort_details"], Dict)
    assert "timestamp" in result["sort_details"]
    assert "original_shape" in result["sort_details"]
    assert "sorted_shape" in result["sort_details"]
    
    # Verify sorted data
    assert isinstance(result["sorted_data"], pd.DataFrame)
    assert "timestamp" in result["sorted_data"].columns
    assert "mining_performance" in result["sorted_data"].columns
    assert "staking_performance" in result["sorted_data"].columns
    assert "trading_performance" in result["sorted_data"].columns
    assert "overall_performance" in result["sorted_data"].columns
    
    # Verify data sorting
    assert len(result["sorted_data"]) == len(test_performance_data)
    assert result["sorted_data"]["mining_performance"].is_monotonic_decreasing
    assert result["sorted_data"]["staking_performance"].is_monotonic_increasing
    
    # Verify database entry
    db_record = db_session.query(SortRecord).filter_by(
        user_id=test_user.user_id,
        sort_id=result["sort_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_sort_risk_data(db_session, test_user, test_risk_data, test_sort_config):
    """Test risk data sorting"""
    # Update sort config for risk data
    risk_config = test_sort_config.copy()
    risk_config["columns"] = ["mining_risk", "staking_risk"]
    
    # Sort risk data
    result = sort_data(
        user_id=test_user.user_id,
        data=test_risk_data,
        sort_config=risk_config,
        db_session=db_session
    )
    
    # Verify sort result
    assert isinstance(result, Dict)
    assert "sort_id" in result
    assert "sorted_data" in result
    
    # Verify sort metadata
    assert result["columns"] == ["mining_risk", "staking_risk"]
    assert result["ascending"] == [False, True]
    assert result["na_position"] == "last"
    assert result["ignore_index"] is True
    
    # Verify sort details
    assert "sort_details" in result
    assert isinstance(result["sort_details"], Dict)
    assert "timestamp" in result["sort_details"]
    assert "original_shape" in result["sort_details"]
    assert "sorted_shape" in result["sort_details"]
    
    # Verify sorted data
    assert isinstance(result["sorted_data"], pd.DataFrame)
    assert "timestamp" in result["sorted_data"].columns
    assert "mining_risk" in result["sorted_data"].columns
    assert "staking_risk" in result["sorted_data"].columns
    assert "trading_risk" in result["sorted_data"].columns
    assert "overall_risk" in result["sorted_data"].columns
    
    # Verify data sorting
    assert len(result["sorted_data"]) == len(test_risk_data)
    assert result["sorted_data"]["mining_risk"].is_monotonic_decreasing
    assert result["sorted_data"]["staking_risk"].is_monotonic_increasing
    
    # Verify database entry
    db_record = db_session.query(SortRecord).filter_by(
        user_id=test_user.user_id,
        sort_id=result["sort_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_sort_reward_data(db_session, test_user, test_reward_data, test_sort_config):
    """Test reward data sorting"""
    # Update sort config for reward data
    reward_config = test_sort_config.copy()
    reward_config["columns"] = ["mining_rewards", "staking_rewards"]
    
    # Sort reward data
    result = sort_data(
        user_id=test_user.user_id,
        data=test_reward_data,
        sort_config=reward_config,
        db_session=db_session
    )
    
    # Verify sort result
    assert isinstance(result, Dict)
    assert "sort_id" in result
    assert "sorted_data" in result
    
    # Verify sort metadata
    assert result["columns"] == ["mining_rewards", "staking_rewards"]
    assert result["ascending"] == [False, True]
    assert result["na_position"] == "last"
    assert result["ignore_index"] is True
    
    # Verify sort details
    assert "sort_details" in result
    assert isinstance(result["sort_details"], Dict)
    assert "timestamp" in result["sort_details"]
    assert "original_shape" in result["sort_details"]
    assert "sorted_shape" in result["sort_details"]
    
    # Verify sorted data
    assert isinstance(result["sorted_data"], pd.DataFrame)
    assert "timestamp" in result["sorted_data"].columns
    assert "mining_rewards" in result["sorted_data"].columns
    assert "staking_rewards" in result["sorted_data"].columns
    assert "trading_rewards" in result["sorted_data"].columns
    assert "overall_rewards" in result["sorted_data"].columns
    
    # Verify data sorting
    assert len(result["sorted_data"]) == len(test_reward_data)
    assert result["sorted_data"]["mining_rewards"].is_monotonic_decreasing
    assert result["sorted_data"]["staking_rewards"].is_monotonic_increasing
    
    # Verify database entry
    db_record = db_session.query(SortRecord).filter_by(
        user_id=test_user.user_id,
        sort_id=result["sort_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_sort_activity_data(db_session, test_user, test_activity_data, test_sort_config):
    """Test activity data sorting"""
    # Update sort config for activity data
    activity_config = test_sort_config.copy()
    activity_config["columns"] = ["mining_activity", "staking_activity"]
    
    # Sort activity data
    result = sort_data(
        user_id=test_user.user_id,
        data=test_activity_data,
        sort_config=activity_config,
        db_session=db_session
    )
    
    # Verify sort result
    assert isinstance(result, Dict)
    assert "sort_id" in result
    assert "sorted_data" in result
    
    # Verify sort metadata
    assert result["columns"] == ["mining_activity", "staking_activity"]
    assert result["ascending"] == [False, True]
    assert result["na_position"] == "last"
    assert result["ignore_index"] is True
    
    # Verify sort details
    assert "sort_details" in result
    assert isinstance(result["sort_details"], Dict)
    assert "timestamp" in result["sort_details"]
    assert "original_shape" in result["sort_details"]
    assert "sorted_shape" in result["sort_details"]
    
    # Verify sorted data
    assert isinstance(result["sorted_data"], pd.DataFrame)
    assert "timestamp" in result["sorted_data"].columns
    assert "mining_activity" in result["sorted_data"].columns
    assert "staking_activity" in result["sorted_data"].columns
    assert "trading_activity" in result["sorted_data"].columns
    assert "overall_activity" in result["sorted_data"].columns
    
    # Verify data sorting
    assert len(result["sorted_data"]) == len(test_activity_data)
    assert result["sorted_data"]["mining_activity"].is_monotonic_decreasing
    assert result["sorted_data"]["staking_activity"].is_monotonic_increasing
    
    # Verify database entry
    db_record = db_session.query(SortRecord).filter_by(
        user_id=test_user.user_id,
        sort_id=result["sort_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_get_sort_info(db_session, test_user, test_performance_data, test_sort_config):
    """Test sort info retrieval"""
    # First, sort performance data
    sort_result = sort_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        sort_config=test_sort_config,
        db_session=db_session
    )
    
    sort_id = sort_result["sort_id"]
    
    # Get sort info
    result = get_sort_info(
        user_id=test_user.user_id,
        sort_id=sort_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "sort_id" in result
    assert result["sort_id"] == sort_id
    
    # Verify sort metadata
    assert result["columns"] == ["mining_performance", "staking_performance"]
    assert result["ascending"] == [False, True]
    assert result["na_position"] == "last"
    assert result["ignore_index"] is True
    
    # Verify sort details
    assert "sort_details" in result
    assert isinstance(result["sort_details"], Dict)
    assert "timestamp" in result["sort_details"]
    assert "original_shape" in result["sort_details"]
    assert "sorted_shape" in result["sort_details"]
    
    # Verify database entry
    db_record = db_session.query(SortRecord).filter_by(
        user_id=test_user.user_id,
        sort_id=sort_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_sort_error_handling(db_session, test_user):
    """Test sort error handling"""
    # Invalid user ID
    with pytest.raises(SortError) as excinfo:
        sort_data(
            user_id=None,
            data=pd.DataFrame(),
            sort_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(SortError) as excinfo:
        sort_data(
            user_id=test_user.user_id,
            data=None,
            sort_config={},
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Invalid sort columns
    with pytest.raises(SortError) as excinfo:
        sort_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            sort_config={"columns": ["invalid_column"]},
            db_session=db_session
        )
    assert "Invalid sort columns" in str(excinfo.value)
    
    # Invalid sort ID
    with pytest.raises(SortError) as excinfo:
        get_sort_info(
            user_id=test_user.user_id,
            sort_id="invalid_sort_id",
            db_session=db_session
        )
    assert "Invalid sort ID" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBSortError) as excinfo:
        sort_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            sort_config={"columns": ["col"]},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 