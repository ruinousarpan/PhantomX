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

from core.grouping import (
    group_data,
    get_group_info,
    GroupError
)
from database.models import User, GroupRecord
from database.exceptions import GroupError as DBGroupError

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
    # Create data with multiple days to test grouping
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
    # Create data with multiple days to test grouping
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
    # Create data with multiple days to test grouping
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
    # Create data with multiple days to test grouping
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
def test_group_config():
    """Create test group configuration"""
    return {
        "group_by": "day",
        "aggregations": {
            "mining_performance": ["mean", "min", "max"],
            "staking_performance": ["mean", "min", "max"],
            "trading_performance": ["mean", "min", "max"],
            "overall_performance": ["mean", "min", "max"]
        },
        "sort_by": "day",
        "ascending": True
    }

def test_group_performance_data(db_session, test_user, test_performance_data, test_group_config):
    """Test performance data grouping"""
    # Group performance data
    result = group_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        group_config=test_group_config,
        db_session=db_session
    )
    
    # Verify group result
    assert isinstance(result, Dict)
    assert "group_id" in result
    assert "grouped_data" in result
    
    # Verify group metadata
    assert result["group_by"] == "day"
    assert result["aggregations"] == test_group_config["aggregations"]
    assert result["sort_by"] == "day"
    assert result["ascending"] is True
    
    # Verify group details
    assert "group_details" in result
    assert isinstance(result["group_details"], Dict)
    assert "timestamp" in result["group_details"]
    assert "original_shape" in result["group_details"]
    assert "grouped_shape" in result["group_details"]
    assert "group_count" in result["group_details"]
    
    # Verify grouped data
    assert isinstance(result["grouped_data"], pd.DataFrame)
    assert "day" in result["grouped_data"].columns
    assert "mining_performance_mean" in result["grouped_data"].columns
    assert "mining_performance_min" in result["grouped_data"].columns
    assert "mining_performance_max" in result["grouped_data"].columns
    assert "staking_performance_mean" in result["grouped_data"].columns
    assert "trading_performance_mean" in result["grouped_data"].columns
    assert "overall_performance_mean" in result["grouped_data"].columns
    
    # Verify data grouping
    assert len(result["grouped_data"]) == len(test_performance_data["day"].unique())
    assert result["grouped_data"]["day"].is_monotonic_increasing
    
    # Verify database entry
    db_record = db_session.query(GroupRecord).filter_by(
        user_id=test_user.user_id,
        group_id=result["group_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_group_risk_data(db_session, test_user, test_risk_data, test_group_config):
    """Test risk data grouping"""
    # Update group config for risk data
    risk_config = test_group_config.copy()
    risk_config["aggregations"] = {
        "mining_risk": ["mean", "min", "max"],
        "staking_risk": ["mean", "min", "max"],
        "trading_risk": ["mean", "min", "max"],
        "overall_risk": ["mean", "min", "max"]
    }
    
    # Group risk data
    result = group_data(
        user_id=test_user.user_id,
        data=test_risk_data,
        group_config=risk_config,
        db_session=db_session
    )
    
    # Verify group result
    assert isinstance(result, Dict)
    assert "group_id" in result
    assert "grouped_data" in result
    
    # Verify group metadata
    assert result["group_by"] == "day"
    assert result["aggregations"] == risk_config["aggregations"]
    assert result["sort_by"] == "day"
    assert result["ascending"] is True
    
    # Verify group details
    assert "group_details" in result
    assert isinstance(result["group_details"], Dict)
    assert "timestamp" in result["group_details"]
    assert "original_shape" in result["group_details"]
    assert "grouped_shape" in result["group_details"]
    assert "group_count" in result["group_details"]
    
    # Verify grouped data
    assert isinstance(result["grouped_data"], pd.DataFrame)
    assert "day" in result["grouped_data"].columns
    assert "mining_risk_mean" in result["grouped_data"].columns
    assert "mining_risk_min" in result["grouped_data"].columns
    assert "mining_risk_max" in result["grouped_data"].columns
    assert "staking_risk_mean" in result["grouped_data"].columns
    assert "trading_risk_mean" in result["grouped_data"].columns
    assert "overall_risk_mean" in result["grouped_data"].columns
    
    # Verify data grouping
    assert len(result["grouped_data"]) == len(test_risk_data["day"].unique())
    assert result["grouped_data"]["day"].is_monotonic_increasing
    
    # Verify database entry
    db_record = db_session.query(GroupRecord).filter_by(
        user_id=test_user.user_id,
        group_id=result["group_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_group_reward_data(db_session, test_user, test_reward_data, test_group_config):
    """Test reward data grouping"""
    # Update group config for reward data
    reward_config = test_group_config.copy()
    reward_config["aggregations"] = {
        "mining_rewards": ["mean", "min", "max"],
        "staking_rewards": ["mean", "min", "max"],
        "trading_rewards": ["mean", "min", "max"],
        "overall_rewards": ["mean", "min", "max"]
    }
    
    # Group reward data
    result = group_data(
        user_id=test_user.user_id,
        data=test_reward_data,
        group_config=reward_config,
        db_session=db_session
    )
    
    # Verify group result
    assert isinstance(result, Dict)
    assert "group_id" in result
    assert "grouped_data" in result
    
    # Verify group metadata
    assert result["group_by"] == "day"
    assert result["aggregations"] == reward_config["aggregations"]
    assert result["sort_by"] == "day"
    assert result["ascending"] is True
    
    # Verify group details
    assert "group_details" in result
    assert isinstance(result["group_details"], Dict)
    assert "timestamp" in result["group_details"]
    assert "original_shape" in result["group_details"]
    assert "grouped_shape" in result["group_details"]
    assert "group_count" in result["group_details"]
    
    # Verify grouped data
    assert isinstance(result["grouped_data"], pd.DataFrame)
    assert "day" in result["grouped_data"].columns
    assert "mining_rewards_mean" in result["grouped_data"].columns
    assert "mining_rewards_min" in result["grouped_data"].columns
    assert "mining_rewards_max" in result["grouped_data"].columns
    assert "staking_rewards_mean" in result["grouped_data"].columns
    assert "trading_rewards_mean" in result["grouped_data"].columns
    assert "overall_rewards_mean" in result["grouped_data"].columns
    
    # Verify data grouping
    assert len(result["grouped_data"]) == len(test_reward_data["day"].unique())
    assert result["grouped_data"]["day"].is_monotonic_increasing
    
    # Verify database entry
    db_record = db_session.query(GroupRecord).filter_by(
        user_id=test_user.user_id,
        group_id=result["group_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_group_activity_data(db_session, test_user, test_activity_data, test_group_config):
    """Test activity data grouping"""
    # Update group config for activity data
    activity_config = test_group_config.copy()
    activity_config["aggregations"] = {
        "mining_activity": ["mean", "min", "max"],
        "staking_activity": ["mean", "min", "max"],
        "trading_activity": ["mean", "min", "max"],
        "overall_activity": ["mean", "min", "max"]
    }
    
    # Group activity data
    result = group_data(
        user_id=test_user.user_id,
        data=test_activity_data,
        group_config=activity_config,
        db_session=db_session
    )
    
    # Verify group result
    assert isinstance(result, Dict)
    assert "group_id" in result
    assert "grouped_data" in result
    
    # Verify group metadata
    assert result["group_by"] == "day"
    assert result["aggregations"] == activity_config["aggregations"]
    assert result["sort_by"] == "day"
    assert result["ascending"] is True
    
    # Verify group details
    assert "group_details" in result
    assert isinstance(result["group_details"], Dict)
    assert "timestamp" in result["group_details"]
    assert "original_shape" in result["group_details"]
    assert "grouped_shape" in result["group_details"]
    assert "group_count" in result["group_details"]
    
    # Verify grouped data
    assert isinstance(result["grouped_data"], pd.DataFrame)
    assert "day" in result["grouped_data"].columns
    assert "mining_activity_mean" in result["grouped_data"].columns
    assert "mining_activity_min" in result["grouped_data"].columns
    assert "mining_activity_max" in result["grouped_data"].columns
    assert "staking_activity_mean" in result["grouped_data"].columns
    assert "trading_activity_mean" in result["grouped_data"].columns
    assert "overall_activity_mean" in result["grouped_data"].columns
    
    # Verify data grouping
    assert len(result["grouped_data"]) == len(test_activity_data["day"].unique())
    assert result["grouped_data"]["day"].is_monotonic_increasing
    
    # Verify database entry
    db_record = db_session.query(GroupRecord).filter_by(
        user_id=test_user.user_id,
        group_id=result["group_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_get_group_info(db_session, test_user, test_performance_data, test_group_config):
    """Test group info retrieval"""
    # First, group performance data
    group_result = group_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        group_config=test_group_config,
        db_session=db_session
    )
    
    group_id = group_result["group_id"]
    
    # Get group info
    result = get_group_info(
        user_id=test_user.user_id,
        group_id=group_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "group_id" in result
    assert result["group_id"] == group_id
    
    # Verify group metadata
    assert result["group_by"] == "day"
    assert result["aggregations"] == test_group_config["aggregations"]
    assert result["sort_by"] == "day"
    assert result["ascending"] is True
    
    # Verify group details
    assert "group_details" in result
    assert isinstance(result["group_details"], Dict)
    assert "timestamp" in result["group_details"]
    assert "original_shape" in result["group_details"]
    assert "grouped_shape" in result["group_details"]
    assert "group_count" in result["group_details"]
    
    # Verify database entry
    db_record = db_session.query(GroupRecord).filter_by(
        user_id=test_user.user_id,
        group_id=group_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_group_error_handling(db_session, test_user):
    """Test group error handling"""
    # Invalid user ID
    with pytest.raises(GroupError) as excinfo:
        group_data(
            user_id=None,
            data=pd.DataFrame(),
            group_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(GroupError) as excinfo:
        group_data(
            user_id=test_user.user_id,
            data=None,
            group_config={},
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Invalid group by column
    with pytest.raises(GroupError) as excinfo:
        group_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            group_config={"group_by": "invalid_column"},
            db_session=db_session
        )
    assert "Invalid group by column" in str(excinfo.value)
    
    # Invalid aggregation
    with pytest.raises(GroupError) as excinfo:
        group_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            group_config={"group_by": "col", "aggregations": {"col": ["invalid_agg"]}},
            db_session=db_session
        )
    assert "Invalid aggregation" in str(excinfo.value)
    
    # Invalid group ID
    with pytest.raises(GroupError) as excinfo:
        get_group_info(
            user_id=test_user.user_id,
            group_id="invalid_group_id",
            db_session=db_session
        )
    assert "Invalid group ID" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBGroupError) as excinfo:
        group_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            group_config={"group_by": "col"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 