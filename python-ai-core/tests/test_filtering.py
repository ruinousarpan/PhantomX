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

from core.filtering import (
    filter_data,
    query_data,
    get_filter_info,
    FilterError
)
from database.models import User, FilterRecord
from database.exceptions import FilterError as DBFilterError

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
def test_filter_config():
    """Create test filter configuration"""
    return {
        "filter_type": "range",
        "columns": None,  # Filter all columns
        "exclude_columns": ["timestamp", "day"],
        "conditions": {
            "mining_performance": {"min": 0.85, "max": 0.9},
            "staking_performance": {"min": 0.9, "max": 0.95},
            "trading_performance": {"min": 0.75, "max": 0.8},
            "overall_performance": {"min": 0.85, "max": 0.9}
        }
    }

def test_filter_performance_data(db_session, test_user, test_performance_data, test_filter_config):
    """Test filtering performance data"""
    # Filter performance data
    result = filter_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        filter_config=test_filter_config,
        db_session=db_session
    )
    
    # Verify filter result
    assert isinstance(result, Dict)
    assert "filter_id" in result
    assert "filtered_data" in result
    assert "filter_details" in result
    
    # Verify filter metadata
    assert result["filter_type"] == "range"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["conditions"] == {
        "mining_performance": {"min": 0.85, "max": 0.9},
        "staking_performance": {"min": 0.9, "max": 0.95},
        "trading_performance": {"min": 0.75, "max": 0.8},
        "overall_performance": {"min": 0.85, "max": 0.9}
    }
    
    # Verify filter details
    assert isinstance(result["filter_details"], Dict)
    assert "timestamp" in result["filter_details"]
    assert "filter_results" in result["filter_details"]
    assert "filtered_rows" in result["filter_details"]
    assert "filtered_columns" in result["filter_details"]
    
    # Verify filter results
    assert isinstance(result["filter_details"]["filter_results"], Dict)
    assert "mining_performance" in result["filter_details"]["filter_results"]
    assert "staking_performance" in result["filter_details"]["filter_results"]
    assert "trading_performance" in result["filter_details"]["filter_results"]
    assert "overall_performance" in result["filter_details"]["filter_results"]
    
    # Verify filtered data
    assert isinstance(result["filtered_data"], pd.DataFrame)
    assert "mining_performance" in result["filtered_data"].columns
    assert "staking_performance" in result["filtered_data"].columns
    assert "trading_performance" in result["filtered_data"].columns
    assert "overall_performance" in result["filtered_data"].columns
    assert "timestamp" in result["filtered_data"].columns
    assert "day" in result["filtered_data"].columns
    
    # Verify data filtering
    assert len(result["filtered_data"]) <= len(test_performance_data)
    
    # Verify database entry
    db_record = db_session.query(FilterRecord).filter_by(
        user_id=test_user.user_id,
        filter_id=result["filter_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_filter_risk_data(db_session, test_user, test_risk_data):
    """Test filtering risk data"""
    # Create filter config for risk data
    risk_config = {
        "filter_type": "range",
        "columns": None,  # Filter all columns
        "exclude_columns": ["timestamp", "day"],
        "conditions": {
            "mining_risk": {"min": 0.15, "max": 0.25},
            "staking_risk": {"min": 0.08, "max": 0.12},
            "trading_risk": {"min": 0.25, "max": 0.35},
            "overall_risk": {"min": 0.15, "max": 0.25}
        }
    }
    
    # Filter risk data
    result = filter_data(
        user_id=test_user.user_id,
        data=test_risk_data,
        filter_config=risk_config,
        db_session=db_session
    )
    
    # Verify filter result
    assert isinstance(result, Dict)
    assert "filter_id" in result
    assert "filtered_data" in result
    assert "filter_details" in result
    
    # Verify filter metadata
    assert result["filter_type"] == "range"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["conditions"] == {
        "mining_risk": {"min": 0.15, "max": 0.25},
        "staking_risk": {"min": 0.08, "max": 0.12},
        "trading_risk": {"min": 0.25, "max": 0.35},
        "overall_risk": {"min": 0.15, "max": 0.25}
    }
    
    # Verify filter details
    assert isinstance(result["filter_details"], Dict)
    assert "timestamp" in result["filter_details"]
    assert "filter_results" in result["filter_details"]
    assert "filtered_rows" in result["filter_details"]
    assert "filtered_columns" in result["filter_details"]
    
    # Verify filter results
    assert isinstance(result["filter_details"]["filter_results"], Dict)
    assert "mining_risk" in result["filter_details"]["filter_results"]
    assert "staking_risk" in result["filter_details"]["filter_results"]
    assert "trading_risk" in result["filter_details"]["filter_results"]
    assert "overall_risk" in result["filter_details"]["filter_results"]
    
    # Verify filtered data
    assert isinstance(result["filtered_data"], pd.DataFrame)
    assert "mining_risk" in result["filtered_data"].columns
    assert "staking_risk" in result["filtered_data"].columns
    assert "trading_risk" in result["filtered_data"].columns
    assert "overall_risk" in result["filtered_data"].columns
    assert "timestamp" in result["filtered_data"].columns
    assert "day" in result["filtered_data"].columns
    
    # Verify data filtering
    assert len(result["filtered_data"]) <= len(test_risk_data)
    
    # Verify database entry
    db_record = db_session.query(FilterRecord).filter_by(
        user_id=test_user.user_id,
        filter_id=result["filter_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_filter_reward_data(db_session, test_user, test_reward_data):
    """Test filtering reward data"""
    # Create filter config for reward data
    reward_config = {
        "filter_type": "range",
        "columns": None,  # Filter all columns
        "exclude_columns": ["timestamp", "day"],
        "conditions": {
            "mining_reward": {"min": 0.7, "max": 0.9},
            "staking_reward": {"min": 0.8, "max": 1.0},
            "trading_reward": {"min": 0.6, "max": 0.8},
            "overall_reward": {"min": 0.7, "max": 0.9}
        }
    }
    
    # Filter reward data
    result = filter_data(
        user_id=test_user.user_id,
        data=test_reward_data,
        filter_config=reward_config,
        db_session=db_session
    )
    
    # Verify filter result
    assert isinstance(result, Dict)
    assert "filter_id" in result
    assert "filtered_data" in result
    assert "filter_details" in result
    
    # Verify filter metadata
    assert result["filter_type"] == "range"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["conditions"] == {
        "mining_reward": {"min": 0.7, "max": 0.9},
        "staking_reward": {"min": 0.8, "max": 1.0},
        "trading_reward": {"min": 0.6, "max": 0.8},
        "overall_reward": {"min": 0.7, "max": 0.9}
    }
    
    # Verify filter details
    assert isinstance(result["filter_details"], Dict)
    assert "timestamp" in result["filter_details"]
    assert "filter_results" in result["filter_details"]
    assert "filtered_rows" in result["filter_details"]
    assert "filtered_columns" in result["filter_details"]
    
    # Verify filter results
    assert isinstance(result["filter_details"]["filter_results"], Dict)
    assert "mining_reward" in result["filter_details"]["filter_results"]
    assert "staking_reward" in result["filter_details"]["filter_results"]
    assert "trading_reward" in result["filter_details"]["filter_results"]
    assert "overall_reward" in result["filter_details"]["filter_results"]
    
    # Verify filtered data
    assert isinstance(result["filtered_data"], pd.DataFrame)
    assert "mining_reward" in result["filtered_data"].columns
    assert "staking_reward" in result["filtered_data"].columns
    assert "trading_reward" in result["filtered_data"].columns
    assert "overall_reward" in result["filtered_data"].columns
    assert "timestamp" in result["filtered_data"].columns
    assert "day" in result["filtered_data"].columns
    
    # Verify data filtering
    assert len(result["filtered_data"]) <= len(test_reward_data)
    
    # Verify database entry
    db_record = db_session.query(FilterRecord).filter_by(
        user_id=test_user.user_id,
        filter_id=result["filter_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_filter_activity_data(db_session, test_user, test_activity_data):
    """Test filtering activity data"""
    # Create filter config for activity data
    activity_config = {
        "filter_type": "range",
        "columns": None,  # Filter all columns
        "exclude_columns": ["timestamp", "day"],
        "conditions": {
            "mining_activity": {"min": 0.8, "max": 0.9},
            "staking_activity": {"min": 0.85, "max": 0.95},
            "trading_activity": {"min": 0.7, "max": 0.8},
            "overall_activity": {"min": 0.8, "max": 0.9}
        }
    }
    
    # Filter activity data
    result = filter_data(
        user_id=test_user.user_id,
        data=test_activity_data,
        filter_config=activity_config,
        db_session=db_session
    )
    
    # Verify filter result
    assert isinstance(result, Dict)
    assert "filter_id" in result
    assert "filtered_data" in result
    assert "filter_details" in result
    
    # Verify filter metadata
    assert result["filter_type"] == "range"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["conditions"] == {
        "mining_activity": {"min": 0.8, "max": 0.9},
        "staking_activity": {"min": 0.85, "max": 0.95},
        "trading_activity": {"min": 0.7, "max": 0.8},
        "overall_activity": {"min": 0.8, "max": 0.9}
    }
    
    # Verify filter details
    assert isinstance(result["filter_details"], Dict)
    assert "timestamp" in result["filter_details"]
    assert "filter_results" in result["filter_details"]
    assert "filtered_rows" in result["filter_details"]
    assert "filtered_columns" in result["filter_details"]
    
    # Verify filter results
    assert isinstance(result["filter_details"]["filter_results"], Dict)
    assert "mining_activity" in result["filter_details"]["filter_results"]
    assert "staking_activity" in result["filter_details"]["filter_results"]
    assert "trading_activity" in result["filter_details"]["filter_results"]
    assert "overall_activity" in result["filter_details"]["filter_results"]
    
    # Verify filtered data
    assert isinstance(result["filtered_data"], pd.DataFrame)
    assert "mining_activity" in result["filtered_data"].columns
    assert "staking_activity" in result["filtered_data"].columns
    assert "trading_activity" in result["filtered_data"].columns
    assert "overall_activity" in result["filtered_data"].columns
    assert "timestamp" in result["filtered_data"].columns
    assert "day" in result["filtered_data"].columns
    
    # Verify data filtering
    assert len(result["filtered_data"]) <= len(test_activity_data)
    
    # Verify database entry
    db_record = db_session.query(FilterRecord).filter_by(
        user_id=test_user.user_id,
        filter_id=result["filter_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_filter_with_regex(db_session, test_user):
    """Test filtering with regex"""
    # Create data with text fields
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
    days = dates.date
    
    text_data = pd.DataFrame({
        "timestamp": dates,
        "day": days,
        "email": [f"user{i}@example.com" for i in range(100)],
        "phone": [f"+1-555-{i:04d}" for i in range(100)],
        "username": [f"user{i}" for i in range(100)],
        "password": [f"pass{i}" for i in range(100)]
    })
    
    # Create filter config with regex
    regex_config = {
        "filter_type": "regex",
        "columns": None,  # Filter all columns
        "exclude_columns": ["timestamp", "day"],
        "conditions": {
            "email": {"pattern": r"user[0-9]+@example\.com"},
            "phone": {"pattern": r"\+1-555-[0-9]{4}"},
            "username": {"pattern": r"user[0-9]+"},
            "password": {"pattern": r"pass[0-9]+"}
        }
    }
    
    # Filter text data
    result = filter_data(
        user_id=test_user.user_id,
        data=text_data,
        filter_config=regex_config,
        db_session=db_session
    )
    
    # Verify filter result
    assert isinstance(result, Dict)
    assert "filter_id" in result
    assert "filtered_data" in result
    assert "filter_details" in result
    
    # Verify filter metadata
    assert result["filter_type"] == "regex"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["conditions"] == {
        "email": {"pattern": r"user[0-9]+@example\.com"},
        "phone": {"pattern": r"\+1-555-[0-9]{4}"},
        "username": {"pattern": r"user[0-9]+"},
        "password": {"pattern": r"pass[0-9]+"}
    }
    
    # Verify filter details
    assert isinstance(result["filter_details"], Dict)
    assert "timestamp" in result["filter_details"]
    assert "filter_results" in result["filter_details"]
    assert "filtered_rows" in result["filter_details"]
    assert "filtered_columns" in result["filter_details"]
    
    # Verify filter results
    assert isinstance(result["filter_details"]["filter_results"], Dict)
    assert "email" in result["filter_details"]["filter_results"]
    assert "phone" in result["filter_details"]["filter_results"]
    assert "username" in result["filter_details"]["filter_results"]
    assert "password" in result["filter_details"]["filter_results"]
    
    # Verify filtered data
    assert isinstance(result["filtered_data"], pd.DataFrame)
    assert "email" in result["filtered_data"].columns
    assert "phone" in result["filtered_data"].columns
    assert "username" in result["filtered_data"].columns
    assert "password" in result["filtered_data"].columns
    assert "timestamp" in result["filtered_data"].columns
    assert "day" in result["filtered_data"].columns
    
    # Verify data filtering
    assert len(result["filtered_data"]) <= len(text_data)
    
    # Verify database entry
    db_record = db_session.query(FilterRecord).filter_by(
        user_id=test_user.user_id,
        filter_id=result["filter_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_filter_with_custom_function(db_session, test_user, test_performance_data):
    """Test filtering with custom function"""
    # Create filter config with custom function
    custom_config = {
        "filter_type": "custom",
        "columns": None,  # Filter all columns
        "exclude_columns": ["timestamp", "day"],
        "conditions": {
            "mining_performance": {"function": "lambda x: x >= 0.85 and x <= 0.9"},
            "staking_performance": {"function": "lambda x: x >= 0.9 and x <= 0.95"},
            "trading_performance": {"function": "lambda x: x >= 0.75 and x <= 0.8"},
            "overall_performance": {"function": "lambda x: x >= 0.85 and x <= 0.9"}
        }
    }
    
    # Filter performance data
    result = filter_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        filter_config=custom_config,
        db_session=db_session
    )
    
    # Verify filter result
    assert isinstance(result, Dict)
    assert "filter_id" in result
    assert "filtered_data" in result
    assert "filter_details" in result
    
    # Verify filter metadata
    assert result["filter_type"] == "custom"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["conditions"] == {
        "mining_performance": {"function": "lambda x: x >= 0.85 and x <= 0.9"},
        "staking_performance": {"function": "lambda x: x >= 0.9 and x <= 0.95"},
        "trading_performance": {"function": "lambda x: x >= 0.75 and x <= 0.8"},
        "overall_performance": {"function": "lambda x: x >= 0.85 and x <= 0.9"}
    }
    
    # Verify filter details
    assert isinstance(result["filter_details"], Dict)
    assert "timestamp" in result["filter_details"]
    assert "filter_results" in result["filter_details"]
    assert "filtered_rows" in result["filter_details"]
    assert "filtered_columns" in result["filter_details"]
    
    # Verify filter results
    assert isinstance(result["filter_details"]["filter_results"], Dict)
    assert "mining_performance" in result["filter_details"]["filter_results"]
    assert "staking_performance" in result["filter_details"]["filter_results"]
    assert "trading_performance" in result["filter_details"]["filter_results"]
    assert "overall_performance" in result["filter_details"]["filter_results"]
    
    # Verify filtered data
    assert isinstance(result["filtered_data"], pd.DataFrame)
    assert "mining_performance" in result["filtered_data"].columns
    assert "staking_performance" in result["filtered_data"].columns
    assert "trading_performance" in result["filtered_data"].columns
    assert "overall_performance" in result["filtered_data"].columns
    assert "timestamp" in result["filtered_data"].columns
    assert "day" in result["filtered_data"].columns
    
    # Verify data filtering
    assert len(result["filtered_data"]) <= len(test_performance_data)
    
    # Verify database entry
    db_record = db_session.query(FilterRecord).filter_by(
        user_id=test_user.user_id,
        filter_id=result["filter_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_filter_with_specific_columns(db_session, test_user, test_performance_data):
    """Test filtering with specific columns"""
    # Create filter config with specific columns
    specific_columns_config = {
        "filter_type": "range",
        "columns": ["mining_performance", "trading_performance"],  # Only filter these columns
        "exclude_columns": ["timestamp", "day"],
        "conditions": {
            "mining_performance": {"min": 0.85, "max": 0.9},
            "trading_performance": {"min": 0.75, "max": 0.8}
        }
    }
    
    # Filter performance data
    result = filter_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        filter_config=specific_columns_config,
        db_session=db_session
    )
    
    # Verify filter result
    assert isinstance(result, Dict)
    assert "filter_id" in result
    assert "filtered_data" in result
    assert "filter_details" in result
    
    # Verify filter metadata
    assert result["filter_type"] == "range"
    assert result["columns"] == ["mining_performance", "trading_performance"]
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["conditions"] == {
        "mining_performance": {"min": 0.85, "max": 0.9},
        "trading_performance": {"min": 0.75, "max": 0.8}
    }
    
    # Verify filter details
    assert isinstance(result["filter_details"], Dict)
    assert "timestamp" in result["filter_details"]
    assert "filter_results" in result["filter_details"]
    assert "filtered_rows" in result["filter_details"]
    assert "filtered_columns" in result["filter_details"]
    
    # Verify filter results
    assert isinstance(result["filter_details"]["filter_results"], Dict)
    assert "mining_performance" in result["filter_details"]["filter_results"]
    assert "trading_performance" in result["filter_details"]["filter_results"]
    assert "staking_performance" not in result["filter_details"]["filter_results"]
    assert "overall_performance" not in result["filter_details"]["filter_results"]
    
    # Verify filtered data
    assert isinstance(result["filtered_data"], pd.DataFrame)
    assert "mining_performance" in result["filtered_data"].columns
    assert "trading_performance" in result["filtered_data"].columns
    assert "staking_performance" in result["filtered_data"].columns
    assert "overall_performance" in result["filtered_data"].columns
    assert "timestamp" in result["filtered_data"].columns
    assert "day" in result["filtered_data"].columns
    
    # Verify data filtering
    assert len(result["filtered_data"]) <= len(test_performance_data)
    
    # Verify database entry
    db_record = db_session.query(FilterRecord).filter_by(
        user_id=test_user.user_id,
        filter_id=result["filter_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_get_filter_info(db_session, test_user, test_performance_data, test_filter_config):
    """Test filter info retrieval"""
    # First, filter performance data
    filter_result = filter_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        filter_config=test_filter_config,
        db_session=db_session
    )
    
    filter_id = filter_result["filter_id"]
    
    # Get filter info
    result = get_filter_info(
        user_id=test_user.user_id,
        filter_id=filter_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "filter_id" in result
    assert result["filter_id"] == filter_id
    
    # Verify filter metadata
    assert result["filter_type"] == "range"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["conditions"] == {
        "mining_performance": {"min": 0.85, "max": 0.9},
        "staking_performance": {"min": 0.9, "max": 0.95},
        "trading_performance": {"min": 0.75, "max": 0.8},
        "overall_performance": {"min": 0.85, "max": 0.9}
    }
    
    # Verify filter details
    assert "filter_details" in result
    assert isinstance(result["filter_details"], Dict)
    assert "timestamp" in result["filter_details"]
    assert "filter_results" in result["filter_details"]
    assert "filtered_rows" in result["filter_details"]
    assert "filtered_columns" in result["filter_details"]
    
    # Verify database entry
    db_record = db_session.query(FilterRecord).filter_by(
        user_id=test_user.user_id,
        filter_id=filter_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_filter_error_handling(db_session, test_user):
    """Test filter error handling"""
    # Invalid user ID
    with pytest.raises(FilterError) as excinfo:
        filter_data(
            user_id=None,
            data=pd.DataFrame(),
            filter_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(FilterError) as excinfo:
        filter_data(
            user_id=test_user.user_id,
            data=None,
            filter_config={},
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Invalid filter type
    with pytest.raises(FilterError) as excinfo:
        filter_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            filter_config={"filter_type": "invalid_type"},
            db_session=db_session
        )
    assert "Invalid filter type" in str(excinfo.value)
    
    # Invalid columns
    with pytest.raises(FilterError) as excinfo:
        filter_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            filter_config={"filter_type": "range", "columns": ["invalid_column"]},
            db_session=db_session
        )
    assert "Invalid columns" in str(excinfo.value)
    
    # Invalid filter ID
    with pytest.raises(FilterError) as excinfo:
        get_filter_info(
            user_id=test_user.user_id,
            filter_id="invalid_filter_id",
            db_session=db_session
        )
    assert "Invalid filter ID" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBFilterError) as excinfo:
        filter_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            filter_config={"filter_type": "range"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 