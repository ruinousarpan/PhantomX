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

from core.transformation import (
    transform_data,
    convert_data,
    get_transformation_info,
    TransformationError
)
from database.models import User, TransformationRecord
from database.exceptions import TransformationError as DBTransformationError

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
def test_transformation_config():
    """Create test transformation configuration"""
    return {
        "transformation_type": "normalize",
        "columns": None,  # Transform all columns
        "exclude_columns": ["timestamp", "day"],
        "method": "minmax",
        "params": {
            "feature_range": (0, 1)
        }
    }

def test_transform_performance_data(db_session, test_user, test_performance_data, test_transformation_config):
    """Test transforming performance data"""
    # Transform performance data
    result = transform_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        transformation_config=test_transformation_config,
        db_session=db_session
    )
    
    # Verify transformation result
    assert isinstance(result, Dict)
    assert "transformation_id" in result
    assert "transformed_data" in result
    assert "transformation_details" in result
    
    # Verify transformation metadata
    assert result["transformation_type"] == "normalize"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["method"] == "minmax"
    assert result["params"] == {
        "feature_range": (0, 1)
    }
    
    # Verify transformation details
    assert isinstance(result["transformation_details"], Dict)
    assert "timestamp" in result["transformation_details"]
    assert "transformation_results" in result["transformation_details"]
    assert "transformed_columns" in result["transformation_details"]
    
    # Verify transformation results
    assert isinstance(result["transformation_details"]["transformation_results"], Dict)
    assert "mining_performance" in result["transformation_details"]["transformation_results"]
    assert "staking_performance" in result["transformation_details"]["transformation_results"]
    assert "trading_performance" in result["transformation_details"]["transformation_results"]
    assert "overall_performance" in result["transformation_details"]["transformation_results"]
    
    # Verify transformed data
    assert isinstance(result["transformed_data"], pd.DataFrame)
    assert "mining_performance" in result["transformed_data"].columns
    assert "staking_performance" in result["transformed_data"].columns
    assert "trading_performance" in result["transformed_data"].columns
    assert "overall_performance" in result["transformed_data"].columns
    assert "timestamp" in result["transformed_data"].columns
    assert "day" in result["transformed_data"].columns
    
    # Verify data transformation
    assert result["transformed_data"]["mining_performance"].min() >= 0
    assert result["transformed_data"]["mining_performance"].max() <= 1
    assert result["transformed_data"]["staking_performance"].min() >= 0
    assert result["transformed_data"]["staking_performance"].max() <= 1
    assert result["transformed_data"]["trading_performance"].min() >= 0
    assert result["transformed_data"]["trading_performance"].max() <= 1
    assert result["transformed_data"]["overall_performance"].min() >= 0
    assert result["transformed_data"]["overall_performance"].max() <= 1
    
    # Verify database entry
    db_record = db_session.query(TransformationRecord).filter_by(
        user_id=test_user.user_id,
        transformation_id=result["transformation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_transform_risk_data(db_session, test_user, test_risk_data):
    """Test transforming risk data"""
    # Create transformation config for risk data
    risk_config = {
        "transformation_type": "normalize",
        "columns": None,  # Transform all columns
        "exclude_columns": ["timestamp", "day"],
        "method": "minmax",
        "params": {
            "feature_range": (0, 1)
        }
    }
    
    # Transform risk data
    result = transform_data(
        user_id=test_user.user_id,
        data=test_risk_data,
        transformation_config=risk_config,
        db_session=db_session
    )
    
    # Verify transformation result
    assert isinstance(result, Dict)
    assert "transformation_id" in result
    assert "transformed_data" in result
    assert "transformation_details" in result
    
    # Verify transformation metadata
    assert result["transformation_type"] == "normalize"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["method"] == "minmax"
    assert result["params"] == {
        "feature_range": (0, 1)
    }
    
    # Verify transformation details
    assert isinstance(result["transformation_details"], Dict)
    assert "timestamp" in result["transformation_details"]
    assert "transformation_results" in result["transformation_details"]
    assert "transformed_columns" in result["transformation_details"]
    
    # Verify transformation results
    assert isinstance(result["transformation_details"]["transformation_results"], Dict)
    assert "mining_risk" in result["transformation_details"]["transformation_results"]
    assert "staking_risk" in result["transformation_details"]["transformation_results"]
    assert "trading_risk" in result["transformation_details"]["transformation_results"]
    assert "overall_risk" in result["transformation_details"]["transformation_results"]
    
    # Verify transformed data
    assert isinstance(result["transformed_data"], pd.DataFrame)
    assert "mining_risk" in result["transformed_data"].columns
    assert "staking_risk" in result["transformed_data"].columns
    assert "trading_risk" in result["transformed_data"].columns
    assert "overall_risk" in result["transformed_data"].columns
    assert "timestamp" in result["transformed_data"].columns
    assert "day" in result["transformed_data"].columns
    
    # Verify data transformation
    assert result["transformed_data"]["mining_risk"].min() >= 0
    assert result["transformed_data"]["mining_risk"].max() <= 1
    assert result["transformed_data"]["staking_risk"].min() >= 0
    assert result["transformed_data"]["staking_risk"].max() <= 1
    assert result["transformed_data"]["trading_risk"].min() >= 0
    assert result["transformed_data"]["trading_risk"].max() <= 1
    assert result["transformed_data"]["overall_risk"].min() >= 0
    assert result["transformed_data"]["overall_risk"].max() <= 1
    
    # Verify database entry
    db_record = db_session.query(TransformationRecord).filter_by(
        user_id=test_user.user_id,
        transformation_id=result["transformation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_transform_reward_data(db_session, test_user, test_reward_data):
    """Test transforming reward data"""
    # Create transformation config for reward data
    reward_config = {
        "transformation_type": "normalize",
        "columns": None,  # Transform all columns
        "exclude_columns": ["timestamp", "day"],
        "method": "minmax",
        "params": {
            "feature_range": (0, 1)
        }
    }
    
    # Transform reward data
    result = transform_data(
        user_id=test_user.user_id,
        data=test_reward_data,
        transformation_config=reward_config,
        db_session=db_session
    )
    
    # Verify transformation result
    assert isinstance(result, Dict)
    assert "transformation_id" in result
    assert "transformed_data" in result
    assert "transformation_details" in result
    
    # Verify transformation metadata
    assert result["transformation_type"] == "normalize"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["method"] == "minmax"
    assert result["params"] == {
        "feature_range": (0, 1)
    }
    
    # Verify transformation details
    assert isinstance(result["transformation_details"], Dict)
    assert "timestamp" in result["transformation_details"]
    assert "transformation_results" in result["transformation_details"]
    assert "transformed_columns" in result["transformation_details"]
    
    # Verify transformation results
    assert isinstance(result["transformation_details"]["transformation_results"], Dict)
    assert "mining_reward" in result["transformation_details"]["transformation_results"]
    assert "staking_reward" in result["transformation_details"]["transformation_results"]
    assert "trading_reward" in result["transformation_details"]["transformation_results"]
    assert "overall_reward" in result["transformation_details"]["transformation_results"]
    
    # Verify transformed data
    assert isinstance(result["transformed_data"], pd.DataFrame)
    assert "mining_reward" in result["transformed_data"].columns
    assert "staking_reward" in result["transformed_data"].columns
    assert "trading_reward" in result["transformed_data"].columns
    assert "overall_reward" in result["transformed_data"].columns
    assert "timestamp" in result["transformed_data"].columns
    assert "day" in result["transformed_data"].columns
    
    # Verify data transformation
    assert result["transformed_data"]["mining_reward"].min() >= 0
    assert result["transformed_data"]["mining_reward"].max() <= 1
    assert result["transformed_data"]["staking_reward"].min() >= 0
    assert result["transformed_data"]["staking_reward"].max() <= 1
    assert result["transformed_data"]["trading_reward"].min() >= 0
    assert result["transformed_data"]["trading_reward"].max() <= 1
    assert result["transformed_data"]["overall_reward"].min() >= 0
    assert result["transformed_data"]["overall_reward"].max() <= 1
    
    # Verify database entry
    db_record = db_session.query(TransformationRecord).filter_by(
        user_id=test_user.user_id,
        transformation_id=result["transformation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_transform_activity_data(db_session, test_user, test_activity_data):
    """Test transforming activity data"""
    # Create transformation config for activity data
    activity_config = {
        "transformation_type": "normalize",
        "columns": None,  # Transform all columns
        "exclude_columns": ["timestamp", "day"],
        "method": "minmax",
        "params": {
            "feature_range": (0, 1)
        }
    }
    
    # Transform activity data
    result = transform_data(
        user_id=test_user.user_id,
        data=test_activity_data,
        transformation_config=activity_config,
        db_session=db_session
    )
    
    # Verify transformation result
    assert isinstance(result, Dict)
    assert "transformation_id" in result
    assert "transformed_data" in result
    assert "transformation_details" in result
    
    # Verify transformation metadata
    assert result["transformation_type"] == "normalize"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["method"] == "minmax"
    assert result["params"] == {
        "feature_range": (0, 1)
    }
    
    # Verify transformation details
    assert isinstance(result["transformation_details"], Dict)
    assert "timestamp" in result["transformation_details"]
    assert "transformation_results" in result["transformation_details"]
    assert "transformed_columns" in result["transformation_details"]
    
    # Verify transformation results
    assert isinstance(result["transformation_details"]["transformation_results"], Dict)
    assert "mining_activity" in result["transformation_details"]["transformation_results"]
    assert "staking_activity" in result["transformation_details"]["transformation_results"]
    assert "trading_activity" in result["transformation_details"]["transformation_results"]
    assert "overall_activity" in result["transformation_details"]["transformation_results"]
    
    # Verify transformed data
    assert isinstance(result["transformed_data"], pd.DataFrame)
    assert "mining_activity" in result["transformed_data"].columns
    assert "staking_activity" in result["transformed_data"].columns
    assert "trading_activity" in result["transformed_data"].columns
    assert "overall_activity" in result["transformed_data"].columns
    assert "timestamp" in result["transformed_data"].columns
    assert "day" in result["transformed_data"].columns
    
    # Verify data transformation
    assert result["transformed_data"]["mining_activity"].min() >= 0
    assert result["transformed_data"]["mining_activity"].max() <= 1
    assert result["transformed_data"]["staking_activity"].min() >= 0
    assert result["transformed_data"]["staking_activity"].max() <= 1
    assert result["transformed_data"]["trading_activity"].min() >= 0
    assert result["transformed_data"]["trading_activity"].max() <= 1
    assert result["transformed_data"]["overall_activity"].min() >= 0
    assert result["transformed_data"]["overall_activity"].max() <= 1
    
    # Verify database entry
    db_record = db_session.query(TransformationRecord).filter_by(
        user_id=test_user.user_id,
        transformation_id=result["transformation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_transform_with_standard_scaler(db_session, test_user, test_performance_data):
    """Test transforming with standard scaler"""
    # Create transformation config with standard scaler
    standard_config = {
        "transformation_type": "normalize",
        "columns": None,  # Transform all columns
        "exclude_columns": ["timestamp", "day"],
        "method": "standard",
        "params": {}
    }
    
    # Transform performance data
    result = transform_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        transformation_config=standard_config,
        db_session=db_session
    )
    
    # Verify transformation result
    assert isinstance(result, Dict)
    assert "transformation_id" in result
    assert "transformed_data" in result
    assert "transformation_details" in result
    
    # Verify transformation metadata
    assert result["transformation_type"] == "normalize"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["method"] == "standard"
    assert result["params"] == {}
    
    # Verify transformation details
    assert isinstance(result["transformation_details"], Dict)
    assert "timestamp" in result["transformation_details"]
    assert "transformation_results" in result["transformation_details"]
    assert "transformed_columns" in result["transformation_details"]
    
    # Verify transformation results
    assert isinstance(result["transformation_details"]["transformation_results"], Dict)
    assert "mining_performance" in result["transformation_details"]["transformation_results"]
    assert "staking_performance" in result["transformation_details"]["transformation_results"]
    assert "trading_performance" in result["transformation_details"]["transformation_results"]
    assert "overall_performance" in result["transformation_details"]["transformation_results"]
    
    # Verify transformed data
    assert isinstance(result["transformed_data"], pd.DataFrame)
    assert "mining_performance" in result["transformed_data"].columns
    assert "staking_performance" in result["transformed_data"].columns
    assert "trading_performance" in result["transformed_data"].columns
    assert "overall_performance" in result["transformed_data"].columns
    assert "timestamp" in result["transformed_data"].columns
    assert "day" in result["transformed_data"].columns
    
    # Verify data transformation
    assert abs(result["transformed_data"]["mining_performance"].mean()) < 0.1
    assert abs(result["transformed_data"]["mining_performance"].std() - 1.0) < 0.1
    assert abs(result["transformed_data"]["staking_performance"].mean()) < 0.1
    assert abs(result["transformed_data"]["staking_performance"].std() - 1.0) < 0.1
    assert abs(result["transformed_data"]["trading_performance"].mean()) < 0.1
    assert abs(result["transformed_data"]["trading_performance"].std() - 1.0) < 0.1
    assert abs(result["transformed_data"]["overall_performance"].mean()) < 0.1
    assert abs(result["transformed_data"]["overall_performance"].std() - 1.0) < 0.1
    
    # Verify database entry
    db_record = db_session.query(TransformationRecord).filter_by(
        user_id=test_user.user_id,
        transformation_id=result["transformation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_transform_with_robust_scaler(db_session, test_user, test_performance_data):
    """Test transforming with robust scaler"""
    # Create transformation config with robust scaler
    robust_config = {
        "transformation_type": "normalize",
        "columns": None,  # Transform all columns
        "exclude_columns": ["timestamp", "day"],
        "method": "robust",
        "params": {}
    }
    
    # Transform performance data
    result = transform_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        transformation_config=robust_config,
        db_session=db_session
    )
    
    # Verify transformation result
    assert isinstance(result, Dict)
    assert "transformation_id" in result
    assert "transformed_data" in result
    assert "transformation_details" in result
    
    # Verify transformation metadata
    assert result["transformation_type"] == "normalize"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["method"] == "robust"
    assert result["params"] == {}
    
    # Verify transformation details
    assert isinstance(result["transformation_details"], Dict)
    assert "timestamp" in result["transformation_details"]
    assert "transformation_results" in result["transformation_details"]
    assert "transformed_columns" in result["transformation_details"]
    
    # Verify transformation results
    assert isinstance(result["transformation_details"]["transformation_results"], Dict)
    assert "mining_performance" in result["transformation_details"]["transformation_results"]
    assert "staking_performance" in result["transformation_details"]["transformation_results"]
    assert "trading_performance" in result["transformation_details"]["transformation_results"]
    assert "overall_performance" in result["transformation_details"]["transformation_results"]
    
    # Verify transformed data
    assert isinstance(result["transformed_data"], pd.DataFrame)
    assert "mining_performance" in result["transformed_data"].columns
    assert "staking_performance" in result["transformed_data"].columns
    assert "trading_performance" in result["transformed_data"].columns
    assert "overall_performance" in result["transformed_data"].columns
    assert "timestamp" in result["transformed_data"].columns
    assert "day" in result["transformed_data"].columns
    
    # Verify data transformation
    assert abs(result["transformed_data"]["mining_performance"].median()) < 0.1
    assert abs(result["transformed_data"]["staking_performance"].median()) < 0.1
    assert abs(result["transformed_data"]["trading_performance"].median()) < 0.1
    assert abs(result["transformed_data"]["overall_performance"].median()) < 0.1
    
    # Verify database entry
    db_record = db_session.query(TransformationRecord).filter_by(
        user_id=test_user.user_id,
        transformation_id=result["transformation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_transform_with_log_transformation(db_session, test_user, test_performance_data):
    """Test transforming with log transformation"""
    # Create transformation config with log transformation
    log_config = {
        "transformation_type": "transform",
        "columns": None,  # Transform all columns
        "exclude_columns": ["timestamp", "day"],
        "method": "log",
        "params": {}
    }
    
    # Transform performance data
    result = transform_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        transformation_config=log_config,
        db_session=db_session
    )
    
    # Verify transformation result
    assert isinstance(result, Dict)
    assert "transformation_id" in result
    assert "transformed_data" in result
    assert "transformation_details" in result
    
    # Verify transformation metadata
    assert result["transformation_type"] == "transform"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["method"] == "log"
    assert result["params"] == {}
    
    # Verify transformation details
    assert isinstance(result["transformation_details"], Dict)
    assert "timestamp" in result["transformation_details"]
    assert "transformation_results" in result["transformation_details"]
    assert "transformed_columns" in result["transformation_details"]
    
    # Verify transformation results
    assert isinstance(result["transformation_details"]["transformation_results"], Dict)
    assert "mining_performance" in result["transformation_details"]["transformation_results"]
    assert "staking_performance" in result["transformation_details"]["transformation_results"]
    assert "trading_performance" in result["transformation_details"]["transformation_results"]
    assert "overall_performance" in result["transformation_details"]["transformation_results"]
    
    # Verify transformed data
    assert isinstance(result["transformed_data"], pd.DataFrame)
    assert "mining_performance" in result["transformed_data"].columns
    assert "staking_performance" in result["transformed_data"].columns
    assert "trading_performance" in result["transformed_data"].columns
    assert "overall_performance" in result["transformed_data"].columns
    assert "timestamp" in result["transformed_data"].columns
    assert "day" in result["transformed_data"].columns
    
    # Verify data transformation
    assert result["transformed_data"]["mining_performance"].min() < 0
    assert result["transformed_data"]["staking_performance"].min() < 0
    assert result["transformed_data"]["trading_performance"].min() < 0
    assert result["transformed_data"]["overall_performance"].min() < 0
    
    # Verify database entry
    db_record = db_session.query(TransformationRecord).filter_by(
        user_id=test_user.user_id,
        transformation_id=result["transformation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_transform_with_power_transformation(db_session, test_user, test_performance_data):
    """Test transforming with power transformation"""
    # Create transformation config with power transformation
    power_config = {
        "transformation_type": "transform",
        "columns": None,  # Transform all columns
        "exclude_columns": ["timestamp", "day"],
        "method": "power",
        "params": {
            "power": 0.5  # Square root transformation
        }
    }
    
    # Transform performance data
    result = transform_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        transformation_config=power_config,
        db_session=db_session
    )
    
    # Verify transformation result
    assert isinstance(result, Dict)
    assert "transformation_id" in result
    assert "transformed_data" in result
    assert "transformation_details" in result
    
    # Verify transformation metadata
    assert result["transformation_type"] == "transform"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["method"] == "power"
    assert result["params"] == {
        "power": 0.5
    }
    
    # Verify transformation details
    assert isinstance(result["transformation_details"], Dict)
    assert "timestamp" in result["transformation_details"]
    assert "transformation_results" in result["transformation_details"]
    assert "transformed_columns" in result["transformation_details"]
    
    # Verify transformation results
    assert isinstance(result["transformation_details"]["transformation_results"], Dict)
    assert "mining_performance" in result["transformation_details"]["transformation_results"]
    assert "staking_performance" in result["transformation_details"]["transformation_results"]
    assert "trading_performance" in result["transformation_details"]["transformation_results"]
    assert "overall_performance" in result["transformation_details"]["transformation_results"]
    
    # Verify transformed data
    assert isinstance(result["transformed_data"], pd.DataFrame)
    assert "mining_performance" in result["transformed_data"].columns
    assert "staking_performance" in result["transformed_data"].columns
    assert "trading_performance" in result["transformed_data"].columns
    assert "overall_performance" in result["transformed_data"].columns
    assert "timestamp" in result["transformed_data"].columns
    assert "day" in result["transformed_data"].columns
    
    # Verify data transformation
    assert result["transformed_data"]["mining_performance"].min() >= 0
    assert result["transformed_data"]["mining_performance"].max() <= 1
    assert result["transformed_data"]["staking_performance"].min() >= 0
    assert result["transformed_data"]["staking_performance"].max() <= 1
    assert result["transformed_data"]["trading_performance"].min() >= 0
    assert result["transformed_data"]["trading_performance"].max() <= 1
    assert result["transformed_data"]["overall_performance"].min() >= 0
    assert result["transformed_data"]["overall_performance"].max() <= 1
    
    # Verify database entry
    db_record = db_session.query(TransformationRecord).filter_by(
        user_id=test_user.user_id,
        transformation_id=result["transformation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_transform_with_specific_columns(db_session, test_user, test_performance_data):
    """Test transforming with specific columns"""
    # Create transformation config with specific columns
    specific_columns_config = {
        "transformation_type": "normalize",
        "columns": ["mining_performance", "trading_performance"],  # Only transform these columns
        "exclude_columns": ["timestamp", "day"],
        "method": "minmax",
        "params": {
            "feature_range": (0, 1)
        }
    }
    
    # Transform performance data
    result = transform_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        transformation_config=specific_columns_config,
        db_session=db_session
    )
    
    # Verify transformation result
    assert isinstance(result, Dict)
    assert "transformation_id" in result
    assert "transformed_data" in result
    assert "transformation_details" in result
    
    # Verify transformation metadata
    assert result["transformation_type"] == "normalize"
    assert result["columns"] == ["mining_performance", "trading_performance"]
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["method"] == "minmax"
    assert result["params"] == {
        "feature_range": (0, 1)
    }
    
    # Verify transformation details
    assert isinstance(result["transformation_details"], Dict)
    assert "timestamp" in result["transformation_details"]
    assert "transformation_results" in result["transformation_details"]
    assert "transformed_columns" in result["transformation_details"]
    
    # Verify transformation results
    assert isinstance(result["transformation_details"]["transformation_results"], Dict)
    assert "mining_performance" in result["transformation_details"]["transformation_results"]
    assert "trading_performance" in result["transformation_details"]["transformation_results"]
    assert "staking_performance" not in result["transformation_details"]["transformation_results"]
    assert "overall_performance" not in result["transformation_details"]["transformation_results"]
    
    # Verify transformed data
    assert isinstance(result["transformed_data"], pd.DataFrame)
    assert "mining_performance" in result["transformed_data"].columns
    assert "trading_performance" in result["transformed_data"].columns
    assert "staking_performance" in result["transformed_data"].columns
    assert "overall_performance" in result["transformed_data"].columns
    assert "timestamp" in result["transformed_data"].columns
    assert "day" in result["transformed_data"].columns
    
    # Verify data transformation
    assert result["transformed_data"]["mining_performance"].min() >= 0
    assert result["transformed_data"]["mining_performance"].max() <= 1
    assert result["transformed_data"]["trading_performance"].min() >= 0
    assert result["transformed_data"]["trading_performance"].max() <= 1
    assert result["transformed_data"]["staking_performance"].min() == test_performance_data["staking_performance"].min()
    assert result["transformed_data"]["staking_performance"].max() == test_performance_data["staking_performance"].max()
    assert result["transformed_data"]["overall_performance"].min() == test_performance_data["overall_performance"].min()
    assert result["transformed_data"]["overall_performance"].max() == test_performance_data["overall_performance"].max()
    
    # Verify database entry
    db_record = db_session.query(TransformationRecord).filter_by(
        user_id=test_user.user_id,
        transformation_id=result["transformation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_get_transformation_info(db_session, test_user, test_performance_data, test_transformation_config):
    """Test transformation info retrieval"""
    # First, transform performance data
    transformation_result = transform_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        transformation_config=test_transformation_config,
        db_session=db_session
    )
    
    transformation_id = transformation_result["transformation_id"]
    
    # Get transformation info
    result = get_transformation_info(
        user_id=test_user.user_id,
        transformation_id=transformation_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "transformation_id" in result
    assert result["transformation_id"] == transformation_id
    
    # Verify transformation metadata
    assert result["transformation_type"] == "normalize"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["method"] == "minmax"
    assert result["params"] == {
        "feature_range": (0, 1)
    }
    
    # Verify transformation details
    assert "transformation_details" in result
    assert isinstance(result["transformation_details"], Dict)
    assert "timestamp" in result["transformation_details"]
    assert "transformation_results" in result["transformation_details"]
    assert "transformed_columns" in result["transformation_details"]
    
    # Verify database entry
    db_record = db_session.query(TransformationRecord).filter_by(
        user_id=test_user.user_id,
        transformation_id=transformation_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_transformation_error_handling(db_session, test_user):
    """Test transformation error handling"""
    # Invalid user ID
    with pytest.raises(TransformationError) as excinfo:
        transform_data(
            user_id=None,
            data=pd.DataFrame(),
            transformation_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(TransformationError) as excinfo:
        transform_data(
            user_id=test_user.user_id,
            data=None,
            transformation_config={},
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Invalid transformation type
    with pytest.raises(TransformationError) as excinfo:
        transform_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            transformation_config={"transformation_type": "invalid_type"},
            db_session=db_session
        )
    assert "Invalid transformation type" in str(excinfo.value)
    
    # Invalid method
    with pytest.raises(TransformationError) as excinfo:
        transform_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            transformation_config={"transformation_type": "normalize", "method": "invalid_method"},
            db_session=db_session
        )
    assert "Invalid transformation method" in str(excinfo.value)
    
    # Invalid columns
    with pytest.raises(TransformationError) as excinfo:
        transform_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            transformation_config={"transformation_type": "normalize", "method": "minmax", "columns": ["invalid_column"]},
            db_session=db_session
        )
    assert "Invalid columns" in str(excinfo.value)
    
    # Invalid transformation ID
    with pytest.raises(TransformationError) as excinfo:
        get_transformation_info(
            user_id=test_user.user_id,
            transformation_id="invalid_transformation_id",
            db_session=db_session
        )
    assert "Invalid transformation ID" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBTransformationError) as excinfo:
        transform_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            transformation_config={"transformation_type": "normalize", "method": "minmax"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 