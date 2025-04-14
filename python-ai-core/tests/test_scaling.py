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

from core.scaling import (
    scale_data,
    get_scaling_info,
    ScalingError
)
from database.models import User, ScalingRecord
from database.exceptions import ScalingError as DBScalingError

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
    # Create data with multiple days to test scaling
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
    # Create data with multiple days to test scaling
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
    # Create data with multiple days to test scaling
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
    # Create data with multiple days to test scaling
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
def test_scaling_config():
    """Create test scaling configuration"""
    return {
        "scaling_type": "minmax",
        "feature_range": (0, 1),
        "columns": None,  # Scale all numeric columns
        "exclude_columns": ["timestamp", "day"]
    }

def test_scale_performance_data(db_session, test_user, test_performance_data, test_scaling_config):
    """Test scaling performance data"""
    # Scale performance data
    result = scale_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        scaling_config=test_scaling_config,
        db_session=db_session
    )
    
    # Verify scaling result
    assert isinstance(result, Dict)
    assert "scaling_id" in result
    assert "scaled_data" in result
    
    # Verify scaling metadata
    assert result["scaling_type"] == "minmax"
    assert result["feature_range"] == (0, 1)
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    
    # Verify scaling details
    assert "scaling_details" in result
    assert isinstance(result["scaling_details"], Dict)
    assert "timestamp" in result["scaling_details"]
    assert "original_shape" in result["scaling_details"]
    assert "scaled_shape" in result["scaling_details"]
    assert "scalers" in result["scaling_details"]
    
    # Verify scaled data
    assert isinstance(result["scaled_data"], pd.DataFrame)
    assert "day" in result["scaled_data"].columns
    assert "mining_performance" in result["scaled_data"].columns
    assert "staking_performance" in result["scaled_data"].columns
    assert "trading_performance" in result["scaled_data"].columns
    assert "overall_performance" in result["scaled_data"].columns
    
    # Verify data scaling
    assert len(result["scaled_data"]) == len(test_performance_data)
    assert result["scaled_data"]["mining_performance"].min() >= 0
    assert result["scaled_data"]["mining_performance"].max() <= 1
    assert result["scaled_data"]["staking_performance"].min() >= 0
    assert result["scaled_data"]["staking_performance"].max() <= 1
    assert result["scaled_data"]["trading_performance"].min() >= 0
    assert result["scaled_data"]["trading_performance"].max() <= 1
    assert result["scaled_data"]["overall_performance"].min() >= 0
    assert result["scaled_data"]["overall_performance"].max() <= 1
    
    # Verify database entry
    db_record = db_session.query(ScalingRecord).filter_by(
        user_id=test_user.user_id,
        scaling_id=result["scaling_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_scale_risk_data(db_session, test_user, test_risk_data, test_scaling_config):
    """Test scaling risk data"""
    # Scale risk data
    result = scale_data(
        user_id=test_user.user_id,
        data=test_risk_data,
        scaling_config=test_scaling_config,
        db_session=db_session
    )
    
    # Verify scaling result
    assert isinstance(result, Dict)
    assert "scaling_id" in result
    assert "scaled_data" in result
    
    # Verify scaling metadata
    assert result["scaling_type"] == "minmax"
    assert result["feature_range"] == (0, 1)
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    
    # Verify scaling details
    assert "scaling_details" in result
    assert isinstance(result["scaling_details"], Dict)
    assert "timestamp" in result["scaling_details"]
    assert "original_shape" in result["scaling_details"]
    assert "scaled_shape" in result["scaling_details"]
    assert "scalers" in result["scaling_details"]
    
    # Verify scaled data
    assert isinstance(result["scaled_data"], pd.DataFrame)
    assert "day" in result["scaled_data"].columns
    assert "mining_risk" in result["scaled_data"].columns
    assert "staking_risk" in result["scaled_data"].columns
    assert "trading_risk" in result["scaled_data"].columns
    assert "overall_risk" in result["scaled_data"].columns
    
    # Verify data scaling
    assert len(result["scaled_data"]) == len(test_risk_data)
    assert result["scaled_data"]["mining_risk"].min() >= 0
    assert result["scaled_data"]["mining_risk"].max() <= 1
    assert result["scaled_data"]["staking_risk"].min() >= 0
    assert result["scaled_data"]["staking_risk"].max() <= 1
    assert result["scaled_data"]["trading_risk"].min() >= 0
    assert result["scaled_data"]["trading_risk"].max() <= 1
    assert result["scaled_data"]["overall_risk"].min() >= 0
    assert result["scaled_data"]["overall_risk"].max() <= 1
    
    # Verify database entry
    db_record = db_session.query(ScalingRecord).filter_by(
        user_id=test_user.user_id,
        scaling_id=result["scaling_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_scale_reward_data(db_session, test_user, test_reward_data, test_scaling_config):
    """Test scaling reward data"""
    # Scale reward data
    result = scale_data(
        user_id=test_user.user_id,
        data=test_reward_data,
        scaling_config=test_scaling_config,
        db_session=db_session
    )
    
    # Verify scaling result
    assert isinstance(result, Dict)
    assert "scaling_id" in result
    assert "scaled_data" in result
    
    # Verify scaling metadata
    assert result["scaling_type"] == "minmax"
    assert result["feature_range"] == (0, 1)
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    
    # Verify scaling details
    assert "scaling_details" in result
    assert isinstance(result["scaling_details"], Dict)
    assert "timestamp" in result["scaling_details"]
    assert "original_shape" in result["scaling_details"]
    assert "scaled_shape" in result["scaling_details"]
    assert "scalers" in result["scaling_details"]
    
    # Verify scaled data
    assert isinstance(result["scaled_data"], pd.DataFrame)
    assert "day" in result["scaled_data"].columns
    assert "mining_rewards" in result["scaled_data"].columns
    assert "staking_rewards" in result["scaled_data"].columns
    assert "trading_rewards" in result["scaled_data"].columns
    assert "overall_rewards" in result["scaled_data"].columns
    
    # Verify data scaling
    assert len(result["scaled_data"]) == len(test_reward_data)
    assert result["scaled_data"]["mining_rewards"].min() >= 0
    assert result["scaled_data"]["mining_rewards"].max() <= 1
    assert result["scaled_data"]["staking_rewards"].min() >= 0
    assert result["scaled_data"]["staking_rewards"].max() <= 1
    assert result["scaled_data"]["trading_rewards"].min() >= 0
    assert result["scaled_data"]["trading_rewards"].max() <= 1
    assert result["scaled_data"]["overall_rewards"].min() >= 0
    assert result["scaled_data"]["overall_rewards"].max() <= 1
    
    # Verify database entry
    db_record = db_session.query(ScalingRecord).filter_by(
        user_id=test_user.user_id,
        scaling_id=result["scaling_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_scale_activity_data(db_session, test_user, test_activity_data, test_scaling_config):
    """Test scaling activity data"""
    # Scale activity data
    result = scale_data(
        user_id=test_user.user_id,
        data=test_activity_data,
        scaling_config=test_scaling_config,
        db_session=db_session
    )
    
    # Verify scaling result
    assert isinstance(result, Dict)
    assert "scaling_id" in result
    assert "scaled_data" in result
    
    # Verify scaling metadata
    assert result["scaling_type"] == "minmax"
    assert result["feature_range"] == (0, 1)
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    
    # Verify scaling details
    assert "scaling_details" in result
    assert isinstance(result["scaling_details"], Dict)
    assert "timestamp" in result["scaling_details"]
    assert "original_shape" in result["scaling_details"]
    assert "scaled_shape" in result["scaling_details"]
    assert "scalers" in result["scaling_details"]
    
    # Verify scaled data
    assert isinstance(result["scaled_data"], pd.DataFrame)
    assert "day" in result["scaled_data"].columns
    assert "mining_activity" in result["scaled_data"].columns
    assert "staking_activity" in result["scaled_data"].columns
    assert "trading_activity" in result["scaled_data"].columns
    assert "overall_activity" in result["scaled_data"].columns
    
    # Verify data scaling
    assert len(result["scaled_data"]) == len(test_activity_data)
    assert result["scaled_data"]["mining_activity"].min() >= 0
    assert result["scaled_data"]["mining_activity"].max() <= 1
    assert result["scaled_data"]["staking_activity"].min() >= 0
    assert result["scaled_data"]["staking_activity"].max() <= 1
    assert result["scaled_data"]["trading_activity"].min() >= 0
    assert result["scaled_data"]["trading_activity"].max() <= 1
    assert result["scaled_data"]["overall_activity"].min() >= 0
    assert result["scaled_data"]["overall_activity"].max() <= 1
    
    # Verify database entry
    db_record = db_session.query(ScalingRecord).filter_by(
        user_id=test_user.user_id,
        scaling_id=result["scaling_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_scale_with_standard_scaler(db_session, test_user, test_performance_data):
    """Test scaling with standard scaler"""
    # Create scaling config with standard scaler
    standard_config = {
        "scaling_type": "standard",
        "feature_range": None,  # Not used for standard scaler
        "columns": None,  # Scale all numeric columns
        "exclude_columns": ["timestamp", "day"]
    }
    
    # Scale performance data
    result = scale_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        scaling_config=standard_config,
        db_session=db_session
    )
    
    # Verify scaling result
    assert isinstance(result, Dict)
    assert "scaling_id" in result
    assert "scaled_data" in result
    
    # Verify scaling metadata
    assert result["scaling_type"] == "standard"
    assert result["feature_range"] is None
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    
    # Verify scaling details
    assert "scaling_details" in result
    assert isinstance(result["scaling_details"], Dict)
    assert "timestamp" in result["scaling_details"]
    assert "original_shape" in result["scaling_details"]
    assert "scaled_shape" in result["scaling_details"]
    assert "scalers" in result["scaling_details"]
    
    # Verify data scaling
    assert len(result["scaled_data"]) == len(test_performance_data)
    # Standard scaler should have mean close to 0 and std close to 1
    assert abs(result["scaled_data"]["mining_performance"].mean()) < 0.1
    assert abs(result["scaled_data"]["mining_performance"].std() - 1) < 0.1
    assert abs(result["scaled_data"]["staking_performance"].mean()) < 0.1
    assert abs(result["scaled_data"]["staking_performance"].std() - 1) < 0.1
    assert abs(result["scaled_data"]["trading_performance"].mean()) < 0.1
    assert abs(result["scaled_data"]["trading_performance"].std() - 1) < 0.1
    assert abs(result["scaled_data"]["overall_performance"].mean()) < 0.1
    assert abs(result["scaled_data"]["overall_performance"].std() - 1) < 0.1
    
    # Verify database entry
    db_record = db_session.query(ScalingRecord).filter_by(
        user_id=test_user.user_id,
        scaling_id=result["scaling_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_scale_with_robust_scaler(db_session, test_user, test_performance_data):
    """Test scaling with robust scaler"""
    # Create scaling config with robust scaler
    robust_config = {
        "scaling_type": "robust",
        "feature_range": None,  # Not used for robust scaler
        "columns": None,  # Scale all numeric columns
        "exclude_columns": ["timestamp", "day"]
    }
    
    # Scale performance data
    result = scale_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        scaling_config=robust_config,
        db_session=db_session
    )
    
    # Verify scaling result
    assert isinstance(result, Dict)
    assert "scaling_id" in result
    assert "scaled_data" in result
    
    # Verify scaling metadata
    assert result["scaling_type"] == "robust"
    assert result["feature_range"] is None
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    
    # Verify scaling details
    assert "scaling_details" in result
    assert isinstance(result["scaling_details"], Dict)
    assert "timestamp" in result["scaling_details"]
    assert "original_shape" in result["scaling_details"]
    assert "scaled_shape" in result["scaling_details"]
    assert "scalers" in result["scaling_details"]
    
    # Verify data scaling
    assert len(result["scaled_data"]) == len(test_performance_data)
    # Robust scaler should have median close to 0
    assert abs(result["scaled_data"]["mining_performance"].median()) < 0.1
    assert abs(result["scaled_data"]["staking_performance"].median()) < 0.1
    assert abs(result["scaled_data"]["trading_performance"].median()) < 0.1
    assert abs(result["scaled_data"]["overall_performance"].median()) < 0.1
    
    # Verify database entry
    db_record = db_session.query(ScalingRecord).filter_by(
        user_id=test_user.user_id,
        scaling_id=result["scaling_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_scale_with_custom_range(db_session, test_user, test_performance_data):
    """Test scaling with custom range"""
    # Create scaling config with custom range
    custom_range_config = {
        "scaling_type": "minmax",
        "feature_range": (-1, 1),  # Custom range
        "columns": None,  # Scale all numeric columns
        "exclude_columns": ["timestamp", "day"]
    }
    
    # Scale performance data
    result = scale_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        scaling_config=custom_range_config,
        db_session=db_session
    )
    
    # Verify scaling result
    assert isinstance(result, Dict)
    assert "scaling_id" in result
    assert "scaled_data" in result
    
    # Verify scaling metadata
    assert result["scaling_type"] == "minmax"
    assert result["feature_range"] == (-1, 1)
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    
    # Verify scaling details
    assert "scaling_details" in result
    assert isinstance(result["scaling_details"], Dict)
    assert "timestamp" in result["scaling_details"]
    assert "original_shape" in result["scaling_details"]
    assert "scaled_shape" in result["scaling_details"]
    assert "scalers" in result["scaling_details"]
    
    # Verify data scaling
    assert len(result["scaled_data"]) == len(test_performance_data)
    assert result["scaled_data"]["mining_performance"].min() >= -1
    assert result["scaled_data"]["mining_performance"].max() <= 1
    assert result["scaled_data"]["staking_performance"].min() >= -1
    assert result["scaled_data"]["staking_performance"].max() <= 1
    assert result["scaled_data"]["trading_performance"].min() >= -1
    assert result["scaled_data"]["trading_performance"].max() <= 1
    assert result["scaled_data"]["overall_performance"].min() >= -1
    assert result["scaled_data"]["overall_performance"].max() <= 1
    
    # Verify database entry
    db_record = db_session.query(ScalingRecord).filter_by(
        user_id=test_user.user_id,
        scaling_id=result["scaling_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_scale_with_specific_columns(db_session, test_user, test_performance_data):
    """Test scaling with specific columns"""
    # Create scaling config with specific columns
    specific_columns_config = {
        "scaling_type": "minmax",
        "feature_range": (0, 1),
        "columns": ["mining_performance", "trading_performance"],  # Only scale these columns
        "exclude_columns": ["timestamp", "day"]
    }
    
    # Scale performance data
    result = scale_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        scaling_config=specific_columns_config,
        db_session=db_session
    )
    
    # Verify scaling result
    assert isinstance(result, Dict)
    assert "scaling_id" in result
    assert "scaled_data" in result
    
    # Verify scaling metadata
    assert result["scaling_type"] == "minmax"
    assert result["feature_range"] == (0, 1)
    assert result["columns"] == ["mining_performance", "trading_performance"]
    assert result["exclude_columns"] == ["timestamp", "day"]
    
    # Verify scaling details
    assert "scaling_details" in result
    assert isinstance(result["scaling_details"], Dict)
    assert "timestamp" in result["scaling_details"]
    assert "original_shape" in result["scaling_details"]
    assert "scaled_shape" in result["scaling_details"]
    assert "scalers" in result["scaling_details"]
    
    # Verify data scaling
    assert len(result["scaled_data"]) == len(test_performance_data)
    # Check scaled columns
    assert result["scaled_data"]["mining_performance"].min() >= 0
    assert result["scaled_data"]["mining_performance"].max() <= 1
    assert result["scaled_data"]["trading_performance"].min() >= 0
    assert result["scaled_data"]["trading_performance"].max() <= 1
    # Check non-scaled columns (should be unchanged)
    assert result["scaled_data"]["staking_performance"].equals(test_performance_data["staking_performance"])
    assert result["scaled_data"]["overall_performance"].equals(test_performance_data["overall_performance"])
    
    # Verify database entry
    db_record = db_session.query(ScalingRecord).filter_by(
        user_id=test_user.user_id,
        scaling_id=result["scaling_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_scale_with_log_transformation(db_session, test_user, test_performance_data):
    """Test scaling with log transformation"""
    # Create scaling config with log transformation
    log_config = {
        "scaling_type": "log",
        "feature_range": None,  # Not used for log transformation
        "columns": None,  # Scale all numeric columns
        "exclude_columns": ["timestamp", "day"]
    }
    
    # Scale performance data
    result = scale_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        scaling_config=log_config,
        db_session=db_session
    )
    
    # Verify scaling result
    assert isinstance(result, Dict)
    assert "scaling_id" in result
    assert "scaled_data" in result
    
    # Verify scaling metadata
    assert result["scaling_type"] == "log"
    assert result["feature_range"] is None
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    
    # Verify scaling details
    assert "scaling_details" in result
    assert isinstance(result["scaling_details"], Dict)
    assert "timestamp" in result["scaling_details"]
    assert "original_shape" in result["scaling_details"]
    assert "scaled_shape" in result["scaling_details"]
    assert "scalers" in result["scaling_details"]
    
    # Verify data scaling
    assert len(result["scaled_data"]) == len(test_performance_data)
    # Log transformation should reduce the range of values
    assert result["scaled_data"]["mining_performance"].max() < test_performance_data["mining_performance"].max()
    assert result["scaled_data"]["staking_performance"].max() < test_performance_data["staking_performance"].max()
    assert result["scaled_data"]["trading_performance"].max() < test_performance_data["trading_performance"].max()
    assert result["scaled_data"]["overall_performance"].max() < test_performance_data["overall_performance"].max()
    
    # Verify database entry
    db_record = db_session.query(ScalingRecord).filter_by(
        user_id=test_user.user_id,
        scaling_id=result["scaling_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_scale_with_power_transformation(db_session, test_user, test_performance_data):
    """Test scaling with power transformation"""
    # Create scaling config with power transformation
    power_config = {
        "scaling_type": "power",
        "feature_range": None,  # Not used for power transformation
        "columns": None,  # Scale all numeric columns
        "exclude_columns": ["timestamp", "day"],
        "power": 0.5  # Square root transformation
    }
    
    # Scale performance data
    result = scale_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        scaling_config=power_config,
        db_session=db_session
    )
    
    # Verify scaling result
    assert isinstance(result, Dict)
    assert "scaling_id" in result
    assert "scaled_data" in result
    
    # Verify scaling metadata
    assert result["scaling_type"] == "power"
    assert result["feature_range"] is None
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["power"] == 0.5
    
    # Verify scaling details
    assert "scaling_details" in result
    assert isinstance(result["scaling_details"], Dict)
    assert "timestamp" in result["scaling_details"]
    assert "original_shape" in result["scaling_details"]
    assert "scaled_shape" in result["scaling_details"]
    assert "scalers" in result["scaling_details"]
    
    # Verify data scaling
    assert len(result["scaled_data"]) == len(test_performance_data)
    # Power transformation with power < 1 should reduce the range of values
    assert result["scaled_data"]["mining_performance"].max() < test_performance_data["mining_performance"].max()
    assert result["scaled_data"]["staking_performance"].max() < test_performance_data["staking_performance"].max()
    assert result["scaled_data"]["trading_performance"].max() < test_performance_data["trading_performance"].max()
    assert result["scaled_data"]["overall_performance"].max() < test_performance_data["overall_performance"].max()
    
    # Verify database entry
    db_record = db_session.query(ScalingRecord).filter_by(
        user_id=test_user.user_id,
        scaling_id=result["scaling_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_get_scaling_info(db_session, test_user, test_performance_data, test_scaling_config):
    """Test scaling info retrieval"""
    # First, scale performance data
    scaling_result = scale_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        scaling_config=test_scaling_config,
        db_session=db_session
    )
    
    scaling_id = scaling_result["scaling_id"]
    
    # Get scaling info
    result = get_scaling_info(
        user_id=test_user.user_id,
        scaling_id=scaling_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "scaling_id" in result
    assert result["scaling_id"] == scaling_id
    
    # Verify scaling metadata
    assert result["scaling_type"] == "minmax"
    assert result["feature_range"] == (0, 1)
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    
    # Verify scaling details
    assert "scaling_details" in result
    assert isinstance(result["scaling_details"], Dict)
    assert "timestamp" in result["scaling_details"]
    assert "original_shape" in result["scaling_details"]
    assert "scaled_shape" in result["scaling_details"]
    assert "scalers" in result["scaling_details"]
    
    # Verify database entry
    db_record = db_session.query(ScalingRecord).filter_by(
        user_id=test_user.user_id,
        scaling_id=scaling_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_scaling_error_handling(db_session, test_user):
    """Test scaling error handling"""
    # Invalid user ID
    with pytest.raises(ScalingError) as excinfo:
        scale_data(
            user_id=None,
            data=pd.DataFrame(),
            scaling_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(ScalingError) as excinfo:
        scale_data(
            user_id=test_user.user_id,
            data=None,
            scaling_config={},
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Invalid scaling type
    with pytest.raises(ScalingError) as excinfo:
        scale_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            scaling_config={"scaling_type": "invalid_type"},
            db_session=db_session
        )
    assert "Invalid scaling type" in str(excinfo.value)
    
    # Invalid feature range
    with pytest.raises(ScalingError) as excinfo:
        scale_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            scaling_config={"scaling_type": "minmax", "feature_range": (1, 0)},
            db_session=db_session
        )
    assert "Invalid feature range" in str(excinfo.value)
    
    # Invalid columns
    with pytest.raises(ScalingError) as excinfo:
        scale_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            scaling_config={"scaling_type": "minmax", "columns": ["invalid_column"]},
            db_session=db_session
        )
    assert "Invalid columns" in str(excinfo.value)
    
    # Invalid scaling ID
    with pytest.raises(ScalingError) as excinfo:
        get_scaling_info(
            user_id=test_user.user_id,
            scaling_id="invalid_scaling_id",
            db_session=db_session
        )
    assert "Invalid scaling ID" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBScalingError) as excinfo:
        scale_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            scaling_config={"scaling_type": "minmax"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 