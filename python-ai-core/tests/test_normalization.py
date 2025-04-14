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

from core.normalization import (
    normalize_data,
    get_normalization_info,
    NormalizationError
)
from database.models import User, NormalizationRecord
from database.exceptions import NormalizationError as DBNormalizationError

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
    # Create data with multiple days to test normalization
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
    # Create data with multiple days to test normalization
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
    # Create data with multiple days to test normalization
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
    # Create data with multiple days to test normalization
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
def test_normalization_config():
    """Create test normalization configuration"""
    return {
        "normalization_type": "minmax",
        "feature_range": (0, 1),
        "columns": None,  # Normalize all numeric columns
        "exclude_columns": ["timestamp", "day"]
    }

def test_normalize_performance_data(db_session, test_user, test_performance_data, test_normalization_config):
    """Test normalizing performance data"""
    # Normalize performance data
    result = normalize_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        normalization_config=test_normalization_config,
        db_session=db_session
    )
    
    # Verify normalization result
    assert isinstance(result, Dict)
    assert "normalization_id" in result
    assert "normalized_data" in result
    
    # Verify normalization metadata
    assert result["normalization_type"] == "minmax"
    assert result["feature_range"] == (0, 1)
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    
    # Verify normalization details
    assert "normalization_details" in result
    assert isinstance(result["normalization_details"], Dict)
    assert "timestamp" in result["normalization_details"]
    assert "original_shape" in result["normalization_details"]
    assert "normalized_shape" in result["normalization_details"]
    assert "scalers" in result["normalization_details"]
    
    # Verify normalized data
    assert isinstance(result["normalized_data"], pd.DataFrame)
    assert "day" in result["normalized_data"].columns
    assert "mining_performance" in result["normalized_data"].columns
    assert "staking_performance" in result["normalized_data"].columns
    assert "trading_performance" in result["normalized_data"].columns
    assert "overall_performance" in result["normalized_data"].columns
    
    # Verify data normalization
    assert len(result["normalized_data"]) == len(test_performance_data)
    assert result["normalized_data"]["mining_performance"].min() >= 0
    assert result["normalized_data"]["mining_performance"].max() <= 1
    assert result["normalized_data"]["staking_performance"].min() >= 0
    assert result["normalized_data"]["staking_performance"].max() <= 1
    assert result["normalized_data"]["trading_performance"].min() >= 0
    assert result["normalized_data"]["trading_performance"].max() <= 1
    assert result["normalized_data"]["overall_performance"].min() >= 0
    assert result["normalized_data"]["overall_performance"].max() <= 1
    
    # Verify database entry
    db_record = db_session.query(NormalizationRecord).filter_by(
        user_id=test_user.user_id,
        normalization_id=result["normalization_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_normalize_risk_data(db_session, test_user, test_risk_data, test_normalization_config):
    """Test normalizing risk data"""
    # Normalize risk data
    result = normalize_data(
        user_id=test_user.user_id,
        data=test_risk_data,
        normalization_config=test_normalization_config,
        db_session=db_session
    )
    
    # Verify normalization result
    assert isinstance(result, Dict)
    assert "normalization_id" in result
    assert "normalized_data" in result
    
    # Verify normalization metadata
    assert result["normalization_type"] == "minmax"
    assert result["feature_range"] == (0, 1)
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    
    # Verify normalization details
    assert "normalization_details" in result
    assert isinstance(result["normalization_details"], Dict)
    assert "timestamp" in result["normalization_details"]
    assert "original_shape" in result["normalization_details"]
    assert "normalized_shape" in result["normalization_details"]
    assert "scalers" in result["normalization_details"]
    
    # Verify normalized data
    assert isinstance(result["normalized_data"], pd.DataFrame)
    assert "day" in result["normalized_data"].columns
    assert "mining_risk" in result["normalized_data"].columns
    assert "staking_risk" in result["normalized_data"].columns
    assert "trading_risk" in result["normalized_data"].columns
    assert "overall_risk" in result["normalized_data"].columns
    
    # Verify data normalization
    assert len(result["normalized_data"]) == len(test_risk_data)
    assert result["normalized_data"]["mining_risk"].min() >= 0
    assert result["normalized_data"]["mining_risk"].max() <= 1
    assert result["normalized_data"]["staking_risk"].min() >= 0
    assert result["normalized_data"]["staking_risk"].max() <= 1
    assert result["normalized_data"]["trading_risk"].min() >= 0
    assert result["normalized_data"]["trading_risk"].max() <= 1
    assert result["normalized_data"]["overall_risk"].min() >= 0
    assert result["normalized_data"]["overall_risk"].max() <= 1
    
    # Verify database entry
    db_record = db_session.query(NormalizationRecord).filter_by(
        user_id=test_user.user_id,
        normalization_id=result["normalization_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_normalize_reward_data(db_session, test_user, test_reward_data, test_normalization_config):
    """Test normalizing reward data"""
    # Normalize reward data
    result = normalize_data(
        user_id=test_user.user_id,
        data=test_reward_data,
        normalization_config=test_normalization_config,
        db_session=db_session
    )
    
    # Verify normalization result
    assert isinstance(result, Dict)
    assert "normalization_id" in result
    assert "normalized_data" in result
    
    # Verify normalization metadata
    assert result["normalization_type"] == "minmax"
    assert result["feature_range"] == (0, 1)
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    
    # Verify normalization details
    assert "normalization_details" in result
    assert isinstance(result["normalization_details"], Dict)
    assert "timestamp" in result["normalization_details"]
    assert "original_shape" in result["normalization_details"]
    assert "normalized_shape" in result["normalization_details"]
    assert "scalers" in result["normalization_details"]
    
    # Verify normalized data
    assert isinstance(result["normalized_data"], pd.DataFrame)
    assert "day" in result["normalized_data"].columns
    assert "mining_rewards" in result["normalized_data"].columns
    assert "staking_rewards" in result["normalized_data"].columns
    assert "trading_rewards" in result["normalized_data"].columns
    assert "overall_rewards" in result["normalized_data"].columns
    
    # Verify data normalization
    assert len(result["normalized_data"]) == len(test_reward_data)
    assert result["normalized_data"]["mining_rewards"].min() >= 0
    assert result["normalized_data"]["mining_rewards"].max() <= 1
    assert result["normalized_data"]["staking_rewards"].min() >= 0
    assert result["normalized_data"]["staking_rewards"].max() <= 1
    assert result["normalized_data"]["trading_rewards"].min() >= 0
    assert result["normalized_data"]["trading_rewards"].max() <= 1
    assert result["normalized_data"]["overall_rewards"].min() >= 0
    assert result["normalized_data"]["overall_rewards"].max() <= 1
    
    # Verify database entry
    db_record = db_session.query(NormalizationRecord).filter_by(
        user_id=test_user.user_id,
        normalization_id=result["normalization_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_normalize_activity_data(db_session, test_user, test_activity_data, test_normalization_config):
    """Test normalizing activity data"""
    # Normalize activity data
    result = normalize_data(
        user_id=test_user.user_id,
        data=test_activity_data,
        normalization_config=test_normalization_config,
        db_session=db_session
    )
    
    # Verify normalization result
    assert isinstance(result, Dict)
    assert "normalization_id" in result
    assert "normalized_data" in result
    
    # Verify normalization metadata
    assert result["normalization_type"] == "minmax"
    assert result["feature_range"] == (0, 1)
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    
    # Verify normalization details
    assert "normalization_details" in result
    assert isinstance(result["normalization_details"], Dict)
    assert "timestamp" in result["normalization_details"]
    assert "original_shape" in result["normalization_details"]
    assert "normalized_shape" in result["normalization_details"]
    assert "scalers" in result["normalization_details"]
    
    # Verify normalized data
    assert isinstance(result["normalized_data"], pd.DataFrame)
    assert "day" in result["normalized_data"].columns
    assert "mining_activity" in result["normalized_data"].columns
    assert "staking_activity" in result["normalized_data"].columns
    assert "trading_activity" in result["normalized_data"].columns
    assert "overall_activity" in result["normalized_data"].columns
    
    # Verify data normalization
    assert len(result["normalized_data"]) == len(test_activity_data)
    assert result["normalized_data"]["mining_activity"].min() >= 0
    assert result["normalized_data"]["mining_activity"].max() <= 1
    assert result["normalized_data"]["staking_activity"].min() >= 0
    assert result["normalized_data"]["staking_activity"].max() <= 1
    assert result["normalized_data"]["trading_activity"].min() >= 0
    assert result["normalized_data"]["trading_activity"].max() <= 1
    assert result["normalized_data"]["overall_activity"].min() >= 0
    assert result["normalized_data"]["overall_activity"].max() <= 1
    
    # Verify database entry
    db_record = db_session.query(NormalizationRecord).filter_by(
        user_id=test_user.user_id,
        normalization_id=result["normalization_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_normalize_with_standard_scaler(db_session, test_user, test_performance_data):
    """Test normalizing with standard scaler"""
    # Create normalization config with standard scaler
    standard_config = {
        "normalization_type": "standard",
        "feature_range": None,  # Not used for standard scaler
        "columns": None,  # Normalize all numeric columns
        "exclude_columns": ["timestamp", "day"]
    }
    
    # Normalize performance data
    result = normalize_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        normalization_config=standard_config,
        db_session=db_session
    )
    
    # Verify normalization result
    assert isinstance(result, Dict)
    assert "normalization_id" in result
    assert "normalized_data" in result
    
    # Verify normalization metadata
    assert result["normalization_type"] == "standard"
    assert result["feature_range"] is None
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    
    # Verify normalization details
    assert "normalization_details" in result
    assert isinstance(result["normalization_details"], Dict)
    assert "timestamp" in result["normalization_details"]
    assert "original_shape" in result["normalization_details"]
    assert "normalized_shape" in result["normalization_details"]
    assert "scalers" in result["normalization_details"]
    
    # Verify data normalization
    assert len(result["normalized_data"]) == len(test_performance_data)
    # Standard scaler should have mean close to 0 and std close to 1
    assert abs(result["normalized_data"]["mining_performance"].mean()) < 0.1
    assert abs(result["normalized_data"]["mining_performance"].std() - 1) < 0.1
    assert abs(result["normalized_data"]["staking_performance"].mean()) < 0.1
    assert abs(result["normalized_data"]["staking_performance"].std() - 1) < 0.1
    assert abs(result["normalized_data"]["trading_performance"].mean()) < 0.1
    assert abs(result["normalized_data"]["trading_performance"].std() - 1) < 0.1
    assert abs(result["normalized_data"]["overall_performance"].mean()) < 0.1
    assert abs(result["normalized_data"]["overall_performance"].std() - 1) < 0.1
    
    # Verify database entry
    db_record = db_session.query(NormalizationRecord).filter_by(
        user_id=test_user.user_id,
        normalization_id=result["normalization_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_normalize_with_robust_scaler(db_session, test_user, test_performance_data):
    """Test normalizing with robust scaler"""
    # Create normalization config with robust scaler
    robust_config = {
        "normalization_type": "robust",
        "feature_range": None,  # Not used for robust scaler
        "columns": None,  # Normalize all numeric columns
        "exclude_columns": ["timestamp", "day"]
    }
    
    # Normalize performance data
    result = normalize_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        normalization_config=robust_config,
        db_session=db_session
    )
    
    # Verify normalization result
    assert isinstance(result, Dict)
    assert "normalization_id" in result
    assert "normalized_data" in result
    
    # Verify normalization metadata
    assert result["normalization_type"] == "robust"
    assert result["feature_range"] is None
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    
    # Verify normalization details
    assert "normalization_details" in result
    assert isinstance(result["normalization_details"], Dict)
    assert "timestamp" in result["normalization_details"]
    assert "original_shape" in result["normalization_details"]
    assert "normalized_shape" in result["normalization_details"]
    assert "scalers" in result["normalization_details"]
    
    # Verify data normalization
    assert len(result["normalized_data"]) == len(test_performance_data)
    # Robust scaler should have median close to 0
    assert abs(result["normalized_data"]["mining_performance"].median()) < 0.1
    assert abs(result["normalized_data"]["staking_performance"].median()) < 0.1
    assert abs(result["normalized_data"]["trading_performance"].median()) < 0.1
    assert abs(result["normalized_data"]["overall_performance"].median()) < 0.1
    
    # Verify database entry
    db_record = db_session.query(NormalizationRecord).filter_by(
        user_id=test_user.user_id,
        normalization_id=result["normalization_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_normalize_with_custom_range(db_session, test_user, test_performance_data):
    """Test normalizing with custom range"""
    # Create normalization config with custom range
    custom_range_config = {
        "normalization_type": "minmax",
        "feature_range": (-1, 1),  # Custom range
        "columns": None,  # Normalize all numeric columns
        "exclude_columns": ["timestamp", "day"]
    }
    
    # Normalize performance data
    result = normalize_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        normalization_config=custom_range_config,
        db_session=db_session
    )
    
    # Verify normalization result
    assert isinstance(result, Dict)
    assert "normalization_id" in result
    assert "normalized_data" in result
    
    # Verify normalization metadata
    assert result["normalization_type"] == "minmax"
    assert result["feature_range"] == (-1, 1)
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    
    # Verify normalization details
    assert "normalization_details" in result
    assert isinstance(result["normalization_details"], Dict)
    assert "timestamp" in result["normalization_details"]
    assert "original_shape" in result["normalization_details"]
    assert "normalized_shape" in result["normalization_details"]
    assert "scalers" in result["normalization_details"]
    
    # Verify data normalization
    assert len(result["normalized_data"]) == len(test_performance_data)
    assert result["normalized_data"]["mining_performance"].min() >= -1
    assert result["normalized_data"]["mining_performance"].max() <= 1
    assert result["normalized_data"]["staking_performance"].min() >= -1
    assert result["normalized_data"]["staking_performance"].max() <= 1
    assert result["normalized_data"]["trading_performance"].min() >= -1
    assert result["normalized_data"]["trading_performance"].max() <= 1
    assert result["normalized_data"]["overall_performance"].min() >= -1
    assert result["normalized_data"]["overall_performance"].max() <= 1
    
    # Verify database entry
    db_record = db_session.query(NormalizationRecord).filter_by(
        user_id=test_user.user_id,
        normalization_id=result["normalization_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_normalize_with_specific_columns(db_session, test_user, test_performance_data):
    """Test normalizing with specific columns"""
    # Create normalization config with specific columns
    specific_columns_config = {
        "normalization_type": "minmax",
        "feature_range": (0, 1),
        "columns": ["mining_performance", "trading_performance"],  # Only normalize these columns
        "exclude_columns": ["timestamp", "day"]
    }
    
    # Normalize performance data
    result = normalize_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        normalization_config=specific_columns_config,
        db_session=db_session
    )
    
    # Verify normalization result
    assert isinstance(result, Dict)
    assert "normalization_id" in result
    assert "normalized_data" in result
    
    # Verify normalization metadata
    assert result["normalization_type"] == "minmax"
    assert result["feature_range"] == (0, 1)
    assert result["columns"] == ["mining_performance", "trading_performance"]
    assert result["exclude_columns"] == ["timestamp", "day"]
    
    # Verify normalization details
    assert "normalization_details" in result
    assert isinstance(result["normalization_details"], Dict)
    assert "timestamp" in result["normalization_details"]
    assert "original_shape" in result["normalization_details"]
    assert "normalized_shape" in result["normalization_details"]
    assert "scalers" in result["normalization_details"]
    
    # Verify data normalization
    assert len(result["normalized_data"]) == len(test_performance_data)
    # Check normalized columns
    assert result["normalized_data"]["mining_performance"].min() >= 0
    assert result["normalized_data"]["mining_performance"].max() <= 1
    assert result["normalized_data"]["trading_performance"].min() >= 0
    assert result["normalized_data"]["trading_performance"].max() <= 1
    # Check non-normalized columns (should be unchanged)
    assert result["normalized_data"]["staking_performance"].equals(test_performance_data["staking_performance"])
    assert result["normalized_data"]["overall_performance"].equals(test_performance_data["overall_performance"])
    
    # Verify database entry
    db_record = db_session.query(NormalizationRecord).filter_by(
        user_id=test_user.user_id,
        normalization_id=result["normalization_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_get_normalization_info(db_session, test_user, test_performance_data, test_normalization_config):
    """Test normalization info retrieval"""
    # First, normalize performance data
    normalization_result = normalize_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        normalization_config=test_normalization_config,
        db_session=db_session
    )
    
    normalization_id = normalization_result["normalization_id"]
    
    # Get normalization info
    result = get_normalization_info(
        user_id=test_user.user_id,
        normalization_id=normalization_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "normalization_id" in result
    assert result["normalization_id"] == normalization_id
    
    # Verify normalization metadata
    assert result["normalization_type"] == "minmax"
    assert result["feature_range"] == (0, 1)
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    
    # Verify normalization details
    assert "normalization_details" in result
    assert isinstance(result["normalization_details"], Dict)
    assert "timestamp" in result["normalization_details"]
    assert "original_shape" in result["normalization_details"]
    assert "normalized_shape" in result["normalization_details"]
    assert "scalers" in result["normalization_details"]
    
    # Verify database entry
    db_record = db_session.query(NormalizationRecord).filter_by(
        user_id=test_user.user_id,
        normalization_id=normalization_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_normalization_error_handling(db_session, test_user):
    """Test normalization error handling"""
    # Invalid user ID
    with pytest.raises(NormalizationError) as excinfo:
        normalize_data(
            user_id=None,
            data=pd.DataFrame(),
            normalization_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(NormalizationError) as excinfo:
        normalize_data(
            user_id=test_user.user_id,
            data=None,
            normalization_config={},
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Invalid normalization type
    with pytest.raises(NormalizationError) as excinfo:
        normalize_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            normalization_config={"normalization_type": "invalid_type"},
            db_session=db_session
        )
    assert "Invalid normalization type" in str(excinfo.value)
    
    # Invalid feature range
    with pytest.raises(NormalizationError) as excinfo:
        normalize_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            normalization_config={"normalization_type": "minmax", "feature_range": (1, 0)},
            db_session=db_session
        )
    assert "Invalid feature range" in str(excinfo.value)
    
    # Invalid columns
    with pytest.raises(NormalizationError) as excinfo:
        normalize_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            normalization_config={"normalization_type": "minmax", "columns": ["invalid_column"]},
            db_session=db_session
        )
    assert "Invalid columns" in str(excinfo.value)
    
    # Invalid normalization ID
    with pytest.raises(NormalizationError) as excinfo:
        get_normalization_info(
            user_id=test_user.user_id,
            normalization_id="invalid_normalization_id",
            db_session=db_session
        )
    assert "Invalid normalization ID" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBNormalizationError) as excinfo:
        normalize_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            normalization_config={"normalization_type": "minmax"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 