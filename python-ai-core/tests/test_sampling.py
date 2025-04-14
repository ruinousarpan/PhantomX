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

from core.sampling import (
    sample_data,
    get_sample_info,
    SampleError
)
from database.models import User, SampleRecord
from database.exceptions import SampleError as DBSampleError

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
    # Create data with multiple days to test sampling
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
    # Create data with multiple days to test sampling
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
    # Create data with multiple days to test sampling
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
    # Create data with multiple days to test sampling
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
def test_sample_config():
    """Create test sample configuration"""
    return {
        "sample_type": "random",
        "sample_size": 50,
        "random_state": 42,
        "replace": False,
        "weights": None
    }

def test_sample_performance_data(db_session, test_user, test_performance_data, test_sample_config):
    """Test sampling performance data"""
    # Sample performance data
    result = sample_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        sample_config=test_sample_config,
        db_session=db_session
    )
    
    # Verify sample result
    assert isinstance(result, Dict)
    assert "sample_id" in result
    assert "sampled_data" in result
    
    # Verify sample metadata
    assert result["sample_type"] == "random"
    assert result["sample_size"] == 50
    assert result["random_state"] == 42
    assert result["replace"] is False
    assert result["weights"] is None
    
    # Verify sample details
    assert "sample_details" in result
    assert isinstance(result["sample_details"], Dict)
    assert "timestamp" in result["sample_details"]
    assert "original_shape" in result["sample_details"]
    assert "sampled_shape" in result["sample_details"]
    assert "sampling_ratio" in result["sample_details"]
    
    # Verify sampled data
    assert isinstance(result["sampled_data"], pd.DataFrame)
    assert "day" in result["sampled_data"].columns
    assert "mining_performance" in result["sampled_data"].columns
    assert "staking_performance" in result["sampled_data"].columns
    assert "trading_performance" in result["sampled_data"].columns
    assert "overall_performance" in result["sampled_data"].columns
    
    # Verify data sampling
    assert len(result["sampled_data"]) == 50
    assert abs(len(result["sampled_data"]) / len(test_performance_data) - 0.5) < 0.1
    
    # Verify database entry
    db_record = db_session.query(SampleRecord).filter_by(
        user_id=test_user.user_id,
        sample_id=result["sample_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_sample_risk_data(db_session, test_user, test_risk_data, test_sample_config):
    """Test sampling risk data"""
    # Sample risk data
    result = sample_data(
        user_id=test_user.user_id,
        data=test_risk_data,
        sample_config=test_sample_config,
        db_session=db_session
    )
    
    # Verify sample result
    assert isinstance(result, Dict)
    assert "sample_id" in result
    assert "sampled_data" in result
    
    # Verify sample metadata
    assert result["sample_type"] == "random"
    assert result["sample_size"] == 50
    assert result["random_state"] == 42
    assert result["replace"] is False
    assert result["weights"] is None
    
    # Verify sample details
    assert "sample_details" in result
    assert isinstance(result["sample_details"], Dict)
    assert "timestamp" in result["sample_details"]
    assert "original_shape" in result["sample_details"]
    assert "sampled_shape" in result["sample_details"]
    assert "sampling_ratio" in result["sample_details"]
    
    # Verify sampled data
    assert isinstance(result["sampled_data"], pd.DataFrame)
    assert "day" in result["sampled_data"].columns
    assert "mining_risk" in result["sampled_data"].columns
    assert "staking_risk" in result["sampled_data"].columns
    assert "trading_risk" in result["sampled_data"].columns
    assert "overall_risk" in result["sampled_data"].columns
    
    # Verify data sampling
    assert len(result["sampled_data"]) == 50
    assert abs(len(result["sampled_data"]) / len(test_risk_data) - 0.5) < 0.1
    
    # Verify database entry
    db_record = db_session.query(SampleRecord).filter_by(
        user_id=test_user.user_id,
        sample_id=result["sample_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_sample_reward_data(db_session, test_user, test_reward_data, test_sample_config):
    """Test sampling reward data"""
    # Sample reward data
    result = sample_data(
        user_id=test_user.user_id,
        data=test_reward_data,
        sample_config=test_sample_config,
        db_session=db_session
    )
    
    # Verify sample result
    assert isinstance(result, Dict)
    assert "sample_id" in result
    assert "sampled_data" in result
    
    # Verify sample metadata
    assert result["sample_type"] == "random"
    assert result["sample_size"] == 50
    assert result["random_state"] == 42
    assert result["replace"] is False
    assert result["weights"] is None
    
    # Verify sample details
    assert "sample_details" in result
    assert isinstance(result["sample_details"], Dict)
    assert "timestamp" in result["sample_details"]
    assert "original_shape" in result["sample_details"]
    assert "sampled_shape" in result["sample_details"]
    assert "sampling_ratio" in result["sample_details"]
    
    # Verify sampled data
    assert isinstance(result["sampled_data"], pd.DataFrame)
    assert "day" in result["sampled_data"].columns
    assert "mining_rewards" in result["sampled_data"].columns
    assert "staking_rewards" in result["sampled_data"].columns
    assert "trading_rewards" in result["sampled_data"].columns
    assert "overall_rewards" in result["sampled_data"].columns
    
    # Verify data sampling
    assert len(result["sampled_data"]) == 50
    assert abs(len(result["sampled_data"]) / len(test_reward_data) - 0.5) < 0.1
    
    # Verify database entry
    db_record = db_session.query(SampleRecord).filter_by(
        user_id=test_user.user_id,
        sample_id=result["sample_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_sample_activity_data(db_session, test_user, test_activity_data, test_sample_config):
    """Test sampling activity data"""
    # Sample activity data
    result = sample_data(
        user_id=test_user.user_id,
        data=test_activity_data,
        sample_config=test_sample_config,
        db_session=db_session
    )
    
    # Verify sample result
    assert isinstance(result, Dict)
    assert "sample_id" in result
    assert "sampled_data" in result
    
    # Verify sample metadata
    assert result["sample_type"] == "random"
    assert result["sample_size"] == 50
    assert result["random_state"] == 42
    assert result["replace"] is False
    assert result["weights"] is None
    
    # Verify sample details
    assert "sample_details" in result
    assert isinstance(result["sample_details"], Dict)
    assert "timestamp" in result["sample_details"]
    assert "original_shape" in result["sample_details"]
    assert "sampled_shape" in result["sample_details"]
    assert "sampling_ratio" in result["sample_details"]
    
    # Verify sampled data
    assert isinstance(result["sampled_data"], pd.DataFrame)
    assert "day" in result["sampled_data"].columns
    assert "mining_activity" in result["sampled_data"].columns
    assert "staking_activity" in result["sampled_data"].columns
    assert "trading_activity" in result["sampled_data"].columns
    assert "overall_activity" in result["sampled_data"].columns
    
    # Verify data sampling
    assert len(result["sampled_data"]) == 50
    assert abs(len(result["sampled_data"]) / len(test_activity_data) - 0.5) < 0.1
    
    # Verify database entry
    db_record = db_session.query(SampleRecord).filter_by(
        user_id=test_user.user_id,
        sample_id=result["sample_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_sample_with_replacement(db_session, test_user, test_performance_data):
    """Test sampling with replacement"""
    # Create sample config with replacement
    replacement_config = {
        "sample_type": "random",
        "sample_size": 50,
        "random_state": 42,
        "replace": True,
        "weights": None
    }
    
    # Sample performance data
    result = sample_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        sample_config=replacement_config,
        db_session=db_session
    )
    
    # Verify sample result
    assert isinstance(result, Dict)
    assert "sample_id" in result
    assert "sampled_data" in result
    
    # Verify sample metadata
    assert result["sample_type"] == "random"
    assert result["sample_size"] == 50
    assert result["random_state"] == 42
    assert result["replace"] is True
    assert result["weights"] is None
    
    # Verify sample details
    assert "sample_details" in result
    assert isinstance(result["sample_details"], Dict)
    assert "timestamp" in result["sample_details"]
    assert "original_shape" in result["sample_details"]
    assert "sampled_shape" in result["sample_details"]
    assert "sampling_ratio" in result["sample_details"]
    
    # Verify data sampling
    assert len(result["sampled_data"]) == 50
    assert abs(len(result["sampled_data"]) / len(test_performance_data) - 0.5) < 0.1
    
    # Verify database entry
    db_record = db_session.query(SampleRecord).filter_by(
        user_id=test_user.user_id,
        sample_id=result["sample_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_sample_with_weights(db_session, test_user, test_performance_data):
    """Test sampling with weights"""
    # Create sample config with weights
    weights_config = {
        "sample_type": "random",
        "sample_size": 50,
        "random_state": 42,
        "replace": False,
        "weights": "mining_performance"  # Use mining_performance as weights
    }
    
    # Sample performance data
    result = sample_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        sample_config=weights_config,
        db_session=db_session
    )
    
    # Verify sample result
    assert isinstance(result, Dict)
    assert "sample_id" in result
    assert "sampled_data" in result
    
    # Verify sample metadata
    assert result["sample_type"] == "random"
    assert result["sample_size"] == 50
    assert result["random_state"] == 42
    assert result["replace"] is False
    assert result["weights"] == "mining_performance"
    
    # Verify sample details
    assert "sample_details" in result
    assert isinstance(result["sample_details"], Dict)
    assert "timestamp" in result["sample_details"]
    assert "original_shape" in result["sample_details"]
    assert "sampled_shape" in result["sample_details"]
    assert "sampling_ratio" in result["sample_details"]
    
    # Verify data sampling
    assert len(result["sampled_data"]) == 50
    assert abs(len(result["sampled_data"]) / len(test_performance_data) - 0.5) < 0.1
    
    # Verify database entry
    db_record = db_session.query(SampleRecord).filter_by(
        user_id=test_user.user_id,
        sample_id=result["sample_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_sample_with_stratified(db_session, test_user, test_performance_data):
    """Test stratified sampling"""
    # Create sample config for stratified sampling
    stratified_config = {
        "sample_type": "stratified",
        "sample_size": 50,
        "random_state": 42,
        "replace": False,
        "weights": None,
        "stratify": "day"  # Stratify by day
    }
    
    # Sample performance data
    result = sample_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        sample_config=stratified_config,
        db_session=db_session
    )
    
    # Verify sample result
    assert isinstance(result, Dict)
    assert "sample_id" in result
    assert "sampled_data" in result
    
    # Verify sample metadata
    assert result["sample_type"] == "stratified"
    assert result["sample_size"] == 50
    assert result["random_state"] == 42
    assert result["replace"] is False
    assert result["weights"] is None
    assert result["stratify"] == "day"
    
    # Verify sample details
    assert "sample_details" in result
    assert isinstance(result["sample_details"], Dict)
    assert "timestamp" in result["sample_details"]
    assert "original_shape" in result["sample_details"]
    assert "sampled_shape" in result["sample_details"]
    assert "sampling_ratio" in result["sample_details"]
    
    # Verify data sampling
    assert len(result["sampled_data"]) == 50
    assert abs(len(result["sampled_data"]) / len(test_performance_data) - 0.5) < 0.1
    
    # Verify database entry
    db_record = db_session.query(SampleRecord).filter_by(
        user_id=test_user.user_id,
        sample_id=result["sample_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_sample_with_systematic(db_session, test_user, test_performance_data):
    """Test systematic sampling"""
    # Create sample config for systematic sampling
    systematic_config = {
        "sample_type": "systematic",
        "sample_size": 50,
        "random_state": 42,
        "replace": False,
        "weights": None,
        "step": 2  # Sample every 2nd record
    }
    
    # Sample performance data
    result = sample_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        sample_config=systematic_config,
        db_session=db_session
    )
    
    # Verify sample result
    assert isinstance(result, Dict)
    assert "sample_id" in result
    assert "sampled_data" in result
    
    # Verify sample metadata
    assert result["sample_type"] == "systematic"
    assert result["sample_size"] == 50
    assert result["random_state"] == 42
    assert result["replace"] is False
    assert result["weights"] is None
    assert result["step"] == 2
    
    # Verify sample details
    assert "sample_details" in result
    assert isinstance(result["sample_details"], Dict)
    assert "timestamp" in result["sample_details"]
    assert "original_shape" in result["sample_details"]
    assert "sampled_shape" in result["sample_details"]
    assert "sampling_ratio" in result["sample_details"]
    
    # Verify data sampling
    assert len(result["sampled_data"]) == 50
    assert abs(len(result["sampled_data"]) / len(test_performance_data) - 0.5) < 0.1
    
    # Verify database entry
    db_record = db_session.query(SampleRecord).filter_by(
        user_id=test_user.user_id,
        sample_id=result["sample_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_get_sample_info(db_session, test_user, test_performance_data, test_sample_config):
    """Test sample info retrieval"""
    # First, sample performance data
    sample_result = sample_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        sample_config=test_sample_config,
        db_session=db_session
    )
    
    sample_id = sample_result["sample_id"]
    
    # Get sample info
    result = get_sample_info(
        user_id=test_user.user_id,
        sample_id=sample_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "sample_id" in result
    assert result["sample_id"] == sample_id
    
    # Verify sample metadata
    assert result["sample_type"] == "random"
    assert result["sample_size"] == 50
    assert result["random_state"] == 42
    assert result["replace"] is False
    assert result["weights"] is None
    
    # Verify sample details
    assert "sample_details" in result
    assert isinstance(result["sample_details"], Dict)
    assert "timestamp" in result["sample_details"]
    assert "original_shape" in result["sample_details"]
    assert "sampled_shape" in result["sample_details"]
    assert "sampling_ratio" in result["sample_details"]
    
    # Verify database entry
    db_record = db_session.query(SampleRecord).filter_by(
        user_id=test_user.user_id,
        sample_id=sample_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_sample_error_handling(db_session, test_user):
    """Test sample error handling"""
    # Invalid user ID
    with pytest.raises(SampleError) as excinfo:
        sample_data(
            user_id=None,
            data=pd.DataFrame(),
            sample_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(SampleError) as excinfo:
        sample_data(
            user_id=test_user.user_id,
            data=None,
            sample_config={},
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Invalid sample type
    with pytest.raises(SampleError) as excinfo:
        sample_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            sample_config={"sample_type": "invalid_type"},
            db_session=db_session
        )
    assert "Invalid sample type" in str(excinfo.value)
    
    # Invalid sample size
    with pytest.raises(SampleError) as excinfo:
        sample_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            sample_config={"sample_type": "random", "sample_size": 0},
            db_session=db_session
        )
    assert "Invalid sample size" in str(excinfo.value)
    
    # Invalid stratify column
    with pytest.raises(SampleError) as excinfo:
        sample_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            sample_config={"sample_type": "stratified", "stratify": "invalid_column"},
            db_session=db_session
        )
    assert "Invalid stratify column" in str(excinfo.value)
    
    # Invalid sample ID
    with pytest.raises(SampleError) as excinfo:
        get_sample_info(
            user_id=test_user.user_id,
            sample_id="invalid_sample_id",
            db_session=db_session
        )
    assert "Invalid sample ID" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBSampleError) as excinfo:
        sample_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            sample_config={"sample_type": "random"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 