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

from core.splitting import (
    split_data,
    get_split_info,
    SplitError
)
from database.models import User, SplitRecord
from database.exceptions import SplitError as DBSplitError

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
    # Create data with multiple days to test splitting
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
    # Create data with multiple days to test splitting
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
    # Create data with multiple days to test splitting
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
    # Create data with multiple days to test splitting
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
def test_split_config():
    """Create test split configuration"""
    return {
        "split_type": "train_test",
        "split_ratio": 0.8,
        "random_state": 42,
        "stratify": "day",
        "shuffle": True
    }

def test_split_performance_data(db_session, test_user, test_performance_data, test_split_config):
    """Test splitting performance data"""
    # Split performance data
    result = split_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        split_config=test_split_config,
        db_session=db_session
    )
    
    # Verify split result
    assert isinstance(result, Dict)
    assert "split_id" in result
    assert "train_data" in result
    assert "test_data" in result
    
    # Verify split metadata
    assert result["split_type"] == "train_test"
    assert result["split_ratio"] == 0.8
    assert result["random_state"] == 42
    assert result["stratify"] == "day"
    assert result["shuffle"] is True
    
    # Verify split details
    assert "split_details" in result
    assert isinstance(result["split_details"], Dict)
    assert "timestamp" in result["split_details"]
    assert "original_shape" in result["split_details"]
    assert "train_shape" in result["split_details"]
    assert "test_shape" in result["split_details"]
    assert "train_ratio" in result["split_details"]
    assert "test_ratio" in result["split_details"]
    
    # Verify train data
    assert isinstance(result["train_data"], pd.DataFrame)
    assert "day" in result["train_data"].columns
    assert "mining_performance" in result["train_data"].columns
    assert "staking_performance" in result["train_data"].columns
    assert "trading_performance" in result["train_data"].columns
    assert "overall_performance" in result["train_data"].columns
    
    # Verify test data
    assert isinstance(result["test_data"], pd.DataFrame)
    assert "day" in result["test_data"].columns
    assert "mining_performance" in result["test_data"].columns
    assert "staking_performance" in result["test_data"].columns
    assert "trading_performance" in result["test_data"].columns
    assert "overall_performance" in result["test_data"].columns
    
    # Verify data splitting
    assert len(result["train_data"]) + len(result["test_data"]) == len(test_performance_data)
    assert abs(len(result["train_data"]) / len(test_performance_data) - 0.8) < 0.1
    assert abs(len(result["test_data"]) / len(test_performance_data) - 0.2) < 0.1
    
    # Verify database entry
    db_record = db_session.query(SplitRecord).filter_by(
        user_id=test_user.user_id,
        split_id=result["split_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_split_risk_data(db_session, test_user, test_risk_data, test_split_config):
    """Test splitting risk data"""
    # Split risk data
    result = split_data(
        user_id=test_user.user_id,
        data=test_risk_data,
        split_config=test_split_config,
        db_session=db_session
    )
    
    # Verify split result
    assert isinstance(result, Dict)
    assert "split_id" in result
    assert "train_data" in result
    assert "test_data" in result
    
    # Verify split metadata
    assert result["split_type"] == "train_test"
    assert result["split_ratio"] == 0.8
    assert result["random_state"] == 42
    assert result["stratify"] == "day"
    assert result["shuffle"] is True
    
    # Verify split details
    assert "split_details" in result
    assert isinstance(result["split_details"], Dict)
    assert "timestamp" in result["split_details"]
    assert "original_shape" in result["split_details"]
    assert "train_shape" in result["split_details"]
    assert "test_shape" in result["split_details"]
    assert "train_ratio" in result["split_details"]
    assert "test_ratio" in result["split_details"]
    
    # Verify train data
    assert isinstance(result["train_data"], pd.DataFrame)
    assert "day" in result["train_data"].columns
    assert "mining_risk" in result["train_data"].columns
    assert "staking_risk" in result["train_data"].columns
    assert "trading_risk" in result["train_data"].columns
    assert "overall_risk" in result["train_data"].columns
    
    # Verify test data
    assert isinstance(result["test_data"], pd.DataFrame)
    assert "day" in result["test_data"].columns
    assert "mining_risk" in result["test_data"].columns
    assert "staking_risk" in result["test_data"].columns
    assert "trading_risk" in result["test_data"].columns
    assert "overall_risk" in result["test_data"].columns
    
    # Verify data splitting
    assert len(result["train_data"]) + len(result["test_data"]) == len(test_risk_data)
    assert abs(len(result["train_data"]) / len(test_risk_data) - 0.8) < 0.1
    assert abs(len(result["test_data"]) / len(test_risk_data) - 0.2) < 0.1
    
    # Verify database entry
    db_record = db_session.query(SplitRecord).filter_by(
        user_id=test_user.user_id,
        split_id=result["split_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_split_reward_data(db_session, test_user, test_reward_data, test_split_config):
    """Test splitting reward data"""
    # Split reward data
    result = split_data(
        user_id=test_user.user_id,
        data=test_reward_data,
        split_config=test_split_config,
        db_session=db_session
    )
    
    # Verify split result
    assert isinstance(result, Dict)
    assert "split_id" in result
    assert "train_data" in result
    assert "test_data" in result
    
    # Verify split metadata
    assert result["split_type"] == "train_test"
    assert result["split_ratio"] == 0.8
    assert result["random_state"] == 42
    assert result["stratify"] == "day"
    assert result["shuffle"] is True
    
    # Verify split details
    assert "split_details" in result
    assert isinstance(result["split_details"], Dict)
    assert "timestamp" in result["split_details"]
    assert "original_shape" in result["split_details"]
    assert "train_shape" in result["split_details"]
    assert "test_shape" in result["split_details"]
    assert "train_ratio" in result["split_details"]
    assert "test_ratio" in result["split_details"]
    
    # Verify train data
    assert isinstance(result["train_data"], pd.DataFrame)
    assert "day" in result["train_data"].columns
    assert "mining_rewards" in result["train_data"].columns
    assert "staking_rewards" in result["train_data"].columns
    assert "trading_rewards" in result["train_data"].columns
    assert "overall_rewards" in result["train_data"].columns
    
    # Verify test data
    assert isinstance(result["test_data"], pd.DataFrame)
    assert "day" in result["test_data"].columns
    assert "mining_rewards" in result["test_data"].columns
    assert "staking_rewards" in result["test_data"].columns
    assert "trading_rewards" in result["test_data"].columns
    assert "overall_rewards" in result["test_data"].columns
    
    # Verify data splitting
    assert len(result["train_data"]) + len(result["test_data"]) == len(test_reward_data)
    assert abs(len(result["train_data"]) / len(test_reward_data) - 0.8) < 0.1
    assert abs(len(result["test_data"]) / len(test_reward_data) - 0.2) < 0.1
    
    # Verify database entry
    db_record = db_session.query(SplitRecord).filter_by(
        user_id=test_user.user_id,
        split_id=result["split_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_split_activity_data(db_session, test_user, test_activity_data, test_split_config):
    """Test splitting activity data"""
    # Split activity data
    result = split_data(
        user_id=test_user.user_id,
        data=test_activity_data,
        split_config=test_split_config,
        db_session=db_session
    )
    
    # Verify split result
    assert isinstance(result, Dict)
    assert "split_id" in result
    assert "train_data" in result
    assert "test_data" in result
    
    # Verify split metadata
    assert result["split_type"] == "train_test"
    assert result["split_ratio"] == 0.8
    assert result["random_state"] == 42
    assert result["stratify"] == "day"
    assert result["shuffle"] is True
    
    # Verify split details
    assert "split_details" in result
    assert isinstance(result["split_details"], Dict)
    assert "timestamp" in result["split_details"]
    assert "original_shape" in result["split_details"]
    assert "train_shape" in result["split_details"]
    assert "test_shape" in result["split_details"]
    assert "train_ratio" in result["split_details"]
    assert "test_ratio" in result["split_details"]
    
    # Verify train data
    assert isinstance(result["train_data"], pd.DataFrame)
    assert "day" in result["train_data"].columns
    assert "mining_activity" in result["train_data"].columns
    assert "staking_activity" in result["train_data"].columns
    assert "trading_activity" in result["train_data"].columns
    assert "overall_activity" in result["train_data"].columns
    
    # Verify test data
    assert isinstance(result["test_data"], pd.DataFrame)
    assert "day" in result["test_data"].columns
    assert "mining_activity" in result["test_data"].columns
    assert "staking_activity" in result["test_data"].columns
    assert "trading_activity" in result["test_data"].columns
    assert "overall_activity" in result["test_data"].columns
    
    # Verify data splitting
    assert len(result["train_data"]) + len(result["test_data"]) == len(test_activity_data)
    assert abs(len(result["train_data"]) / len(test_activity_data) - 0.8) < 0.1
    assert abs(len(result["test_data"]) / len(test_activity_data) - 0.2) < 0.1
    
    # Verify database entry
    db_record = db_session.query(SplitRecord).filter_by(
        user_id=test_user.user_id,
        split_id=result["split_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_split_with_custom_ratio(db_session, test_user, test_performance_data):
    """Test splitting with custom ratio"""
    # Create custom split config
    custom_split_config = {
        "split_type": "train_test",
        "split_ratio": 0.7,
        "random_state": 42,
        "stratify": "day",
        "shuffle": True
    }
    
    # Split performance data
    result = split_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        split_config=custom_split_config,
        db_session=db_session
    )
    
    # Verify split result
    assert isinstance(result, Dict)
    assert "split_id" in result
    assert "train_data" in result
    assert "test_data" in result
    
    # Verify split metadata
    assert result["split_type"] == "train_test"
    assert result["split_ratio"] == 0.7
    assert result["random_state"] == 42
    assert result["stratify"] == "day"
    assert result["shuffle"] is True
    
    # Verify split details
    assert "split_details" in result
    assert isinstance(result["split_details"], Dict)
    assert "timestamp" in result["split_details"]
    assert "original_shape" in result["split_details"]
    assert "train_shape" in result["split_details"]
    assert "test_shape" in result["split_details"]
    assert "train_ratio" in result["split_details"]
    assert "test_ratio" in result["split_details"]
    
    # Verify data splitting
    assert len(result["train_data"]) + len(result["test_data"]) == len(test_performance_data)
    assert abs(len(result["train_data"]) / len(test_performance_data) - 0.7) < 0.1
    assert abs(len(result["test_data"]) / len(test_performance_data) - 0.3) < 0.1
    
    # Verify database entry
    db_record = db_session.query(SplitRecord).filter_by(
        user_id=test_user.user_id,
        split_id=result["split_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_split_without_stratification(db_session, test_user, test_performance_data):
    """Test splitting without stratification"""
    # Create split config without stratification
    no_stratify_config = {
        "split_type": "train_test",
        "split_ratio": 0.8,
        "random_state": 42,
        "stratify": None,
        "shuffle": True
    }
    
    # Split performance data
    result = split_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        split_config=no_stratify_config,
        db_session=db_session
    )
    
    # Verify split result
    assert isinstance(result, Dict)
    assert "split_id" in result
    assert "train_data" in result
    assert "test_data" in result
    
    # Verify split metadata
    assert result["split_type"] == "train_test"
    assert result["split_ratio"] == 0.8
    assert result["random_state"] == 42
    assert result["stratify"] is None
    assert result["shuffle"] is True
    
    # Verify split details
    assert "split_details" in result
    assert isinstance(result["split_details"], Dict)
    assert "timestamp" in result["split_details"]
    assert "original_shape" in result["split_details"]
    assert "train_shape" in result["split_details"]
    assert "test_shape" in result["split_details"]
    assert "train_ratio" in result["split_details"]
    assert "test_ratio" in result["split_details"]
    
    # Verify data splitting
    assert len(result["train_data"]) + len(result["test_data"]) == len(test_performance_data)
    assert abs(len(result["train_data"]) / len(test_performance_data) - 0.8) < 0.1
    assert abs(len(result["test_data"]) / len(test_performance_data) - 0.2) < 0.1
    
    # Verify database entry
    db_record = db_session.query(SplitRecord).filter_by(
        user_id=test_user.user_id,
        split_id=result["split_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_split_without_shuffle(db_session, test_user, test_performance_data):
    """Test splitting without shuffle"""
    # Create split config without shuffle
    no_shuffle_config = {
        "split_type": "train_test",
        "split_ratio": 0.8,
        "random_state": 42,
        "stratify": "day",
        "shuffle": False
    }
    
    # Split performance data
    result = split_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        split_config=no_shuffle_config,
        db_session=db_session
    )
    
    # Verify split result
    assert isinstance(result, Dict)
    assert "split_id" in result
    assert "train_data" in result
    assert "test_data" in result
    
    # Verify split metadata
    assert result["split_type"] == "train_test"
    assert result["split_ratio"] == 0.8
    assert result["random_state"] == 42
    assert result["stratify"] == "day"
    assert result["shuffle"] is False
    
    # Verify split details
    assert "split_details" in result
    assert isinstance(result["split_details"], Dict)
    assert "timestamp" in result["split_details"]
    assert "original_shape" in result["split_details"]
    assert "train_shape" in result["split_details"]
    assert "test_shape" in result["split_details"]
    assert "train_ratio" in result["split_details"]
    assert "test_ratio" in result["split_details"]
    
    # Verify data splitting
    assert len(result["train_data"]) + len(result["test_data"]) == len(test_performance_data)
    assert abs(len(result["train_data"]) / len(test_performance_data) - 0.8) < 0.1
    assert abs(len(result["test_data"]) / len(test_performance_data) - 0.2) < 0.1
    
    # Verify database entry
    db_record = db_session.query(SplitRecord).filter_by(
        user_id=test_user.user_id,
        split_id=result["split_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_get_split_info(db_session, test_user, test_performance_data, test_split_config):
    """Test split info retrieval"""
    # First, split performance data
    split_result = split_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        split_config=test_split_config,
        db_session=db_session
    )
    
    split_id = split_result["split_id"]
    
    # Get split info
    result = get_split_info(
        user_id=test_user.user_id,
        split_id=split_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "split_id" in result
    assert result["split_id"] == split_id
    
    # Verify split metadata
    assert result["split_type"] == "train_test"
    assert result["split_ratio"] == 0.8
    assert result["random_state"] == 42
    assert result["stratify"] == "day"
    assert result["shuffle"] is True
    
    # Verify split details
    assert "split_details" in result
    assert isinstance(result["split_details"], Dict)
    assert "timestamp" in result["split_details"]
    assert "original_shape" in result["split_details"]
    assert "train_shape" in result["split_details"]
    assert "test_shape" in result["split_details"]
    assert "train_ratio" in result["split_details"]
    assert "test_ratio" in result["split_details"]
    
    # Verify database entry
    db_record = db_session.query(SplitRecord).filter_by(
        user_id=test_user.user_id,
        split_id=split_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_split_error_handling(db_session, test_user):
    """Test split error handling"""
    # Invalid user ID
    with pytest.raises(SplitError) as excinfo:
        split_data(
            user_id=None,
            data=pd.DataFrame(),
            split_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(SplitError) as excinfo:
        split_data(
            user_id=test_user.user_id,
            data=None,
            split_config={},
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Invalid split type
    with pytest.raises(SplitError) as excinfo:
        split_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            split_config={"split_type": "invalid_type"},
            db_session=db_session
        )
    assert "Invalid split type" in str(excinfo.value)
    
    # Invalid split ratio
    with pytest.raises(SplitError) as excinfo:
        split_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            split_config={"split_type": "train_test", "split_ratio": 1.5},
            db_session=db_session
        )
    assert "Invalid split ratio" in str(excinfo.value)
    
    # Invalid stratify column
    with pytest.raises(SplitError) as excinfo:
        split_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            split_config={"split_type": "train_test", "stratify": "invalid_column"},
            db_session=db_session
        )
    assert "Invalid stratify column" in str(excinfo.value)
    
    # Invalid split ID
    with pytest.raises(SplitError) as excinfo:
        get_split_info(
            user_id=test_user.user_id,
            split_id="invalid_split_id",
            db_session=db_session
        )
    assert "Invalid split ID" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBSplitError) as excinfo:
        split_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            split_config={"split_type": "train_test"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 