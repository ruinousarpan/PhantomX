import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from decimal import Decimal
import json
import os
import tempfile
import logging
from logging.handlers import RotatingFileHandler
import uuid

from core.logging import (
    log_performance_data,
    log_risk_data,
    log_reward_data,
    log_activity_data,
    log_analytics_data,
    customize_log,
    manage_log,
    LoggingError
)
from database.models import User, Log, LogEntry
from database.exceptions import LoggingError as DBLoggingError

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
def test_log_config():
    """Create test log configuration"""
    return {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "handlers": [
            {
                "type": "file",
                "filename": "test_log.log",
                "max_bytes": 10485760,  # 10MB
                "backup_count": 5
            },
            {
                "type": "console",
                "stream": "stdout"
            }
        ],
        "options": {
            "rotation": "daily",
            "compression": True,
            "retention": 30,  # days
            "notification": {
                "on_error": True,
                "recipients": ["test@example.com"]
            }
        }
    }

def test_log_performance_data(db_session, test_user, test_performance_data, test_log_config):
    """Test performance data logging"""
    # Log performance data
    result = log_performance_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        log_config=test_log_config,
        db_session=db_session
    )
    
    # Verify logging result
    assert isinstance(result, Dict)
    assert "log_id" in result
    assert "entry_id" in result
    
    # Verify log metadata
    assert result["log_type"] == "PERFORMANCE_LOG"
    assert result["log_config"]["level"] == "INFO"
    
    # Verify log details
    assert "log_details" in result
    assert isinstance(result["log_details"], Dict)
    assert "timestamp" in result["log_details"]
    assert "level" in result["log_details"]
    assert "message" in result["log_details"]
    
    # Verify database entry
    db_log = db_session.query(Log).filter_by(
        user_id=test_user.user_id,
        log_type="PERFORMANCE_LOG"
    ).first()
    assert db_log is not None
    assert db_log.is_active is True
    assert db_log.error is None
    
    # Verify log entry
    db_entry = db_session.query(LogEntry).filter_by(
        log_id=db_log.log_id
    ).first()
    assert db_entry is not None
    assert db_entry.level == "INFO"
    assert "Performance data logged" in db_entry.message

def test_log_risk_data(db_session, test_user, test_risk_data, test_log_config):
    """Test risk data logging"""
    # Log risk data
    result = log_risk_data(
        user_id=test_user.user_id,
        data=test_risk_data,
        log_config=test_log_config,
        db_session=db_session
    )
    
    # Verify logging result
    assert isinstance(result, Dict)
    assert "log_id" in result
    assert "entry_id" in result
    
    # Verify log metadata
    assert result["log_type"] == "RISK_LOG"
    assert result["log_config"]["level"] == "INFO"
    
    # Verify log details
    assert "log_details" in result
    assert isinstance(result["log_details"], Dict)
    assert "timestamp" in result["log_details"]
    assert "level" in result["log_details"]
    assert "message" in result["log_details"]
    
    # Verify database entry
    db_log = db_session.query(Log).filter_by(
        user_id=test_user.user_id,
        log_type="RISK_LOG"
    ).first()
    assert db_log is not None
    assert db_log.is_active is True
    assert db_log.error is None
    
    # Verify log entry
    db_entry = db_session.query(LogEntry).filter_by(
        log_id=db_log.log_id
    ).first()
    assert db_entry is not None
    assert db_entry.level == "INFO"
    assert "Risk data logged" in db_entry.message

def test_log_reward_data(db_session, test_user, test_reward_data, test_log_config):
    """Test reward data logging"""
    # Log reward data
    result = log_reward_data(
        user_id=test_user.user_id,
        data=test_reward_data,
        log_config=test_log_config,
        db_session=db_session
    )
    
    # Verify logging result
    assert isinstance(result, Dict)
    assert "log_id" in result
    assert "entry_id" in result
    
    # Verify log metadata
    assert result["log_type"] == "REWARD_LOG"
    assert result["log_config"]["level"] == "INFO"
    
    # Verify log details
    assert "log_details" in result
    assert isinstance(result["log_details"], Dict)
    assert "timestamp" in result["log_details"]
    assert "level" in result["log_details"]
    assert "message" in result["log_details"]
    
    # Verify database entry
    db_log = db_session.query(Log).filter_by(
        user_id=test_user.user_id,
        log_type="REWARD_LOG"
    ).first()
    assert db_log is not None
    assert db_log.is_active is True
    assert db_log.error is None
    
    # Verify log entry
    db_entry = db_session.query(LogEntry).filter_by(
        log_id=db_log.log_id
    ).first()
    assert db_entry is not None
    assert db_entry.level == "INFO"
    assert "Reward data logged" in db_entry.message

def test_log_activity_data(db_session, test_user, test_activity_data, test_log_config):
    """Test activity data logging"""
    # Log activity data
    result = log_activity_data(
        user_id=test_user.user_id,
        data=test_activity_data,
        log_config=test_log_config,
        db_session=db_session
    )
    
    # Verify logging result
    assert isinstance(result, Dict)
    assert "log_id" in result
    assert "entry_id" in result
    
    # Verify log metadata
    assert result["log_type"] == "ACTIVITY_LOG"
    assert result["log_config"]["level"] == "INFO"
    
    # Verify log details
    assert "log_details" in result
    assert isinstance(result["log_details"], Dict)
    assert "timestamp" in result["log_details"]
    assert "level" in result["log_details"]
    assert "message" in result["log_details"]
    
    # Verify database entry
    db_log = db_session.query(Log).filter_by(
        user_id=test_user.user_id,
        log_type="ACTIVITY_LOG"
    ).first()
    assert db_log is not None
    assert db_log.is_active is True
    assert db_log.error is None
    
    # Verify log entry
    db_entry = db_session.query(LogEntry).filter_by(
        log_id=db_log.log_id
    ).first()
    assert db_entry is not None
    assert db_entry.level == "INFO"
    assert "Activity data logged" in db_entry.message

def test_log_analytics_data(db_session, test_user, test_analytics_data, test_log_config):
    """Test analytics data logging"""
    # Log analytics data
    result = log_analytics_data(
        user_id=test_user.user_id,
        data=test_analytics_data,
        log_config=test_log_config,
        db_session=db_session
    )
    
    # Verify logging result
    assert isinstance(result, Dict)
    assert "log_id" in result
    assert "entry_id" in result
    
    # Verify log metadata
    assert result["log_type"] == "ANALYTICS_LOG"
    assert result["log_config"]["level"] == "INFO"
    
    # Verify log details
    assert "log_details" in result
    assert isinstance(result["log_details"], Dict)
    assert "timestamp" in result["log_details"]
    assert "level" in result["log_details"]
    assert "message" in result["log_details"]
    
    # Verify database entry
    db_log = db_session.query(Log).filter_by(
        user_id=test_user.user_id,
        log_type="ANALYTICS_LOG"
    ).first()
    assert db_log is not None
    assert db_log.is_active is True
    assert db_log.error is None
    
    # Verify log entry
    db_entry = db_session.query(LogEntry).filter_by(
        log_id=db_log.log_id
    ).first()
    assert db_entry is not None
    assert db_entry.level == "INFO"
    assert "Analytics data logged" in db_entry.message

def test_customize_log(db_session, test_user, test_performance_data, test_log_config):
    """Test log customization"""
    # Customize log
    result = customize_log(
        user_id=test_user.user_id,
        data=test_performance_data,
        log_config=test_log_config,
        db_session=db_session
    )
    
    # Verify customization result
    assert isinstance(result, Dict)
    assert "log_id" in result
    assert "entry_id" in result
    
    # Verify log metadata
    assert result["log_type"] == "CUSTOMIZED_LOG"
    assert result["log_config"]["level"] == "INFO"
    
    # Verify log details
    assert "log_details" in result
    assert isinstance(result["log_details"], Dict)
    assert "timestamp" in result["log_details"]
    assert "level" in result["log_details"]
    assert "message" in result["log_details"]
    
    # Verify database entry
    db_log = db_session.query(Log).filter_by(
        user_id=test_user.user_id,
        log_type="CUSTOMIZED_LOG"
    ).first()
    assert db_log is not None
    assert db_log.is_active is True
    assert db_log.error is None
    
    # Verify log entry
    db_entry = db_session.query(LogEntry).filter_by(
        log_id=db_log.log_id
    ).first()
    assert db_entry is not None
    assert db_entry.level == "INFO"
    assert "Customized log created" in db_entry.message

def test_manage_log(db_session, test_user, test_performance_data, test_log_config):
    """Test log management"""
    # First, create a log to get a log ID
    log_result = log_performance_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        log_config=test_log_config,
        db_session=db_session
    )
    
    log_id = log_result["log_id"]
    
    # Test log management operations
    # 1. Archive log
    archive_result = manage_log(
        user_id=test_user.user_id,
        log_id=log_id,
        action="archive",
        db_session=db_session
    )
    
    assert isinstance(archive_result, Dict)
    assert "log_id" in archive_result
    assert "status" in archive_result
    assert archive_result["status"] == "ARCHIVED"
    
    # 2. Restore log
    restore_result = manage_log(
        user_id=test_user.user_id,
        log_id=log_id,
        action="restore",
        db_session=db_session
    )
    
    assert isinstance(restore_result, Dict)
    assert "log_id" in restore_result
    assert "status" in restore_result
    assert restore_result["status"] == "ACTIVE"
    
    # 3. Modify log
    modified_config = test_log_config.copy()
    modified_config["level"] = "DEBUG"
    
    modify_result = manage_log(
        user_id=test_user.user_id,
        log_id=log_id,
        action="modify",
        log_config=modified_config,
        db_session=db_session
    )
    
    assert isinstance(modify_result, Dict)
    assert "log_id" in modify_result
    assert "status" in modify_result
    assert "log_details" in modify_result
    assert modify_result["log_details"]["level"] == "DEBUG"
    
    # 4. Delete log
    delete_result = manage_log(
        user_id=test_user.user_id,
        log_id=log_id,
        action="delete",
        db_session=db_session
    )
    
    assert isinstance(delete_result, Dict)
    assert "log_id" in delete_result
    assert "status" in delete_result
    assert delete_result["status"] == "DELETED"
    
    # Verify database entry
    db_log = db_session.query(Log).filter_by(
        user_id=test_user.user_id,
        log_id=log_id
    ).first()
    assert db_log is not None
    assert db_log.is_active is False
    assert db_log.error is None

def test_logging_error_handling(db_session, test_user):
    """Test logging error handling"""
    # Invalid user ID
    with pytest.raises(LoggingError) as excinfo:
        log_performance_data(
            user_id=None,
            data=pd.DataFrame(),
            log_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(LoggingError) as excinfo:
        log_risk_data(
            user_id=test_user.user_id,
            data=None,
            log_config={},
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Empty data
    with pytest.raises(LoggingError) as excinfo:
        log_reward_data(
            user_id=test_user.user_id,
            data=pd.DataFrame(),
            log_config={},
            db_session=db_session
        )
    assert "Empty data" in str(excinfo.value)
    
    # Invalid log level
    with pytest.raises(LoggingError) as excinfo:
        log_activity_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            log_config={"level": "INVALID_LEVEL"},
            db_session=db_session
        )
    assert "Invalid log level" in str(excinfo.value)
    
    # Invalid log handler
    with pytest.raises(LoggingError) as excinfo:
        log_analytics_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            log_config={"handlers": [{"type": "invalid_handler"}]},
            db_session=db_session
        )
    assert "Invalid log handler" in str(excinfo.value)
    
    # Invalid log action
    with pytest.raises(LoggingError) as excinfo:
        manage_log(
            user_id=test_user.user_id,
            log_id="test_log_id",
            action="invalid_action",
            db_session=db_session
        )
    assert "Invalid log action" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBLoggingError) as excinfo:
        log_performance_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            log_config={"level": "INFO"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 