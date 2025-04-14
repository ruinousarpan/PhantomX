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
import croniter
import pytz
from unittest.mock import patch, MagicMock

from core.scheduling import (
    schedule_task,
    get_schedule_info,
    update_schedule,
    cancel_schedule,
    get_execution_history,
    SchedulingError
)
from database.models import User, ScheduleRecord, ExecutionRecord
from database.exceptions import SchedulingError as DBSchedulingError

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
def test_schedule_config():
    """Create test schedule configuration"""
    return {
        "task_type": "performance_analysis",
        "schedule_type": "cron",
        "schedule": "0 0 * * *",  # Daily at midnight
        "timezone": "UTC",
        "data_type": "performance",
        "parameters": {
            "analysis_type": "trend",
            "window_size": 7,
            "metrics": ["mining_performance", "staking_performance", "trading_performance"]
        },
        "notification": {
            "enabled": True,
            "channels": ["email"],
            "recipients": ["test@example.com"],
            "conditions": {
                "mining_performance": {"operator": "<", "value": 0.7},
                "trading_performance": {"operator": "<", "value": 0.6}
            }
        },
        "retry": {
            "enabled": True,
            "max_attempts": 3,
            "delay": 300  # 5 minutes
        },
        "metadata": {
            "description": "Daily performance analysis",
            "priority": "high",
            "tags": ["performance", "daily", "analysis"]
        }
    }

def test_schedule_performance_analysis(db_session, test_user, test_performance_data, test_schedule_config):
    """Test scheduling performance analysis task"""
    # Schedule performance analysis task
    result = schedule_task(
        user_id=test_user.user_id,
        schedule_config=test_schedule_config,
        db_session=db_session
    )
    
    # Verify schedule result
    assert isinstance(result, Dict)
    assert "schedule_id" in result
    assert "next_execution" in result
    assert "schedule_details" in result
    
    # Verify schedule metadata
    assert result["task_type"] == "performance_analysis"
    assert result["schedule_type"] == "cron"
    assert result["schedule"] == "0 0 * * *"
    assert result["timezone"] == "UTC"
    assert result["data_type"] == "performance"
    
    # Verify parameters
    assert "parameters" in result
    parameters = result["parameters"]
    assert parameters["analysis_type"] == "trend"
    assert parameters["window_size"] == 7
    assert all(metric in parameters["metrics"] for metric in ["mining_performance", "staking_performance", "trading_performance"])
    
    # Verify notification settings
    assert "notification" in result
    notification = result["notification"]
    assert notification["enabled"] is True
    assert "email" in notification["channels"]
    assert "test@example.com" in notification["recipients"]
    assert "conditions" in notification
    
    # Verify retry settings
    assert "retry" in result
    retry = result["retry"]
    assert retry["enabled"] is True
    assert retry["max_attempts"] == 3
    assert retry["delay"] == 300
    
    # Verify metadata
    assert "metadata" in result
    metadata = result["metadata"]
    assert metadata["description"] == "Daily performance analysis"
    assert metadata["priority"] == "high"
    assert all(tag in metadata["tags"] for tag in ["performance", "daily", "analysis"])
    
    # Verify next execution time
    assert isinstance(result["next_execution"], datetime)
    
    # Verify database entry
    db_record = db_session.query(ScheduleRecord).filter_by(
        user_id=test_user.user_id,
        schedule_id=result["schedule_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_schedule_risk_monitoring(db_session, test_user, test_risk_data):
    """Test scheduling risk monitoring task"""
    # Create schedule config for risk monitoring
    risk_config = {
        "task_type": "risk_monitoring",
        "schedule_type": "interval",
        "schedule": 3600,  # Every hour
        "timezone": "UTC",
        "data_type": "risk",
        "parameters": {
            "monitoring_type": "threshold",
            "thresholds": {
                "mining_risk": 0.3,
                "staking_risk": 0.2,
                "trading_risk": 0.4
            },
            "metrics": ["mining_risk", "staking_risk", "trading_risk", "overall_risk"]
        },
        "notification": {
            "enabled": True,
            "channels": ["email", "webhook"],
            "recipients": ["test@example.com"],
            "webhook_url": "https://example.com/webhook",
            "conditions": {
                "mining_risk": {"operator": ">", "value": 0.3},
                "trading_risk": {"operator": ">", "value": 0.4}
            }
        },
        "retry": {
            "enabled": True,
            "max_attempts": 5,
            "delay": 600  # 10 minutes
        },
        "metadata": {
            "description": "Hourly risk monitoring",
            "priority": "critical",
            "tags": ["risk", "monitoring", "hourly"]
        }
    }
    
    # Schedule risk monitoring task
    result = schedule_task(
        user_id=test_user.user_id,
        schedule_config=risk_config,
        db_session=db_session
    )
    
    # Verify schedule result
    assert isinstance(result, Dict)
    assert "schedule_id" in result
    assert "next_execution" in result
    assert "schedule_details" in result
    
    # Verify schedule metadata
    assert result["task_type"] == "risk_monitoring"
    assert result["schedule_type"] == "interval"
    assert result["schedule"] == 3600
    assert result["timezone"] == "UTC"
    assert result["data_type"] == "risk"
    
    # Verify parameters
    assert "parameters" in result
    parameters = result["parameters"]
    assert parameters["monitoring_type"] == "threshold"
    assert "thresholds" in parameters
    assert all(metric in parameters["metrics"] for metric in ["mining_risk", "staking_risk", "trading_risk", "overall_risk"])
    
    # Verify notification settings
    assert "notification" in result
    notification = result["notification"]
    assert notification["enabled"] is True
    assert all(channel in notification["channels"] for channel in ["email", "webhook"])
    assert "test@example.com" in notification["recipients"]
    assert notification["webhook_url"] == "https://example.com/webhook"
    assert "conditions" in notification
    
    # Verify retry settings
    assert "retry" in result
    retry = result["retry"]
    assert retry["enabled"] is True
    assert retry["max_attempts"] == 5
    assert retry["delay"] == 600
    
    # Verify metadata
    assert "metadata" in result
    metadata = result["metadata"]
    assert metadata["description"] == "Hourly risk monitoring"
    assert metadata["priority"] == "critical"
    assert all(tag in metadata["tags"] for tag in ["risk", "monitoring", "hourly"])
    
    # Verify next execution time
    assert isinstance(result["next_execution"], datetime)
    
    # Verify database entry
    db_record = db_session.query(ScheduleRecord).filter_by(
        user_id=test_user.user_id,
        schedule_id=result["schedule_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_schedule_reward_export(db_session, test_user, test_reward_data):
    """Test scheduling reward data export task"""
    # Create schedule config for reward export
    reward_config = {
        "task_type": "reward_export",
        "schedule_type": "date",
        "schedule": "2023-12-31 23:59:59",
        "timezone": "UTC",
        "data_type": "reward",
        "parameters": {
            "export_type": "csv",
            "data_columns": ["mining_reward", "staking_reward", "trading_reward", "overall_reward"],
            "time_column": "timestamp",
            "options": {
                "delimiter": ",",
                "encoding": "utf-8",
                "header": True,
                "index": False,
                "date_format": "%Y-%m-%d %H:%M:%S"
            }
        },
        "notification": {
            "enabled": True,
            "channels": ["email"],
            "recipients": ["test@example.com"],
            "on_success": True,
            "on_failure": True
        },
        "retry": {
            "enabled": False
        },
        "metadata": {
            "description": "Year-end reward data export",
            "priority": "medium",
            "tags": ["reward", "export", "yearly"]
        }
    }
    
    # Schedule reward export task
    result = schedule_task(
        user_id=test_user.user_id,
        schedule_config=reward_config,
        db_session=db_session
    )
    
    # Verify schedule result
    assert isinstance(result, Dict)
    assert "schedule_id" in result
    assert "next_execution" in result
    assert "schedule_details" in result
    
    # Verify schedule metadata
    assert result["task_type"] == "reward_export"
    assert result["schedule_type"] == "date"
    assert result["schedule"] == "2023-12-31 23:59:59"
    assert result["timezone"] == "UTC"
    assert result["data_type"] == "reward"
    
    # Verify parameters
    assert "parameters" in result
    parameters = result["parameters"]
    assert parameters["export_type"] == "csv"
    assert all(col in parameters["data_columns"] for col in ["mining_reward", "staking_reward", "trading_reward", "overall_reward"])
    assert parameters["time_column"] == "timestamp"
    assert "options" in parameters
    
    # Verify notification settings
    assert "notification" in result
    notification = result["notification"]
    assert notification["enabled"] is True
    assert "email" in notification["channels"]
    assert "test@example.com" in notification["recipients"]
    assert notification["on_success"] is True
    assert notification["on_failure"] is True
    
    # Verify retry settings
    assert "retry" in result
    retry = result["retry"]
    assert retry["enabled"] is False
    
    # Verify metadata
    assert "metadata" in result
    metadata = result["metadata"]
    assert metadata["description"] == "Year-end reward data export"
    assert metadata["priority"] == "medium"
    assert all(tag in metadata["tags"] for tag in ["reward", "export", "yearly"])
    
    # Verify next execution time
    assert isinstance(result["next_execution"], datetime)
    
    # Verify database entry
    db_record = db_session.query(ScheduleRecord).filter_by(
        user_id=test_user.user_id,
        schedule_id=result["schedule_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_schedule_activity_report(db_session, test_user, test_activity_data):
    """Test scheduling activity report task"""
    # Create schedule config for activity report
    activity_config = {
        "task_type": "activity_report",
        "schedule_type": "cron",
        "schedule": "0 0 * * 1",  # Every Monday at midnight
        "timezone": "UTC",
        "data_type": "activity",
        "parameters": {
            "report_type": "weekly",
            "metrics": ["mining_activity", "staking_activity", "trading_activity", "overall_activity"],
            "aggregation": {
                "min": True,
                "max": True,
                "mean": True,
                "std": True
            },
            "visualization": {
                "enabled": True,
                "type": "line",
                "title": "Weekly Activity Report"
            }
        },
        "notification": {
            "enabled": True,
            "channels": ["email"],
            "recipients": ["test@example.com"],
            "attachments": ["report.pdf", "visualization.png"]
        },
        "retry": {
            "enabled": True,
            "max_attempts": 2,
            "delay": 1800  # 30 minutes
        },
        "metadata": {
            "description": "Weekly activity report",
            "priority": "medium",
            "tags": ["activity", "report", "weekly"]
        }
    }
    
    # Schedule activity report task
    result = schedule_task(
        user_id=test_user.user_id,
        schedule_config=activity_config,
        db_session=db_session
    )
    
    # Verify schedule result
    assert isinstance(result, Dict)
    assert "schedule_id" in result
    assert "next_execution" in result
    assert "schedule_details" in result
    
    # Verify schedule metadata
    assert result["task_type"] == "activity_report"
    assert result["schedule_type"] == "cron"
    assert result["schedule"] == "0 0 * * 1"
    assert result["timezone"] == "UTC"
    assert result["data_type"] == "activity"
    
    # Verify parameters
    assert "parameters" in result
    parameters = result["parameters"]
    assert parameters["report_type"] == "weekly"
    assert all(metric in parameters["metrics"] for metric in ["mining_activity", "staking_activity", "trading_activity", "overall_activity"])
    assert "aggregation" in parameters
    assert "visualization" in parameters
    
    # Verify notification settings
    assert "notification" in result
    notification = result["notification"]
    assert notification["enabled"] is True
    assert "email" in notification["channels"]
    assert "test@example.com" in notification["recipients"]
    assert all(attachment in notification["attachments"] for attachment in ["report.pdf", "visualization.png"])
    
    # Verify retry settings
    assert "retry" in result
    retry = result["retry"]
    assert retry["enabled"] is True
    assert retry["max_attempts"] == 2
    assert retry["delay"] == 1800
    
    # Verify metadata
    assert "metadata" in result
    metadata = result["metadata"]
    assert metadata["description"] == "Weekly activity report"
    assert metadata["priority"] == "medium"
    assert all(tag in metadata["tags"] for tag in ["activity", "report", "weekly"])
    
    # Verify next execution time
    assert isinstance(result["next_execution"], datetime)
    
    # Verify database entry
    db_record = db_session.query(ScheduleRecord).filter_by(
        user_id=test_user.user_id,
        schedule_id=result["schedule_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_update_schedule(db_session, test_user, test_performance_data, test_schedule_config):
    """Test updating a scheduled task"""
    # First, schedule a task
    schedule_result = schedule_task(
        user_id=test_user.user_id,
        schedule_config=test_schedule_config,
        db_session=db_session
    )
    
    schedule_id = schedule_result["schedule_id"]
    
    # Create updated schedule config
    updated_config = test_schedule_config.copy()
    updated_config["schedule"] = "0 12 * * *"  # Change to daily at noon
    updated_config["parameters"]["window_size"] = 14  # Change window size
    updated_config["notification"]["recipients"].append("updated@example.com")
    
    # Update schedule
    result = update_schedule(
        user_id=test_user.user_id,
        schedule_id=schedule_id,
        schedule_config=updated_config,
        db_session=db_session
    )
    
    # Verify update result
    assert isinstance(result, Dict)
    assert "schedule_id" in result
    assert result["schedule_id"] == schedule_id
    assert "next_execution" in result
    assert "schedule_details" in result
    
    # Verify updated schedule
    assert result["schedule"] == "0 12 * * *"
    assert result["parameters"]["window_size"] == 14
    assert "updated@example.com" in result["notification"]["recipients"]
    
    # Verify database entry
    db_record = db_session.query(ScheduleRecord).filter_by(
        user_id=test_user.user_id,
        schedule_id=schedule_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_cancel_schedule(db_session, test_user, test_performance_data, test_schedule_config):
    """Test canceling a scheduled task"""
    # First, schedule a task
    schedule_result = schedule_task(
        user_id=test_user.user_id,
        schedule_config=test_schedule_config,
        db_session=db_session
    )
    
    schedule_id = schedule_result["schedule_id"]
    
    # Cancel schedule
    result = cancel_schedule(
        user_id=test_user.user_id,
        schedule_id=schedule_id,
        db_session=db_session
    )
    
    # Verify cancel result
    assert isinstance(result, Dict)
    assert "schedule_id" in result
    assert result["schedule_id"] == schedule_id
    assert "status" in result
    assert result["status"] == "CANCELED"
    
    # Verify database entry
    db_record = db_session.query(ScheduleRecord).filter_by(
        user_id=test_user.user_id,
        schedule_id=schedule_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is False
    assert db_record.error is None

def test_get_schedule_info(db_session, test_user, test_performance_data, test_schedule_config):
    """Test retrieving schedule information"""
    # First, schedule a task
    schedule_result = schedule_task(
        user_id=test_user.user_id,
        schedule_config=test_schedule_config,
        db_session=db_session
    )
    
    schedule_id = schedule_result["schedule_id"]
    
    # Get schedule info
    result = get_schedule_info(
        user_id=test_user.user_id,
        schedule_id=schedule_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "schedule_id" in result
    assert result["schedule_id"] == schedule_id
    
    # Verify schedule metadata
    assert result["task_type"] == "performance_analysis"
    assert result["schedule_type"] == "cron"
    assert result["schedule"] == "0 0 * * *"
    assert result["timezone"] == "UTC"
    assert result["data_type"] == "performance"
    
    # Verify schedule details
    assert "schedule_details" in result
    assert isinstance(result["schedule_details"], Dict)
    assert "created_at" in result["schedule_details"]
    assert "last_modified" in result["schedule_details"]
    assert "next_execution" in result["schedule_details"]
    assert "execution_count" in result["schedule_details"]
    
    # Verify database entry
    db_record = db_session.query(ScheduleRecord).filter_by(
        user_id=test_user.user_id,
        schedule_id=schedule_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_get_execution_history(db_session, test_user, test_performance_data, test_schedule_config):
    """Test retrieving execution history"""
    # First, schedule a task
    schedule_result = schedule_task(
        user_id=test_user.user_id,
        schedule_config=test_schedule_config,
        db_session=db_session
    )
    
    schedule_id = schedule_result["schedule_id"]
    
    # Create mock execution records
    execution_records = []
    for i in range(5):
        execution_record = ExecutionRecord(
            execution_id=str(uuid.uuid4()),
            user_id=test_user.user_id,
            schedule_id=schedule_id,
            status="COMPLETED" if i % 2 == 0 else "FAILED",
            started_at=datetime.utcnow() - timedelta(hours=i),
            completed_at=datetime.utcnow() - timedelta(hours=i) + timedelta(minutes=5),
            error=None if i % 2 == 0 else "Test error",
            result={"execution_number": i} if i % 2 == 0 else None
        )
        execution_records.append(execution_record)
        db_session.add(execution_record)
    
    db_session.commit()
    
    # Get execution history
    result = get_execution_history(
        user_id=test_user.user_id,
        schedule_id=schedule_id,
        db_session=db_session
    )
    
    # Verify history result
    assert isinstance(result, List)
    assert len(result) == 5
    
    # Verify execution records
    for i, record in enumerate(result):
        assert "execution_id" in record
        assert "status" in record
        assert "started_at" in record
        assert "completed_at" in record
        assert "duration" in record
        
        if i % 2 == 0:
            assert record["status"] == "COMPLETED"
            assert record["error"] is None
            assert "result" in record
        else:
            assert record["status"] == "FAILED"
            assert record["error"] == "Test error"
            assert "result" not in record

def test_scheduling_error_handling(db_session, test_user):
    """Test scheduling error handling"""
    # Invalid user ID
    with pytest.raises(SchedulingError) as excinfo:
        schedule_task(
            user_id=None,
            schedule_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid task type
    with pytest.raises(SchedulingError) as excinfo:
        schedule_task(
            user_id=test_user.user_id,
            schedule_config={"task_type": "invalid_type"},
            db_session=db_session
        )
    assert "Invalid task type" in str(excinfo.value)
    
    # Invalid schedule type
    with pytest.raises(SchedulingError) as excinfo:
        schedule_task(
            user_id=test_user.user_id,
            schedule_config={"task_type": "performance_analysis", "schedule_type": "invalid_type"},
            db_session=db_session
        )
    assert "Invalid schedule type" in str(excinfo.value)
    
    # Invalid schedule
    with pytest.raises(SchedulingError) as excinfo:
        schedule_task(
            user_id=test_user.user_id,
            schedule_config={"task_type": "performance_analysis", "schedule_type": "cron", "schedule": "invalid_cron"},
            db_session=db_session
        )
    assert "Invalid schedule" in str(excinfo.value)
    
    # Invalid schedule ID
    with pytest.raises(SchedulingError) as excinfo:
        get_schedule_info(
            user_id=test_user.user_id,
            schedule_id="invalid_schedule_id",
            db_session=db_session
        )
    assert "Invalid schedule ID" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBSchedulingError) as excinfo:
        schedule_task(
            user_id=test_user.user_id,
            schedule_config={"task_type": "performance_analysis"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 