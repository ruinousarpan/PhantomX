import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from decimal import Decimal
import json
import os
import tempfile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import uuid
from unittest.mock import patch, MagicMock

from core.notification import (
    generate_performance_notification,
    generate_risk_notification,
    generate_reward_notification,
    generate_activity_notification,
    generate_analytics_notification,
    customize_notification,
    deliver_notification,
    NotificationError,
    create_notification,
    configure_notification,
    trigger_notification,
    format_message,
    send_notification,
    track_notification_history,
    get_notification_info,
    list_notifications,
    delete_notification
)
from database.models import User, Notification, NotificationDelivery, NotificationRecord, NotificationConfig, NotificationHistory
from database.exceptions import NotificationError as DBNotificationError

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
def test_notification_config():
    """Create test notification configuration"""
    return {
        "notification_types": {
            "enabled": True,
            "types": ["alert", "warning", "info", "success", "error"],
            "priorities": ["high", "medium", "low"],
            "categories": ["system", "security", "performance", "compliance"]
        },
        "event_triggers": {
            "enabled": True,
            "trigger_types": ["threshold", "schedule", "event", "condition"],
            "conditions": {
                "threshold": {
                    "operators": ["gt", "lt", "ge", "le", "eq", "ne"],
                    "value_types": ["numeric", "string", "boolean", "datetime"]
                },
                "schedule": {
                    "frequencies": ["once", "hourly", "daily", "weekly", "monthly"],
                    "timezone": "UTC"
                },
                "event": {
                    "sources": ["system", "user", "external"],
                    "types": ["create", "update", "delete", "error"]
                }
            }
        },
        "message_formatting": {
            "enabled": True,
            "templates": {
                "alert": "{severity}: {message}",
                "warning": "Warning: {message}",
                "info": "Info: {message}",
                "success": "Success: {message}",
                "error": "Error: {message}"
            },
            "placeholders": ["severity", "message", "timestamp", "user", "details"],
            "max_length": 1000,
            "supported_formats": ["text", "html", "markdown"]
        },
        "delivery_channels": {
            "enabled": True,
            "channels": {
                "email": {
                    "enabled": True,
                    "provider": "smtp",
                    "config": {
                        "host": "smtp.example.com",
                        "port": 587,
                        "use_tls": True
                    }
                },
                "sms": {
                    "enabled": True,
                    "provider": "twilio",
                    "config": {
                        "account_sid": "test_sid",
                        "auth_token": "test_token"
                    }
                },
                "webhook": {
                    "enabled": True,
                    "endpoints": ["https://api.example.com/webhook"],
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"}
                },
                "in_app": {
                    "enabled": True,
                    "storage": "database",
                    "max_notifications": 100
                }
            },
            "retry_policy": {
                "max_retries": 3,
                "retry_delay": 300,
                "timeout": 30
            }
        },
        "history_tracking": {
            "enabled": True,
            "track_metadata": True,
            "track_delivery": True,
            "track_responses": True,
            "retention_period": 90,
            "max_history_records": 1000
        }
    }

def test_generate_performance_notification(db_session, test_user, test_performance_data, test_notification_config):
    """Test performance notification generation"""
    # Generate performance notification
    result = generate_performance_notification(
        user_id=test_user.user_id,
        data=test_performance_data,
        notification_config=test_notification_config,
        db_session=db_session
    )
    
    # Verify notification result
    assert isinstance(result, Dict)
    assert "notification_id" in result
    assert "notification_type" in result
    
    # Verify notification metadata
    assert result["notification_type"] == "PERFORMANCE_NOTIFICATION"
    assert result["notification_config"]["type"] == "email"
    
    # Verify notification content
    assert "notification_content" in result
    assert isinstance(result["notification_content"], Dict)
    assert "title" in result["notification_content"]
    assert "message" in result["notification_content"]
    assert "data" in result["notification_content"]
    
    # Verify database entry
    db_notification = db_session.query(Notification).filter_by(
        user_id=test_user.user_id,
        notification_type="PERFORMANCE_NOTIFICATION"
    ).first()
    assert db_notification is not None
    assert db_notification.is_success is True
    assert db_notification.error is None

def test_generate_risk_notification(db_session, test_user, test_risk_data, test_notification_config):
    """Test risk notification generation"""
    # Generate risk notification
    result = generate_risk_notification(
        user_id=test_user.user_id,
        data=test_risk_data,
        notification_config=test_notification_config,
        db_session=db_session
    )
    
    # Verify notification result
    assert isinstance(result, Dict)
    assert "notification_id" in result
    assert "notification_type" in result
    
    # Verify notification metadata
    assert result["notification_type"] == "RISK_NOTIFICATION"
    assert result["notification_config"]["type"] == "email"
    
    # Verify notification content
    assert "notification_content" in result
    assert isinstance(result["notification_content"], Dict)
    assert "title" in result["notification_content"]
    assert "message" in result["notification_content"]
    assert "data" in result["notification_content"]
    
    # Verify database entry
    db_notification = db_session.query(Notification).filter_by(
        user_id=test_user.user_id,
        notification_type="RISK_NOTIFICATION"
    ).first()
    assert db_notification is not None
    assert db_notification.is_success is True
    assert db_notification.error is None

def test_generate_reward_notification(db_session, test_user, test_reward_data, test_notification_config):
    """Test reward notification generation"""
    # Generate reward notification
    result = generate_reward_notification(
        user_id=test_user.user_id,
        data=test_reward_data,
        notification_config=test_notification_config,
        db_session=db_session
    )
    
    # Verify notification result
    assert isinstance(result, Dict)
    assert "notification_id" in result
    assert "notification_type" in result
    
    # Verify notification metadata
    assert result["notification_type"] == "REWARD_NOTIFICATION"
    assert result["notification_config"]["type"] == "email"
    
    # Verify notification content
    assert "notification_content" in result
    assert isinstance(result["notification_content"], Dict)
    assert "title" in result["notification_content"]
    assert "message" in result["notification_content"]
    assert "data" in result["notification_content"]
    
    # Verify database entry
    db_notification = db_session.query(Notification).filter_by(
        user_id=test_user.user_id,
        notification_type="REWARD_NOTIFICATION"
    ).first()
    assert db_notification is not None
    assert db_notification.is_success is True
    assert db_notification.error is None

def test_generate_activity_notification(db_session, test_user, test_activity_data, test_notification_config):
    """Test activity notification generation"""
    # Generate activity notification
    result = generate_activity_notification(
        user_id=test_user.user_id,
        data=test_activity_data,
        notification_config=test_notification_config,
        db_session=db_session
    )
    
    # Verify notification result
    assert isinstance(result, Dict)
    assert "notification_id" in result
    assert "notification_type" in result
    
    # Verify notification metadata
    assert result["notification_type"] == "ACTIVITY_NOTIFICATION"
    assert result["notification_config"]["type"] == "email"
    
    # Verify notification content
    assert "notification_content" in result
    assert isinstance(result["notification_content"], Dict)
    assert "title" in result["notification_content"]
    assert "message" in result["notification_content"]
    assert "data" in result["notification_content"]
    
    # Verify database entry
    db_notification = db_session.query(Notification).filter_by(
        user_id=test_user.user_id,
        notification_type="ACTIVITY_NOTIFICATION"
    ).first()
    assert db_notification is not None
    assert db_notification.is_success is True
    assert db_notification.error is None

def test_generate_analytics_notification(db_session, test_user, test_analytics_data, test_notification_config):
    """Test analytics notification generation"""
    # Generate analytics notification
    result = generate_analytics_notification(
        user_id=test_user.user_id,
        data=test_analytics_data,
        notification_config=test_notification_config,
        db_session=db_session
    )
    
    # Verify notification result
    assert isinstance(result, Dict)
    assert "notification_id" in result
    assert "notification_type" in result
    
    # Verify notification metadata
    assert result["notification_type"] == "ANALYTICS_NOTIFICATION"
    assert result["notification_config"]["type"] == "email"
    
    # Verify notification content
    assert "notification_content" in result
    assert isinstance(result["notification_content"], Dict)
    assert "title" in result["notification_content"]
    assert "message" in result["notification_content"]
    assert "data" in result["notification_content"]
    
    # Verify database entry
    db_notification = db_session.query(Notification).filter_by(
        user_id=test_user.user_id,
        notification_type="ANALYTICS_NOTIFICATION"
    ).first()
    assert db_notification is not None
    assert db_notification.is_success is True
    assert db_notification.error is None

def test_customize_notification(db_session, test_user, test_performance_data, test_notification_config):
    """Test notification customization"""
    # Customize notification
    result = customize_notification(
        user_id=test_user.user_id,
        data=test_performance_data,
        notification_config=test_notification_config,
        db_session=db_session
    )
    
    # Verify customization result
    assert isinstance(result, Dict)
    assert "notification_id" in result
    assert "notification_type" in result
    
    # Verify notification metadata
    assert result["notification_type"] == "CUSTOMIZED_NOTIFICATION"
    assert result["notification_config"]["type"] == "email"
    
    # Verify notification content
    assert "notification_content" in result
    assert isinstance(result["notification_content"], Dict)
    assert "title" in result["notification_content"]
    assert "message" in result["notification_content"]
    assert "data" in result["notification_content"]
    
    # Verify database entry
    db_notification = db_session.query(Notification).filter_by(
        user_id=test_user.user_id,
        notification_type="CUSTOMIZED_NOTIFICATION"
    ).first()
    assert db_notification is not None
    assert db_notification.is_success is True
    assert db_notification.error is None

def test_deliver_notification(db_session, test_user, test_performance_data, test_notification_config):
    """Test notification delivery"""
    # Deliver notification
    result = deliver_notification(
        user_id=test_user.user_id,
        data=test_performance_data,
        notification_config=test_notification_config,
        db_session=db_session
    )
    
    # Verify delivery result
    assert isinstance(result, Dict)
    assert "delivery_id" in result
    assert "delivery_status" in result
    
    # Verify delivery metadata
    assert result["delivery_type"] == "email"
    assert result["delivery_status"] in ["SUCCESS", "FAILED"]
    
    # Verify delivery details
    assert "delivery_details" in result
    assert isinstance(result["delivery_details"], Dict)
    assert "recipients" in result["delivery_details"]
    assert "timestamp" in result["delivery_details"]
    
    # Verify database entry
    db_delivery = db_session.query(NotificationDelivery).filter_by(
        user_id=test_user.user_id,
        delivery_type="EMAIL"
    ).first()
    assert db_delivery is not None
    assert db_delivery.is_success is True
    assert db_delivery.error is None

def test_notification_error_handling(db_session, test_user):
    """Test notification error handling"""
    # Invalid user ID
    with pytest.raises(NotificationError) as excinfo:
        generate_performance_notification(
            user_id=None,
            data=pd.DataFrame(),
            notification_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(NotificationError) as excinfo:
        generate_risk_notification(
            user_id=test_user.user_id,
            data=None,
            notification_config={},
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Empty data
    with pytest.raises(NotificationError) as excinfo:
        generate_reward_notification(
            user_id=test_user.user_id,
            data=pd.DataFrame(),
            notification_config={},
            db_session=db_session
        )
    assert "Empty data" in str(excinfo.value)
    
    # Invalid notification type
    with pytest.raises(NotificationError) as excinfo:
        generate_activity_notification(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            notification_config={"type": "invalid_type"},
            db_session=db_session
        )
    assert "Invalid notification type" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBNotificationError) as excinfo:
        generate_performance_notification(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            notification_config={"type": "email"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value)

def test_create_notification(db_session, test_user, test_notification_config):
    """Test notification creation"""
    # Create notification
    result = create_notification(
        notification_type="alert",
        priority="high",
        category="performance",
        message="Performance threshold exceeded",
        user_id=test_user.user_id,
        config=test_notification_config["notification_types"],
        db_session=db_session
    )
    
    # Verify notification result
    assert isinstance(result, Dict)
    assert "notification_id" in result
    assert "timestamp" in result
    assert "notification" in result
    
    # Verify notification details
    notification = result["notification"]
    assert "type" in notification
    assert "priority" in notification
    assert "category" in notification
    assert "message" in notification
    assert "status" in notification
    
    # Verify notification type and priority
    assert notification["type"] in test_notification_config["notification_types"]["types"]
    assert notification["priority"] in test_notification_config["notification_types"]["priorities"]
    assert notification["category"] in test_notification_config["notification_types"]["categories"]
    
    # Verify notification record
    notification_record = db_session.query(NotificationRecord).filter_by(
        notification_id=result["notification_id"]
    ).first()
    assert notification_record is not None
    assert notification_record.status == "CREATED"
    assert notification_record.error is None

def test_configure_notification(db_session, test_user, test_notification_config):
    """Test notification configuration"""
    # Configure notification
    result = configure_notification(
        user_id=test_user.user_id,
        config=test_notification_config,
        message="Configure notification settings",
        db_session=db_session
    )
    
    # Verify configuration result
    assert isinstance(result, Dict)
    assert "config_id" in result
    assert "timestamp" in result
    assert "config" in result
    
    # Verify configuration details
    config = result["config"]
    assert "notification_types" in config
    assert "event_triggers" in config
    assert "message_formatting" in config
    assert "delivery_channels" in config
    assert "history_tracking" in config
    
    # Verify configuration settings
    assert config["notification_types"]["enabled"] is True
    assert config["event_triggers"]["enabled"] is True
    assert config["message_formatting"]["enabled"] is True
    assert config["delivery_channels"]["enabled"] is True
    assert config["history_tracking"]["enabled"] is True
    
    # Verify configuration record
    config_record = db_session.query(NotificationConfig).filter_by(
        config_id=result["config_id"]
    ).first()
    assert config_record is not None
    assert config_record.status == "ACTIVE"
    assert config_record.error is None

def test_trigger_notification(db_session, test_user, test_data, test_notification_config):
    """Test notification triggering"""
    # Create trigger condition
    trigger_condition = {
        "type": "threshold",
        "metric": "trading_performance",
        "operator": "gt",
        "value": 0.9,
        "message": "High trading performance detected"
    }
    
    # Trigger notification
    result = trigger_notification(
        data=test_data,
        trigger_config=test_notification_config["event_triggers"],
        condition=trigger_condition,
        user_id=test_user.user_id,
        message="Trigger performance notification",
        db_session=db_session
    )
    
    # Verify trigger result
    assert isinstance(result, Dict)
    assert "trigger_id" in result
    assert "timestamp" in result
    assert "trigger" in result
    
    # Verify trigger details
    trigger = result["trigger"]
    assert "type" in trigger
    assert "condition" in trigger
    assert "status" in trigger
    assert "notifications" in trigger
    
    # Verify trigger type and condition
    assert trigger["type"] in test_notification_config["event_triggers"]["trigger_types"]
    assert trigger["condition"]["type"] == "threshold"
    assert trigger["condition"]["operator"] in test_notification_config["event_triggers"]["conditions"]["threshold"]["operators"]
    
    # Verify notification record
    notification_record = db_session.query(NotificationRecord).filter_by(
        trigger_id=result["trigger_id"]
    ).first()
    assert notification_record is not None
    assert notification_record.status == "TRIGGERED"
    assert notification_record.error is None

def test_format_message(db_session, test_user, test_notification_config):
    """Test message formatting"""
    # Create message data
    message_data = {
        "severity": "HIGH",
        "message": "Critical system alert",
        "timestamp": datetime.utcnow().isoformat(),
        "user": test_user.username,
        "details": "System performance degraded"
    }
    
    # Format message
    result = format_message(
        template_type="alert",
        message_data=message_data,
        format_config=test_notification_config["message_formatting"],
        user_id=test_user.user_id,
        message="Format alert message",
        db_session=db_session
    )
    
    # Verify format result
    assert isinstance(result, Dict)
    assert "format_id" in result
    assert "timestamp" in result
    assert "message" in result
    
    # Verify message details
    formatted_message = result["message"]
    assert "content" in formatted_message
    assert "format" in formatted_message
    assert "length" in formatted_message
    
    # Verify message format
    assert formatted_message["format"] in test_notification_config["message_formatting"]["supported_formats"]
    assert len(formatted_message["content"]) <= test_notification_config["message_formatting"]["max_length"]
    
    # Verify notification record
    notification_record = db_session.query(NotificationRecord).filter_by(
        format_id=result["format_id"]
    ).first()
    assert notification_record is not None
    assert notification_record.status == "FORMATTED"
    assert notification_record.error is None

def test_send_notification(db_session, test_user, test_notification_config):
    """Test notification sending"""
    # Create notification first
    notification = create_notification(
        notification_type="alert",
        priority="high",
        category="performance",
        message="Performance threshold exceeded",
        user_id=test_user.user_id,
        config=test_notification_config["notification_types"],
        db_session=db_session
    )
    
    # Send notification
    result = send_notification(
        notification_id=notification["notification_id"],
        channels=["email", "in_app"],
        delivery_config=test_notification_config["delivery_channels"],
        user_id=test_user.user_id,
        message="Send performance alert",
        db_session=db_session
    )
    
    # Verify send result
    assert isinstance(result, Dict)
    assert "delivery_id" in result
    assert "timestamp" in result
    assert "delivery" in result
    
    # Verify delivery details
    delivery = result["delivery"]
    assert "channels" in delivery
    assert "status" in delivery
    assert "attempts" in delivery
    
    # Verify delivery channels
    for channel in delivery["channels"]:
        assert channel in test_notification_config["delivery_channels"]["channels"]
        assert "status" in delivery["channels"][channel]
        assert "timestamp" in delivery["channels"][channel]
    
    # Verify notification record
    notification_record = db_session.query(NotificationRecord).filter_by(
        delivery_id=result["delivery_id"]
    ).first()
    assert notification_record is not None
    assert notification_record.status == "DELIVERED"
    assert notification_record.error is None

def test_track_notification_history(db_session, test_user, test_notification_config):
    """Test notification history tracking"""
    # Create notification first
    notification = create_notification(
        notification_type="alert",
        priority="high",
        category="performance",
        message="Performance threshold exceeded",
        user_id=test_user.user_id,
        config=test_notification_config["notification_types"],
        db_session=db_session
    )
    
    # Track notification history
    result = track_notification_history(
        notification_id=notification["notification_id"],
        history_config=test_notification_config["history_tracking"],
        user_id=test_user.user_id,
        message="Track notification history",
        db_session=db_session
    )
    
    # Verify history result
    assert isinstance(result, Dict)
    assert "history_id" in result
    assert "timestamp" in result
    assert "history" in result
    
    # Verify history details
    history = result["history"]
    assert "notification_id" in history
    assert "metadata" in history
    assert "delivery" in history
    assert "responses" in history
    
    # Verify history record
    history_record = db_session.query(NotificationHistory).filter_by(
        history_id=result["history_id"]
    ).first()
    assert history_record is not None
    assert history_record.status == "TRACKED"
    assert history_record.error is None

def test_get_notification_info(db_session, test_user, test_notification_config):
    """Test notification information retrieval"""
    # Create notification first
    notification = create_notification(
        notification_type="alert",
        priority="high",
        category="performance",
        message="Performance threshold exceeded",
        user_id=test_user.user_id,
        config=test_notification_config["notification_types"],
        db_session=db_session
    )
    
    # Get notification info
    result = get_notification_info(
        notification_id=notification["notification_id"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "notification_id" in result
    assert "timestamp" in result
    assert "info" in result
    
    # Verify notification info
    info = result["info"]
    assert "type" in info
    assert "status" in info
    assert "metadata" in info
    assert "history" in info
    
    # Verify notification record
    notification_record = db_session.query(NotificationRecord).filter_by(
        notification_id=result["notification_id"]
    ).first()
    assert notification_record is not None
    assert notification_record.status == "RETRIEVED"
    assert notification_record.error is None

def test_list_notifications(db_session, test_user, test_notification_config):
    """Test notification listing"""
    # Create multiple notifications
    for i in range(5):
        create_notification(
            notification_type="alert",
            priority="high",
            category="performance",
            message=f"Test notification {i+1}",
            user_id=test_user.user_id,
            config=test_notification_config["notification_types"],
            db_session=db_session
        )
    
    # List notifications
    result = list_notifications(
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify listing result
    assert isinstance(result, Dict)
    assert "timestamp" in result
    assert "notifications" in result
    
    # Verify notifications list
    notifications = result["notifications"]
    assert isinstance(notifications, List)
    assert len(notifications) == 5
    
    # Verify notification details
    for notification in notifications:
        assert "notification_id" in notification
        assert "timestamp" in notification
        assert "type" in notification
        assert "status" in notification

def test_delete_notification(db_session, test_user, test_notification_config):
    """Test notification deletion"""
    # Create notification first
    notification = create_notification(
        notification_type="alert",
        priority="high",
        category="performance",
        message="Test notification",
        user_id=test_user.user_id,
        config=test_notification_config["notification_types"],
        db_session=db_session
    )
    
    # Delete notification
    result = delete_notification(
        notification_id=notification["notification_id"],
        user_id=test_user.user_id,
        message="Delete test notification",
        db_session=db_session
    )
    
    # Verify deletion result
    assert isinstance(result, Dict)
    assert "deletion_id" in result
    assert "timestamp" in result
    assert "status" in result
    
    # Verify status
    assert result["status"] == "DELETED"
    
    # Verify notification record
    notification_record = db_session.query(NotificationRecord).filter_by(
        notification_id=notification["notification_id"]
    ).first()
    assert notification_record is not None
    assert notification_record.status == "DELETED"
    assert notification_record.error is None

def test_notification_error_handling(db_session, test_user):
    """Test notification error handling"""
    # Invalid notification configuration
    with pytest.raises(NotificationError) as excinfo:
        create_notification(
            notification_type="invalid_type",
            priority="high",
            category="performance",
            message="Test",
            user_id=test_user.user_id,
            config={},
            db_session=db_session
        )
    assert "Invalid notification type" in str(excinfo.value)
    
    # Invalid trigger configuration
    with pytest.raises(NotificationError) as excinfo:
        trigger_notification(
            data=pd.DataFrame(),
            trigger_config={},
            condition={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid trigger configuration" in str(excinfo.value)
    
    # Invalid message format
    with pytest.raises(NotificationError) as excinfo:
        format_message(
            template_type="invalid_template",
            message_data={},
            format_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid message template" in str(excinfo.value)
    
    # Invalid delivery channel
    with pytest.raises(NotificationError) as excinfo:
        send_notification(
            notification_id="invalid_id",
            channels=["invalid_channel"],
            delivery_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid delivery channel" in str(excinfo.value)
    
    # Invalid notification ID
    with pytest.raises(NotificationError) as excinfo:
        track_notification_history(
            notification_id="invalid_id",
            history_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid notification ID" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBNotificationError) as excinfo:
        create_notification(
            notification_type="alert",
            priority="high",
            category="performance",
            message="Test",
            user_id=test_user.user_id,
            config={},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 