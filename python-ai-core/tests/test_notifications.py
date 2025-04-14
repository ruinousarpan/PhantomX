import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from decimal import Decimal

from core.notifications import (
    generate_performance_alert,
    generate_risk_alert,
    generate_reward_alert,
    send_notification,
    update_notification_preferences,
    get_notification_history,
    NotificationType,
    NotificationPriority,
    NotificationChannel,
    NotificationStatus
)
from database.models import User, Activity, Performance, Risk, Reward, Notification, NotificationPreference
from database.exceptions import NotificationError

@pytest.fixture
def test_user(db_session):
    """Create test user"""
    user = User(
        user_id="test_user",
        username="testuser",
        email="test@example.com"
    )
    db_session.add(user)
    db_session.commit()
    return user

@pytest.fixture
def test_notification_preferences(db_session, test_user):
    """Create test notification preferences"""
    preferences = NotificationPreference(
        user_id=test_user.user_id,
        notification_type=NotificationType.PERFORMANCE,
        channel=NotificationChannel.EMAIL,
        enabled=True,
        priority_threshold=NotificationPriority.MEDIUM
    )
    db_session.add(preferences)
    db_session.commit()
    return preferences

def test_generate_performance_alert(db_session, test_user):
    """Test performance alert generation"""
    # Create test performance metrics
    performance_metrics = [
        Performance(
            performance_id=f"performance_{i}",
            activity_id=f"activity_{i}",
            performance_score=0.2 * (i + 1),  # 0.2, 0.4, 0.6, 0.8, 1.0
            performance_level="LOW" if i < 2 else "MEDIUM" if i < 4 else "HIGH",
            performance_factors=["HIGH_WIN_RATE"] if i > 2 else [],
            timestamp=datetime.utcnow() - timedelta(days=i),
            type="TRADING"
        )
        for i in range(5)
    ]
    
    db_session.add_all(performance_metrics)
    db_session.commit()
    
    # Generate alert
    alert = generate_performance_alert(
        user_id=test_user.user_id,
        performance_metrics=performance_metrics,
        db_session=db_session
    )
    
    # Verify alert
    assert isinstance(alert, Dict)
    assert "alert_id" in alert
    assert "alert_type" in alert
    assert "alert_priority" in alert
    assert "alert_content" in alert
    assert "alert_metadata" in alert
    
    # Verify alert content
    assert "summary" in alert["alert_content"]
    assert "performance_level" in alert["alert_content"]
    assert "performance_score" in alert["alert_content"]
    assert "performance_factors" in alert["alert_content"]
    assert "recommendations" in alert["alert_content"]

def test_generate_risk_alert(db_session, test_user):
    """Test risk alert generation"""
    # Create test risk assessments
    risk_assessments = [
        Risk(
            risk_id=f"risk_{i}",
            activity_id=f"activity_{i}",
            risk_score=0.2 * (i + 1),  # 0.2, 0.4, 0.6, 0.8, 1.0
            risk_level="LOW" if i < 2 else "MEDIUM" if i < 4 else "HIGH",
            risk_factors=["HIGH_LEVERAGE"] if i > 2 else [],
            timestamp=datetime.utcnow() - timedelta(days=i),
            type="TRADING"
        )
        for i in range(5)
    ]
    
    db_session.add_all(risk_assessments)
    db_session.commit()
    
    # Generate alert
    alert = generate_risk_alert(
        user_id=test_user.user_id,
        risk_assessments=risk_assessments,
        db_session=db_session
    )
    
    # Verify alert
    assert isinstance(alert, Dict)
    assert "alert_id" in alert
    assert "alert_type" in alert
    assert "alert_priority" in alert
    assert "alert_content" in alert
    assert "alert_metadata" in alert
    
    # Verify alert content
    assert "summary" in alert["alert_content"]
    assert "risk_level" in alert["alert_content"]
    assert "risk_score" in alert["alert_content"]
    assert "risk_factors" in alert["alert_content"]
    assert "mitigation_steps" in alert["alert_content"]

def test_generate_reward_alert(db_session, test_user):
    """Test reward alert generation"""
    # Create test rewards
    rewards = [
        Reward(
            reward_id=f"reward_{i}",
            activity_id=f"activity_{i}",
            reward_amount=0.1 * (i + 1),  # 0.1, 0.2, 0.3, 0.4, 0.5
            reward_type="BLOCK_REWARD" if i % 2 == 0 else "STAKING_REWARD",
            timestamp=datetime.utcnow() - timedelta(days=i),
            status="CONFIRMED"
        )
        for i in range(5)
    ]
    
    db_session.add_all(rewards)
    db_session.commit()
    
    # Generate alert
    alert = generate_reward_alert(
        user_id=test_user.user_id,
        rewards=rewards,
        db_session=db_session
    )
    
    # Verify alert
    assert isinstance(alert, Dict)
    assert "alert_id" in alert
    assert "alert_type" in alert
    assert "alert_priority" in alert
    assert "alert_content" in alert
    assert "alert_metadata" in alert
    
    # Verify alert content
    assert "summary" in alert["alert_content"]
    assert "reward_amount" in alert["alert_content"]
    assert "reward_type" in alert["alert_content"]
    assert "reward_status" in alert["alert_content"]
    assert "total_rewards" in alert["alert_content"]

def test_send_notification(db_session, test_user, test_notification_preferences):
    """Test notification sending"""
    # Create test notification
    notification = Notification(
        notification_id="test_notification",
        user_id=test_user.user_id,
        notification_type=NotificationType.PERFORMANCE,
        priority=NotificationPriority.HIGH,
        content={
            "summary": "Test notification",
            "details": "This is a test notification"
        },
        status=NotificationStatus.PENDING,
        created_at=datetime.utcnow()
    )
    
    db_session.add(notification)
    db_session.commit()
    
    # Send notification
    result = send_notification(
        notification_id=notification.notification_id,
        db_session=db_session
    )
    
    # Verify result
    assert isinstance(result, Dict)
    assert "notification_id" in result
    assert "status" in result
    assert "sent_at" in result
    assert "delivery_channel" in result
    
    # Verify notification status updated
    updated_notification = db_session.query(Notification).filter_by(
        notification_id=notification.notification_id
    ).first()
    assert updated_notification.status == NotificationStatus.SENT

def test_update_notification_preferences(db_session, test_user):
    """Test notification preferences update"""
    # Create test preferences
    preferences = NotificationPreference(
        user_id=test_user.user_id,
        notification_type=NotificationType.PERFORMANCE,
        channel=NotificationChannel.EMAIL,
        enabled=True,
        priority_threshold=NotificationPriority.MEDIUM
    )
    
    db_session.add(preferences)
    db_session.commit()
    
    # Update preferences
    updated_preferences = update_notification_preferences(
        user_id=test_user.user_id,
        notification_type=NotificationType.PERFORMANCE,
        channel=NotificationChannel.PUSH,
        enabled=False,
        priority_threshold=NotificationPriority.HIGH,
        db_session=db_session
    )
    
    # Verify updated preferences
    assert isinstance(updated_preferences, Dict)
    assert "user_id" in updated_preferences
    assert "notification_type" in updated_preferences
    assert "channel" in updated_preferences
    assert "enabled" in updated_preferences
    assert "priority_threshold" in updated_preferences
    
    # Verify values updated
    assert updated_preferences["channel"] == NotificationChannel.PUSH
    assert updated_preferences["enabled"] is False
    assert updated_preferences["priority_threshold"] == NotificationPriority.HIGH

def test_get_notification_history(db_session, test_user):
    """Test notification history retrieval"""
    # Create test notifications
    notifications = [
        Notification(
            notification_id=f"notification_{i}",
            user_id=test_user.user_id,
            notification_type=NotificationType.PERFORMANCE if i % 2 == 0 else NotificationType.RISK,
            priority=NotificationPriority.HIGH if i < 2 else NotificationPriority.MEDIUM,
            content={
                "summary": f"Test notification {i}",
                "details": f"This is test notification {i}"
            },
            status=NotificationStatus.SENT if i < 3 else NotificationStatus.PENDING,
            created_at=datetime.utcnow() - timedelta(days=i)
        )
        for i in range(5)
    ]
    
    db_session.add_all(notifications)
    db_session.commit()
    
    # Get notification history
    history = get_notification_history(
        user_id=test_user.user_id,
        start_time=datetime.utcnow() - timedelta(days=5),
        end_time=datetime.utcnow(),
        db_session=db_session
    )
    
    # Verify history
    assert isinstance(history, Dict)
    assert "notifications" in history
    assert "summary" in history
    
    # Verify notifications
    assert len(history["notifications"]) > 0
    for notification in history["notifications"]:
        assert "notification_id" in notification
        assert "notification_type" in notification
        assert "priority" in notification
        assert "content" in notification
        assert "status" in notification
        assert "created_at" in notification

def test_notification_error_handling():
    """Test notification error handling"""
    # Invalid notification type
    with pytest.raises(NotificationError) as excinfo:
        generate_performance_alert(
            user_id="test_user",
            performance_metrics=[],
            db_session=None
        )
    assert "Invalid performance metrics" in str(excinfo.value)
    
    # Invalid notification priority
    with pytest.raises(NotificationError) as excinfo:
        update_notification_preferences(
            user_id="test_user",
            notification_type=NotificationType.PERFORMANCE,
            channel=NotificationChannel.EMAIL,
            enabled=True,
            priority_threshold="INVALID_PRIORITY",
            db_session=None
        )
    assert "Invalid priority threshold" in str(excinfo.value)
    
    # Invalid notification channel
    with pytest.raises(NotificationError) as excinfo:
        update_notification_preferences(
            user_id="test_user",
            notification_type=NotificationType.PERFORMANCE,
            channel="INVALID_CHANNEL",
            enabled=True,
            priority_threshold=NotificationPriority.MEDIUM,
            db_session=None
        )
    assert "Invalid notification channel" in str(excinfo.value)
    
    # Invalid notification status
    with pytest.raises(NotificationError) as excinfo:
        send_notification(
            notification_id="test_notification",
            db_session=None
        )
    assert "Notification not found" in str(excinfo.value)
    
    # Invalid time range
    with pytest.raises(NotificationError) as excinfo:
        get_notification_history(
            user_id="test_user",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() - timedelta(days=1),
            db_session=None
        )
    assert "Invalid time range" in str(excinfo.value) 