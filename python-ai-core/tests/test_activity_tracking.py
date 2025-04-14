import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from decimal import Decimal

from core.activity_tracking import (
    log_mining_activity,
    log_staking_activity,
    log_trading_activity,
    get_activity_history,
    get_activity_summary,
    analyze_activity_patterns,
    monitor_activity_health,
    ActivityType,
    ActivityStatus,
    ActivityHealth
)
from database.models import User, Activity, ActivityMetrics, ActivityHealth as ActivityHealthModel
from database.exceptions import ActivityTrackingError

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
def test_mining_activity_data():
    """Create test mining activity data"""
    return {
        "hash_rate": 95.5,
        "power_usage": 1450.0,
        "temperature": 75.0,
        "uptime": 0.98,
        "efficiency": 0.85,
        "block_rewards": 0.5,
        "network_difficulty": 45.0,
        "profitability": 0.25
    }

@pytest.fixture
def test_staking_activity_data():
    """Create test staking activity data"""
    return {
        "validator_uptime": 0.99,
        "missed_blocks": 2,
        "reward_rate": 0.12,
        "peer_count": 50,
        "network_participation": 0.85,
        "slashing_events": 0,
        "stake_amount": 1000.0,
        "validator_count": 100
    }

@pytest.fixture
def test_trading_activity_data():
    """Create test trading activity data"""
    return {
        "position_size": 50000.0,
        "leverage_ratio": 2.0,
        "win_rate": 0.6,
        "profit_loss": 500.0,
        "drawdown": 0.15,
        "sharpe_ratio": 1.5,
        "volume": 100000.0,
        "execution_quality": 0.9
    }

def test_log_mining_activity(db_session, test_user, test_mining_activity_data):
    """Test mining activity logging"""
    # Log mining activity
    activity = log_mining_activity(
        user_id=test_user.user_id,
        hash_rate=test_mining_activity_data["hash_rate"],
        power_usage=test_mining_activity_data["power_usage"],
        temperature=test_mining_activity_data["temperature"],
        uptime=test_mining_activity_data["uptime"],
        efficiency=test_mining_activity_data["efficiency"],
        block_rewards=test_mining_activity_data["block_rewards"],
        network_difficulty=test_mining_activity_data["network_difficulty"],
        profitability=test_mining_activity_data["profitability"],
        db_session=db_session
    )
    
    # Verify activity
    assert isinstance(activity, Dict)
    assert "activity_id" in activity
    assert "user_id" in activity
    assert "activity_type" in activity
    assert "status" in activity
    assert "start_time" in activity
    assert "end_time" in activity
    assert "metrics" in activity
    
    # Verify activity type and status
    assert activity["activity_type"] == ActivityType.MINING.value
    assert activity["status"] == ActivityStatus.COMPLETED.value
    
    # Verify metrics
    assert activity["metrics"]["hash_rate"] == test_mining_activity_data["hash_rate"]
    assert activity["metrics"]["power_usage"] == test_mining_activity_data["power_usage"]
    assert activity["metrics"]["temperature"] == test_mining_activity_data["temperature"]
    assert activity["metrics"]["uptime"] == test_mining_activity_data["uptime"]
    assert activity["metrics"]["efficiency"] == test_mining_activity_data["efficiency"]
    assert activity["metrics"]["block_rewards"] == test_mining_activity_data["block_rewards"]
    assert activity["metrics"]["network_difficulty"] == test_mining_activity_data["network_difficulty"]
    assert activity["metrics"]["profitability"] == test_mining_activity_data["profitability"]
    
    # Verify database entry
    db_activity = db_session.query(Activity).filter_by(activity_id=activity["activity_id"]).first()
    assert db_activity is not None
    assert db_activity.user_id == test_user.user_id
    assert db_activity.activity_type == ActivityType.MINING.value
    assert db_activity.status == ActivityStatus.COMPLETED.value
    
    # Verify metrics in database
    db_metrics = db_session.query(ActivityMetrics).filter_by(activity_id=activity["activity_id"]).first()
    assert db_metrics is not None
    assert db_metrics.hash_rate == test_mining_activity_data["hash_rate"]
    assert db_metrics.power_usage == test_mining_activity_data["power_usage"]
    assert db_metrics.temperature == test_mining_activity_data["temperature"]
    assert db_metrics.uptime == test_mining_activity_data["uptime"]
    assert db_metrics.efficiency == test_mining_activity_data["efficiency"]
    assert db_metrics.block_rewards == test_mining_activity_data["block_rewards"]
    assert db_metrics.network_difficulty == test_mining_activity_data["network_difficulty"]
    assert db_metrics.profitability == test_mining_activity_data["profitability"]

def test_log_staking_activity(db_session, test_user, test_staking_activity_data):
    """Test staking activity logging"""
    # Log staking activity
    activity = log_staking_activity(
        user_id=test_user.user_id,
        validator_uptime=test_staking_activity_data["validator_uptime"],
        missed_blocks=test_staking_activity_data["missed_blocks"],
        reward_rate=test_staking_activity_data["reward_rate"],
        peer_count=test_staking_activity_data["peer_count"],
        network_participation=test_staking_activity_data["network_participation"],
        slashing_events=test_staking_activity_data["slashing_events"],
        stake_amount=test_staking_activity_data["stake_amount"],
        validator_count=test_staking_activity_data["validator_count"],
        db_session=db_session
    )
    
    # Verify activity
    assert isinstance(activity, Dict)
    assert "activity_id" in activity
    assert "user_id" in activity
    assert "activity_type" in activity
    assert "status" in activity
    assert "start_time" in activity
    assert "end_time" in activity
    assert "metrics" in activity
    
    # Verify activity type and status
    assert activity["activity_type"] == ActivityType.STAKING.value
    assert activity["status"] == ActivityStatus.COMPLETED.value
    
    # Verify metrics
    assert activity["metrics"]["validator_uptime"] == test_staking_activity_data["validator_uptime"]
    assert activity["metrics"]["missed_blocks"] == test_staking_activity_data["missed_blocks"]
    assert activity["metrics"]["reward_rate"] == test_staking_activity_data["reward_rate"]
    assert activity["metrics"]["peer_count"] == test_staking_activity_data["peer_count"]
    assert activity["metrics"]["network_participation"] == test_staking_activity_data["network_participation"]
    assert activity["metrics"]["slashing_events"] == test_staking_activity_data["slashing_events"]
    assert activity["metrics"]["stake_amount"] == test_staking_activity_data["stake_amount"]
    assert activity["metrics"]["validator_count"] == test_staking_activity_data["validator_count"]
    
    # Verify database entry
    db_activity = db_session.query(Activity).filter_by(activity_id=activity["activity_id"]).first()
    assert db_activity is not None
    assert db_activity.user_id == test_user.user_id
    assert db_activity.activity_type == ActivityType.STAKING.value
    assert db_activity.status == ActivityStatus.COMPLETED.value
    
    # Verify metrics in database
    db_metrics = db_session.query(ActivityMetrics).filter_by(activity_id=activity["activity_id"]).first()
    assert db_metrics is not None
    assert db_metrics.validator_uptime == test_staking_activity_data["validator_uptime"]
    assert db_metrics.missed_blocks == test_staking_activity_data["missed_blocks"]
    assert db_metrics.reward_rate == test_staking_activity_data["reward_rate"]
    assert db_metrics.peer_count == test_staking_activity_data["peer_count"]
    assert db_metrics.network_participation == test_staking_activity_data["network_participation"]
    assert db_metrics.slashing_events == test_staking_activity_data["slashing_events"]
    assert db_metrics.stake_amount == test_staking_activity_data["stake_amount"]
    assert db_metrics.validator_count == test_staking_activity_data["validator_count"]

def test_log_trading_activity(db_session, test_user, test_trading_activity_data):
    """Test trading activity logging"""
    # Log trading activity
    activity = log_trading_activity(
        user_id=test_user.user_id,
        position_size=test_trading_activity_data["position_size"],
        leverage_ratio=test_trading_activity_data["leverage_ratio"],
        win_rate=test_trading_activity_data["win_rate"],
        profit_loss=test_trading_activity_data["profit_loss"],
        drawdown=test_trading_activity_data["drawdown"],
        sharpe_ratio=test_trading_activity_data["sharpe_ratio"],
        volume=test_trading_activity_data["volume"],
        execution_quality=test_trading_activity_data["execution_quality"],
        db_session=db_session
    )
    
    # Verify activity
    assert isinstance(activity, Dict)
    assert "activity_id" in activity
    assert "user_id" in activity
    assert "activity_type" in activity
    assert "status" in activity
    assert "start_time" in activity
    assert "end_time" in activity
    assert "metrics" in activity
    
    # Verify activity type and status
    assert activity["activity_type"] == ActivityType.TRADING.value
    assert activity["status"] == ActivityStatus.COMPLETED.value
    
    # Verify metrics
    assert activity["metrics"]["position_size"] == test_trading_activity_data["position_size"]
    assert activity["metrics"]["leverage_ratio"] == test_trading_activity_data["leverage_ratio"]
    assert activity["metrics"]["win_rate"] == test_trading_activity_data["win_rate"]
    assert activity["metrics"]["profit_loss"] == test_trading_activity_data["profit_loss"]
    assert activity["metrics"]["drawdown"] == test_trading_activity_data["drawdown"]
    assert activity["metrics"]["sharpe_ratio"] == test_trading_activity_data["sharpe_ratio"]
    assert activity["metrics"]["volume"] == test_trading_activity_data["volume"]
    assert activity["metrics"]["execution_quality"] == test_trading_activity_data["execution_quality"]
    
    # Verify database entry
    db_activity = db_session.query(Activity).filter_by(activity_id=activity["activity_id"]).first()
    assert db_activity is not None
    assert db_activity.user_id == test_user.user_id
    assert db_activity.activity_type == ActivityType.TRADING.value
    assert db_activity.status == ActivityStatus.COMPLETED.value
    
    # Verify metrics in database
    db_metrics = db_session.query(ActivityMetrics).filter_by(activity_id=activity["activity_id"]).first()
    assert db_metrics is not None
    assert db_metrics.position_size == test_trading_activity_data["position_size"]
    assert db_metrics.leverage_ratio == test_trading_activity_data["leverage_ratio"]
    assert db_metrics.win_rate == test_trading_activity_data["win_rate"]
    assert db_metrics.profit_loss == test_trading_activity_data["profit_loss"]
    assert db_metrics.drawdown == test_trading_activity_data["drawdown"]
    assert db_metrics.sharpe_ratio == test_trading_activity_data["sharpe_ratio"]
    assert db_metrics.volume == test_trading_activity_data["volume"]
    assert db_metrics.execution_quality == test_trading_activity_data["execution_quality"]

def test_get_activity_history(db_session, test_user):
    """Test activity history retrieval"""
    # Create test activities
    activities = []
    
    # Mining activity
    mining_activity = Activity(
        activity_id="mining_activity",
        user_id=test_user.user_id,
        activity_type=ActivityType.MINING.value,
        status=ActivityStatus.COMPLETED.value,
        start_time=datetime.utcnow() - timedelta(hours=2),
        end_time=datetime.utcnow() - timedelta(hours=1)
    )
    activities.append(mining_activity)
    
    # Staking activity
    staking_activity = Activity(
        activity_id="staking_activity",
        user_id=test_user.user_id,
        activity_type=ActivityType.STAKING.value,
        status=ActivityStatus.COMPLETED.value,
        start_time=datetime.utcnow() - timedelta(hours=1),
        end_time=datetime.utcnow()
    )
    activities.append(staking_activity)
    
    # Add to database
    db_session.add_all(activities)
    db_session.commit()
    
    # Get activity history
    history = get_activity_history(
        user_id=test_user.user_id,
        activity_type=None,
        start_time=datetime.utcnow() - timedelta(days=1),
        end_time=datetime.utcnow(),
        limit=10,
        offset=0,
        db_session=db_session
    )
    
    # Verify history
    assert isinstance(history, Dict)
    assert "activities" in history
    assert "total" in history
    assert "limit" in history
    assert "offset" in history
    
    # Verify activity list
    assert len(history["activities"]) >= 2  # At least mining and staking activities
    assert history["total"] >= 2
    assert history["limit"] == 10
    assert history["offset"] == 0
    
    # Verify activity details
    activity_ids = [activity["activity_id"] for activity in history["activities"]]
    assert "mining_activity" in activity_ids
    assert "staking_activity" in activity_ids
    
    # Get activity history by type
    mining_history = get_activity_history(
        user_id=test_user.user_id,
        activity_type=ActivityType.MINING,
        start_time=datetime.utcnow() - timedelta(days=1),
        end_time=datetime.utcnow(),
        limit=10,
        offset=0,
        db_session=db_session
    )
    assert len(mining_history["activities"]) >= 1  # At least mining activity
    assert all(activity["activity_type"] == ActivityType.MINING.value for activity in mining_history["activities"])

def test_get_activity_summary(db_session, test_user):
    """Test activity summary retrieval"""
    # Create test activities with metrics
    activities = []
    activity_metrics = []
    
    # Mining activity
    mining_activity = Activity(
        activity_id="mining_activity",
        user_id=test_user.user_id,
        activity_type=ActivityType.MINING.value,
        status=ActivityStatus.COMPLETED.value,
        start_time=datetime.utcnow() - timedelta(hours=2),
        end_time=datetime.utcnow() - timedelta(hours=1)
    )
    activities.append(mining_activity)
    
    mining_metrics = ActivityMetrics(
        activity_id="mining_activity",
        hash_rate=95.5,
        power_usage=1450.0,
        temperature=75.0,
        uptime=0.98,
        efficiency=0.85,
        block_rewards=0.5,
        network_difficulty=45.0,
        profitability=0.25
    )
    activity_metrics.append(mining_metrics)
    
    # Staking activity
    staking_activity = Activity(
        activity_id="staking_activity",
        user_id=test_user.user_id,
        activity_type=ActivityType.STAKING.value,
        status=ActivityStatus.COMPLETED.value,
        start_time=datetime.utcnow() - timedelta(hours=1),
        end_time=datetime.utcnow()
    )
    activities.append(staking_activity)
    
    staking_metrics = ActivityMetrics(
        activity_id="staking_activity",
        validator_uptime=0.99,
        missed_blocks=2,
        reward_rate=0.12,
        peer_count=50,
        network_participation=0.85,
        slashing_events=0,
        stake_amount=1000.0,
        validator_count=100
    )
    activity_metrics.append(staking_metrics)
    
    # Add to database
    db_session.add_all(activities)
    db_session.add_all(activity_metrics)
    db_session.commit()
    
    # Get activity summary
    summary = get_activity_summary(
        user_id=test_user.user_id,
        start_time=datetime.utcnow() - timedelta(days=1),
        end_time=datetime.utcnow(),
        db_session=db_session
    )
    
    # Verify summary
    assert isinstance(summary, Dict)
    assert "total_activities" in summary
    assert "activity_types" in summary
    assert "mining_summary" in summary
    assert "staking_summary" in summary
    assert "trading_summary" in summary
    
    # Verify activity counts
    assert summary["total_activities"] >= 2  # At least mining and staking activities
    assert "MINING" in summary["activity_types"]
    assert "STAKING" in summary["activity_types"]
    
    # Verify mining summary
    assert "count" in summary["mining_summary"]
    assert "total_hash_rate" in summary["mining_summary"]
    assert "total_power_usage" in summary["mining_summary"]
    assert "avg_temperature" in summary["mining_summary"]
    assert "avg_uptime" in summary["mining_summary"]
    assert "avg_efficiency" in summary["mining_summary"]
    assert "total_block_rewards" in summary["mining_summary"]
    assert "avg_network_difficulty" in summary["mining_summary"]
    assert "avg_profitability" in summary["mining_summary"]
    
    # Verify staking summary
    assert "count" in summary["staking_summary"]
    assert "avg_validator_uptime" in summary["staking_summary"]
    assert "total_missed_blocks" in summary["staking_summary"]
    assert "avg_reward_rate" in summary["staking_summary"]
    assert "avg_peer_count" in summary["staking_summary"]
    assert "avg_network_participation" in summary["staking_summary"]
    assert "total_slashing_events" in summary["staking_summary"]
    assert "total_stake_amount" in summary["staking_summary"]
    assert "avg_validator_count" in summary["staking_summary"]

def test_analyze_activity_patterns(db_session, test_user):
    """Test activity pattern analysis"""
    # Create test activities with timestamps
    activities = []
    
    # Create activities over the past week
    for i in range(7):
        # Mining activity
        mining_activity = Activity(
            activity_id=f"mining_activity_{i}",
            user_id=test_user.user_id,
            activity_type=ActivityType.MINING.value,
            status=ActivityStatus.COMPLETED.value,
            start_time=datetime.utcnow() - timedelta(days=i, hours=2),
            end_time=datetime.utcnow() - timedelta(days=i, hours=1)
        )
        activities.append(mining_activity)
        
        # Staking activity
        staking_activity = Activity(
            activity_id=f"staking_activity_{i}",
            user_id=test_user.user_id,
            activity_type=ActivityType.STAKING.value,
            status=ActivityStatus.COMPLETED.value,
            start_time=datetime.utcnow() - timedelta(days=i, hours=1),
            end_time=datetime.utcnow() - timedelta(days=i)
        )
        activities.append(staking_activity)
    
    # Add to database
    db_session.add_all(activities)
    db_session.commit()
    
    # Analyze activity patterns
    patterns = analyze_activity_patterns(
        user_id=test_user.user_id,
        start_time=datetime.utcnow() - timedelta(days=7),
        end_time=datetime.utcnow(),
        db_session=db_session
    )
    
    # Verify patterns
    assert isinstance(patterns, Dict)
    assert "activity_frequency" in patterns
    assert "activity_distribution" in patterns
    assert "peak_activity_times" in patterns
    assert "activity_correlations" in patterns
    assert "activity_trends" in patterns
    
    # Verify activity frequency
    assert "daily" in patterns["activity_frequency"]
    assert "weekly" in patterns["activity_frequency"]
    assert "monthly" in patterns["activity_frequency"]
    
    # Verify activity distribution
    assert "by_type" in patterns["activity_distribution"]
    assert "by_status" in patterns["activity_distribution"]
    assert "by_time_of_day" in patterns["activity_distribution"]
    assert "by_day_of_week" in patterns["activity_distribution"]
    
    # Verify peak activity times
    assert "mining" in patterns["peak_activity_times"]
    assert "staking" in patterns["peak_activity_times"]
    assert "trading" in patterns["peak_activity_times"]
    
    # Verify activity correlations
    assert "mining_staking_correlation" in patterns["activity_correlations"]
    assert "mining_trading_correlation" in patterns["activity_correlations"]
    assert "staking_trading_correlation" in patterns["activity_correlations"]
    
    # Verify activity trends
    assert "mining_trend" in patterns["activity_trends"]
    assert "staking_trend" in patterns["activity_trends"]
    assert "trading_trend" in patterns["activity_trends"]

def test_monitor_activity_health(db_session, test_user):
    """Test activity health monitoring"""
    # Create test activities with health metrics
    activities = []
    activity_health = []
    
    # Mining activity with good health
    mining_activity = Activity(
        activity_id="mining_activity",
        user_id=test_user.user_id,
        activity_type=ActivityType.MINING.value,
        status=ActivityStatus.COMPLETED.value,
        start_time=datetime.utcnow() - timedelta(hours=2),
        end_time=datetime.utcnow() - timedelta(hours=1)
    )
    activities.append(mining_activity)
    
    mining_health = ActivityHealthModel(
        activity_id="mining_activity",
        health_score=0.85,
        health_status=ActivityHealth.GOOD.value,
        health_factors=["HIGH_EFFICIENCY", "HIGH_UPTIME"],
        timestamp=datetime.utcnow()
    )
    activity_health.append(mining_health)
    
    # Staking activity with warning health
    staking_activity = Activity(
        activity_id="staking_activity",
        user_id=test_user.user_id,
        activity_type=ActivityType.STAKING.value,
        status=ActivityStatus.COMPLETED.value,
        start_time=datetime.utcnow() - timedelta(hours=1),
        end_time=datetime.utcnow()
    )
    activities.append(staking_activity)
    
    staking_health = ActivityHealthModel(
        activity_id="staking_activity",
        health_score=0.65,
        health_status=ActivityHealth.WARNING.value,
        health_factors=["HIGH_UPTIME"],
        timestamp=datetime.utcnow()
    )
    activity_health.append(staking_health)
    
    # Add to database
    db_session.add_all(activities)
    db_session.add_all(activity_health)
    db_session.commit()
    
    # Monitor activity health
    health_report = monitor_activity_health(
        user_id=test_user.user_id,
        start_time=datetime.utcnow() - timedelta(days=1),
        end_time=datetime.utcnow(),
        db_session=db_session
    )
    
    # Verify health report
    assert isinstance(health_report, Dict)
    assert "overall_health" in health_report
    assert "health_by_activity_type" in health_report
    assert "health_trends" in health_report
    assert "health_issues" in health_report
    assert "recommendations" in health_report
    
    # Verify overall health
    assert "health_score" in health_report["overall_health"]
    assert "health_status" in health_report["overall_health"]
    assert "health_factors" in health_report["overall_health"]
    
    # Verify health by activity type
    assert "mining" in health_report["health_by_activity_type"]
    assert "staking" in health_report["health_by_activity_type"]
    assert "trading" in health_report["health_by_activity_type"]
    
    # Verify mining health
    assert "health_score" in health_report["health_by_activity_type"]["mining"]
    assert "health_status" in health_report["health_by_activity_type"]["mining"]
    assert "health_factors" in health_report["health_by_activity_type"]["mining"]
    assert health_report["health_by_activity_type"]["mining"]["health_score"] == 0.85
    assert health_report["health_by_activity_type"]["mining"]["health_status"] == ActivityHealth.GOOD.value
    
    # Verify staking health
    assert "health_score" in health_report["health_by_activity_type"]["staking"]
    assert "health_status" in health_report["health_by_activity_type"]["staking"]
    assert "health_factors" in health_report["health_by_activity_type"]["staking"]
    assert health_report["health_by_activity_type"]["staking"]["health_score"] == 0.65
    assert health_report["health_by_activity_type"]["staking"]["health_status"] == ActivityHealth.WARNING.value
    
    # Verify health trends
    assert "mining_trend" in health_report["health_trends"]
    assert "staking_trend" in health_report["health_trends"]
    assert "trading_trend" in health_report["health_trends"]
    
    # Verify health issues
    assert len(health_report["health_issues"]) >= 1  # At least one issue for staking
    
    # Verify recommendations
    assert len(health_report["recommendations"]) >= 1  # At least one recommendation

def test_activity_tracking_error_handling():
    """Test activity tracking error handling"""
    # Invalid user ID
    with pytest.raises(ActivityTrackingError) as excinfo:
        log_mining_activity(
            user_id="non_existent_user",
            hash_rate=95.5,
            power_usage=1450.0,
            temperature=75.0,
            uptime=0.98,
            efficiency=0.85,
            block_rewards=0.5,
            network_difficulty=45.0,
            profitability=0.25,
            db_session=None
        )
    assert "User not found" in str(excinfo.value)
    
    # Invalid activity type
    with pytest.raises(ActivityTrackingError) as excinfo:
        get_activity_history(
            user_id="test_user",
            activity_type="INVALID_TYPE",
            start_time=datetime.utcnow() - timedelta(days=1),
            end_time=datetime.utcnow(),
            limit=10,
            offset=0,
            db_session=None
        )
    assert "Invalid activity type" in str(excinfo.value)
    
    # Invalid time range
    with pytest.raises(ActivityTrackingError) as excinfo:
        get_activity_history(
            user_id="test_user",
            activity_type=None,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() - timedelta(days=1),
            limit=10,
            offset=0,
            db_session=None
        )
    assert "Invalid time range" in str(excinfo.value)
    
    # Invalid limit
    with pytest.raises(ActivityTrackingError) as excinfo:
        get_activity_history(
            user_id="test_user",
            activity_type=None,
            start_time=datetime.utcnow() - timedelta(days=1),
            end_time=datetime.utcnow(),
            limit=-1,
            offset=0,
            db_session=None
        )
    assert "Invalid limit" in str(excinfo.value)
    
    # Invalid offset
    with pytest.raises(ActivityTrackingError) as excinfo:
        get_activity_history(
            user_id="test_user",
            activity_type=None,
            start_time=datetime.utcnow() - timedelta(days=1),
            end_time=datetime.utcnow(),
            limit=10,
            offset=-1,
            db_session=None
        )
    assert "Invalid offset" in str(excinfo.value)
    
    # Invalid health status
    with pytest.raises(ActivityTrackingError) as excinfo:
        monitor_activity_health(
            user_id="test_user",
            start_time=datetime.utcnow() - timedelta(days=1),
            end_time=datetime.utcnow(),
            db_session=None
        )
    assert "No activity data found" in str(excinfo.value) 