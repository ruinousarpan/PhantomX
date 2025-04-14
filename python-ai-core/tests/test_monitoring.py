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
from unittest.mock import patch, MagicMock

from core.monitoring import (
    monitor_performance,
    monitor_risk,
    monitor_reward,
    monitor_activity,
    generate_alerts,
    check_system_health,
    validate_monitoring,
    monitor_data,
    get_monitor_info,
    check_health,
    manage_alerts,
    collect_metrics,
    generate_report,
    get_monitoring_info,
    list_metrics,
    delete_metric,
    MonitoringError
)
from database.models import User, MonitoringResult, Alert, MonitorRecord, MonitoringRecord, AlertRecord, MetricRecord, HealthCheck
from database.exceptions import MonitoringError as DBMonitoringError

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
def test_thresholds():
    """Create test thresholds"""
    return {
        "performance": {
            "mining": {"warning": 0.7, "critical": 0.5},
            "staking": {"warning": 0.75, "critical": 0.6},
            "trading": {"warning": 0.65, "critical": 0.5},
            "overall": {"warning": 0.7, "critical": 0.6}
        },
        "risk": {
            "mining": {"warning": 0.4, "critical": 0.6},
            "staking": {"warning": 0.3, "critical": 0.5},
            "trading": {"warning": 0.5, "critical": 0.7},
            "overall": {"warning": 0.4, "critical": 0.6}
        },
        "reward": {
            "mining": {"warning": 0.3, "critical": 0.2},
            "staking": {"warning": 0.08, "critical": 0.05},
            "trading": {"warning": 0.07, "critical": 0.04},
            "overall": {"warning": 0.15, "critical": 0.1}
        },
        "activity": {
            "mining": {"warning": 0.6, "critical": 0.5},
            "staking": {"warning": 0.7, "critical": 0.6},
            "trading": {"warning": 0.5, "critical": 0.4},
            "overall": {"warning": 0.6, "critical": 0.5}
        }
    }

@pytest.fixture
def test_health_checks():
    """Create test health checks"""
    return {
        "system": {
            "cpu_usage": {"warning": 80, "critical": 90},
            "memory_usage": {"warning": 80, "critical": 90},
            "disk_usage": {"warning": 80, "critical": 90},
            "network_latency": {"warning": 100, "critical": 200}
        },
        "database": {
            "connection_pool": {"warning": 80, "critical": 90},
            "query_time": {"warning": 100, "critical": 200},
            "transaction_time": {"warning": 100, "critical": 200},
            "error_rate": {"warning": 5, "critical": 10}
        },
        "api": {
            "response_time": {"warning": 100, "critical": 200},
            "error_rate": {"warning": 5, "critical": 10},
            "request_rate": {"warning": 1000, "critical": 2000},
            "availability": {"warning": 99, "critical": 95}
        }
    }

@pytest.fixture
def test_monitor_config():
    """Create test monitor configuration"""
    return {
        "monitor_type": "performance",
        "metrics": {
            "thresholds": {
                "mining_performance": {"min": 0.75, "max": 0.95, "warning": 0.05},
                "staking_performance": {"min": 0.80, "max": 1.00, "warning": 0.05},
                "trading_performance": {"min": 0.65, "max": 0.85, "warning": 0.05},
                "overall_performance": {"min": 0.75, "max": 0.95, "warning": 0.05}
            },
            "anomaly_detection": {
                "method": "zscore",
                "window": 24,
                "threshold": 3.0
            },
            "trend_detection": {
                "method": "regression",
                "window": 24,
                "threshold": 0.1
            }
        },
        "alerts": {
            "threshold_breach": {
                "severity": "high",
                "channels": ["email", "slack"],
                "throttle": "1h"
            },
            "anomaly_detected": {
                "severity": "medium",
                "channels": ["email"],
                "throttle": "6h"
            },
            "trend_change": {
                "severity": "low",
                "channels": ["slack"],
                "throttle": "12h"
            }
        },
        "schedule": {
            "frequency": "5m",
            "backtest": "24h",
            "timeout": "30s"
        }
    }

@pytest.fixture
def test_data():
    """Create test data"""
    # Generate test data
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="H")
    data = pd.DataFrame({
        "timestamp": dates,
        "user_id": [f"user_{i}" for i in range(1000)],
        "trading_performance": np.random.uniform(0.7, 1.0, 1000),
        "risk_score": np.random.uniform(0.1, 0.5, 1000),
        "transaction_amount": np.random.uniform(100, 10000, 1000),
        "category": np.random.choice(["A", "B", "C"], 1000),
        "status": np.random.choice(["active", "inactive", "pending"], 1000)
    })
    return data

@pytest.fixture
def test_monitoring_config():
    """Create test monitoring configuration"""
    return {
        "performance_monitoring": {
            "enabled": True,
            "metrics": [
                "cpu_usage",
                "memory_usage",
                "disk_usage",
                "network_latency",
                "response_time",
                "throughput",
                "error_rate"
            ],
            "thresholds": {
                "cpu_usage": 80.0,
                "memory_usage": 85.0,
                "disk_usage": 90.0,
                "network_latency": 100.0,
                "response_time": 500.0,
                "error_rate": 1.0
            },
            "collection_interval": 60,
            "retention_period": 30
        },
        "health_checks": {
            "enabled": True,
            "checks": [
                "database_connection",
                "api_endpoints",
                "service_status",
                "data_integrity",
                "system_resources"
            ],
            "intervals": {
                "database_connection": 300,
                "api_endpoints": 60,
                "service_status": 120,
                "data_integrity": 3600,
                "system_resources": 300
            },
            "timeout": 30,
            "retries": 3
        },
        "alert_management": {
            "enabled": True,
            "alert_types": [
                "critical",
                "error",
                "warning",
                "info"
            ],
            "channels": [
                "email",
                "slack",
                "webhook",
                "dashboard"
            ],
            "throttling": {
                "enabled": True,
                "window": 300,
                "max_alerts": 10
            },
            "aggregation": {
                "enabled": True,
                "window": 300,
                "group_by": ["alert_type", "source"]
            }
        },
        "metric_collection": {
            "enabled": True,
            "types": [
                "counter",
                "gauge",
                "histogram",
                "summary"
            ],
            "dimensions": [
                "service",
                "endpoint",
                "method",
                "status_code"
            ],
            "aggregations": [
                "sum",
                "avg",
                "min",
                "max",
                "count",
                "p95",
                "p99"
            ],
            "storage": {
                "type": "timeseries",
                "resolution": 60,
                "retention": 90
            }
        },
        "status_reporting": {
            "enabled": True,
            "report_types": [
                "system_status",
                "performance_metrics",
                "health_status",
                "alert_summary",
                "resource_usage"
            ],
            "formats": [
                "json",
                "html",
                "pdf"
            ],
            "scheduling": {
                "enabled": True,
                "interval": 3600,
                "retention": 30
            }
        }
    }

@pytest.fixture
def test_metrics_data():
    """Create test metrics data"""
    timestamps = pd.date_range(start="2023-01-01", periods=100, freq="H")
    return pd.DataFrame({
        "timestamp": timestamps,
        "cpu_usage": np.random.uniform(20, 90, 100),
        "memory_usage": np.random.uniform(30, 85, 100),
        "disk_usage": np.random.uniform(40, 95, 100),
        "network_latency": np.random.uniform(10, 200, 100),
        "response_time": np.random.uniform(100, 1000, 100),
        "throughput": np.random.uniform(1000, 5000, 100),
        "error_rate": np.random.uniform(0, 2, 100)
    })

def test_monitor_performance(db_session, test_user, test_metrics_data, test_monitoring_config):
    """Test performance monitoring"""
    # Monitor performance
    result = monitor_performance(
        metrics=test_metrics_data,
        config=test_monitoring_config["performance_monitoring"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify monitoring result
    assert isinstance(result, Dict)
    assert "monitoring_id" in result
    assert "timestamp" in result
    assert "metrics" in result
    
    # Verify metrics
    metrics = result["metrics"]
    for metric_name in test_monitoring_config["performance_monitoring"]["metrics"]:
        assert metric_name in metrics
        assert "value" in metrics[metric_name]
        assert "threshold" in metrics[metric_name]
        assert "status" in metrics[metric_name]
    
    # Verify monitoring record
    monitoring_record = db_session.query(MonitoringRecord).filter_by(
        monitoring_id=result["monitoring_id"]
    ).first()
    assert monitoring_record is not None
    assert monitoring_record.status == "COMPLETED"
    assert monitoring_record.error is None

def test_check_system_health(db_session, test_user, test_monitoring_config):
    """Test system health checks"""
    # Check system health
    result = check_system_health(
        config=test_monitoring_config["health_checks"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify health check result
    assert isinstance(result, Dict)
    assert "check_id" in result
    assert "timestamp" in result
    assert "checks" in result
    
    # Verify health checks
    checks = result["checks"]
    for check_name in test_monitoring_config["health_checks"]["checks"]:
        assert check_name in checks
        assert "status" in checks[check_name]
        assert "latency" in checks[check_name]
        assert "message" in checks[check_name]
    
    # Verify health check record
    health_check = db_session.query(HealthCheck).filter_by(
        check_id=result["check_id"]
    ).first()
    assert health_check is not None
    assert health_check.status == "COMPLETED"
    assert health_check.error is None

def test_manage_alerts(db_session, test_user, test_monitoring_config):
    """Test alert management"""
    # Create test alerts
    alerts = [
        {
            "type": "critical",
            "source": "cpu_usage",
            "message": "CPU usage exceeded threshold",
            "value": 95.0,
            "threshold": 80.0
        },
        {
            "type": "warning",
            "source": "memory_usage",
            "message": "High memory usage detected",
            "value": 82.0,
            "threshold": 85.0
        }
    ]
    
    # Manage alerts
    result = manage_alerts(
        alerts=alerts,
        config=test_monitoring_config["alert_management"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify alert management result
    assert isinstance(result, Dict)
    assert "alert_id" in result
    assert "timestamp" in result
    assert "alerts" in result
    
    # Verify alerts
    managed_alerts = result["alerts"]
    assert len(managed_alerts) == len(alerts)
    for alert in managed_alerts:
        assert "id" in alert
        assert "type" in alert
        assert "source" in alert
        assert "status" in alert
        assert "timestamp" in alert
    
    # Verify alert records
    alert_records = db_session.query(AlertRecord).filter_by(
        alert_batch_id=result["alert_id"]
    ).all()
    assert len(alert_records) == len(alerts)
    for record in alert_records:
        assert record.status == "PROCESSED"
        assert record.error is None

def test_collect_metrics(db_session, test_user, test_metrics_data, test_monitoring_config):
    """Test metric collection"""
    # Collect metrics
    result = collect_metrics(
        metrics=test_metrics_data,
        config=test_monitoring_config["metric_collection"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify metric collection result
    assert isinstance(result, Dict)
    assert "collection_id" in result
    assert "timestamp" in result
    assert "metrics" in result
    
    # Verify collected metrics
    collected_metrics = result["metrics"]
    for metric_type in test_monitoring_config["metric_collection"]["types"]:
        assert metric_type in collected_metrics
        assert "values" in collected_metrics[metric_type]
        assert "dimensions" in collected_metrics[metric_type]
        assert "aggregations" in collected_metrics[metric_type]
    
    # Verify metric records
    metric_records = db_session.query(MetricRecord).filter_by(
        collection_id=result["collection_id"]
    ).all()
    assert len(metric_records) > 0
    for record in metric_records:
        assert record.status == "COLLECTED"
        assert record.error is None

def test_generate_status_report(db_session, test_user, test_metrics_data, test_monitoring_config):
    """Test status report generation"""
    # Generate status report
    result = generate_status_report(
        metrics=test_metrics_data,
        config=test_monitoring_config["status_reporting"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify report result
    assert isinstance(result, Dict)
    assert "report_id" in result
    assert "timestamp" in result
    assert "report" in result
    
    # Verify report content
    report = result["report"]
    for report_type in test_monitoring_config["status_reporting"]["report_types"]:
        assert report_type in report
        assert isinstance(report[report_type], Dict)
    
    # Verify report format
    assert "format" in result
    assert result["format"] in test_monitoring_config["status_reporting"]["formats"]
    
    # Verify monitoring record
    monitoring_record = db_session.query(MonitoringRecord).filter_by(
        report_id=result["report_id"]
    ).first()
    assert monitoring_record is not None
    assert monitoring_record.status == "REPORTED"
    assert monitoring_record.error is None

def test_get_monitoring_info(db_session, test_user, test_metrics_data, test_monitoring_config):
    """Test monitoring information retrieval"""
    # First, create monitoring record
    monitoring = monitor_performance(
        metrics=test_metrics_data,
        config=test_monitoring_config["performance_monitoring"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Get monitoring info
    result = get_monitoring_info(
        monitoring_id=monitoring["monitoring_id"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "monitoring_id" in result
    assert "timestamp" in result
    assert "info" in result
    
    # Verify monitoring info
    info = result["info"]
    assert "metrics" in info
    assert "health_checks" in info
    assert "alerts" in info
    assert "status" in info
    
    # Verify monitoring record
    monitoring_record = db_session.query(MonitoringRecord).filter_by(
        monitoring_id=result["monitoring_id"]
    ).first()
    assert monitoring_record is not None
    assert monitoring_record.status == "RETRIEVED"
    assert monitoring_record.error is None

def test_list_monitoring_records(db_session, test_user, test_metrics_data, test_monitoring_config):
    """Test monitoring record listing"""
    # Create multiple monitoring records
    for _ in range(5):
        monitor_performance(
            metrics=test_metrics_data,
            config=test_monitoring_config["performance_monitoring"],
            user_id=test_user.user_id,
            db_session=db_session
        )
    
    # List monitoring records
    result = list_monitoring_records(
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify listing result
    assert isinstance(result, Dict)
    assert "timestamp" in result
    assert "records" in result
    
    # Verify records list
    records = result["records"]
    assert isinstance(records, List)
    assert len(records) == 5
    
    # Verify record details
    for record in records:
        assert "monitoring_id" in record
        assert "timestamp" in record
        assert "type" in record
        assert "status" in record

def test_delete_monitoring_record(db_session, test_user, test_metrics_data, test_monitoring_config):
    """Test monitoring record deletion"""
    # First, create monitoring record
    monitoring = monitor_performance(
        metrics=test_metrics_data,
        config=test_monitoring_config["performance_monitoring"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Delete monitoring record
    result = delete_monitoring_record(
        monitoring_id=monitoring["monitoring_id"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify deletion result
    assert isinstance(result, Dict)
    assert "deletion_id" in result
    assert "timestamp" in result
    assert "status" in result
    
    # Verify status
    assert result["status"] == "DELETED"
    
    # Verify monitoring record
    monitoring_record = db_session.query(MonitoringRecord).filter_by(
        monitoring_id=monitoring["monitoring_id"]
    ).first()
    assert monitoring_record is not None
    assert monitoring_record.status == "DELETED"
    assert monitoring_record.error is None

def test_monitoring_error_handling(db_session, test_user):
    """Test monitoring error handling"""
    # Invalid metrics data
    with pytest.raises(MonitoringError) as excinfo:
        monitor_performance(
            metrics=None,
            config={},
            user_id=test_user.user_id,
            db_session=db_session
        )
    assert "Invalid metrics data" in str(excinfo.value)
    
    # Invalid health check configuration
    with pytest.raises(MonitoringError) as excinfo:
        check_system_health(
            config={},
            user_id=test_user.user_id,
            db_session=db_session
        )
    assert "Invalid health check configuration" in str(excinfo.value)
    
    # Invalid alert data
    with pytest.raises(MonitoringError) as excinfo:
        manage_alerts(
            alerts=None,
            config={},
            user_id=test_user.user_id,
            db_session=db_session
        )
    assert "Invalid alert data" in str(excinfo.value)
    
    # Invalid metric collection configuration
    with pytest.raises(MonitoringError) as excinfo:
        collect_metrics(
            metrics=pd.DataFrame(),
            config={},
            user_id=test_user.user_id,
            db_session=db_session
        )
    assert "Invalid metric collection configuration" in str(excinfo.value)
    
    # Invalid monitoring ID
    with pytest.raises(MonitoringError) as excinfo:
        get_monitoring_info(
            monitoring_id="invalid_id",
            user_id=test_user.user_id,
            db_session=db_session
        )
    assert "Invalid monitoring ID" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBMonitoringError) as excinfo:
        monitor_performance(
            metrics=pd.DataFrame(),
            config={},
            user_id=test_user.user_id,
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 