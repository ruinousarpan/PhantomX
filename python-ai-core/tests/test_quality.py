import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import uuid
from unittest.mock import patch, MagicMock

from core.quality import (
    calculate_quality_metrics,
    validate_data_rules,
    profile_data,
    monitor_quality,
    detect_anomalies,
    track_quality_issues,
    remediate_issues,
    generate_quality_report,
    QualityError
)
from database.models import User, QualityRecord, QualityIssueRecord
from database.exceptions import QualityError as DBQualityError

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
def test_data():
    """Create test data"""
    # Generate test data with known quality issues
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="H")
    data = pd.DataFrame({
        "timestamp": dates,
        "user_id": [f"user_{i}" for i in range(1000)],
        "trading_performance": np.random.uniform(0.7, 1.0, 1000),
        "risk_score": np.random.uniform(0.1, 0.5, 1000),
        "transaction_amount": np.random.uniform(100, 10000, 1000),
        "category": np.random.choice(["A", "B", "C", None], 1000),
        "status": np.random.choice(["active", "inactive", "pending", "invalid"], 1000),
        "email": [f"user_{i}@example.com" if i % 10 != 0 else None for i in range(1000)],
        "age": np.random.randint(18, 80, 1000)
    })
    
    # Introduce some quality issues
    data.loc[0:10, "trading_performance"] = None  # Missing values
    data.loc[11:20, "trading_performance"] = 2.0  # Out of range values
    data.loc[21:30, "risk_score"] = -1.0  # Invalid values
    data.loc[31:40, "transaction_amount"] = -500  # Negative amounts
    data.loc[41:50, "email"] = "invalid_email"  # Invalid format
    data.loc[51:60, "age"] = 0  # Invalid age
    
    return data

@pytest.fixture
def test_quality_config():
    """Create test quality configuration"""
    return {
        "metrics": {
            "completeness": {
                "enabled": True,
                "threshold": 0.95,
                "fields": ["trading_performance", "risk_score", "transaction_amount", "email"]
            },
            "accuracy": {
                "enabled": True,
                "threshold": 0.98,
                "fields": ["trading_performance", "risk_score", "transaction_amount"]
            },
            "consistency": {
                "enabled": True,
                "threshold": 0.99,
                "fields": ["status", "category"]
            },
            "validity": {
                "enabled": True,
                "threshold": 0.97,
                "fields": ["email", "age"]
            },
            "timeliness": {
                "enabled": True,
                "threshold": 0.99,
                "fields": ["timestamp"]
            }
        },
        "validation_rules": {
            "trading_performance": {
                "type": "numeric",
                "required": True,
                "min": 0.0,
                "max": 1.0,
                "decimals": 4
            },
            "risk_score": {
                "type": "numeric",
                "required": True,
                "min": 0.0,
                "max": 1.0,
                "decimals": 4
            },
            "transaction_amount": {
                "type": "numeric",
                "required": True,
                "min": 0.0,
                "decimals": 2
            },
            "email": {
                "type": "string",
                "required": True,
                "format": "email",
                "max_length": 100
            },
            "age": {
                "type": "numeric",
                "required": True,
                "min": 18,
                "max": 120,
                "decimals": 0
            },
            "status": {
                "type": "categorical",
                "required": True,
                "allowed_values": ["active", "inactive", "pending"]
            },
            "category": {
                "type": "categorical",
                "required": False,
                "allowed_values": ["A", "B", "C"]
            }
        },
        "profiling": {
            "enabled": True,
            "statistics": [
                "count",
                "mean",
                "std",
                "min",
                "max",
                "quartiles",
                "missing_count",
                "unique_count"
            ],
            "visualizations": [
                "distribution",
                "boxplot",
                "correlation"
            ],
            "sample_size": 1000
        },
        "monitoring": {
            "frequency": "hourly",
            "lookback_periods": 24,
            "metrics_to_monitor": [
                "completeness",
                "accuracy",
                "validity"
            ],
            "anomaly_detection": {
                "enabled": True,
                "method": "zscore",
                "threshold": 3.0,
                "min_periods": 10
            },
            "alerts": {
                "enabled": True,
                "channels": ["email"],
                "recipients": ["test@example.com"]
            }
        },
        "remediation": {
            "auto_remediation": {
                "enabled": True,
                "strategies": {
                    "missing_values": "interpolate",
                    "outliers": "clip",
                    "invalid_values": "remove"
                }
            },
            "manual_review": {
                "enabled": True,
                "threshold": 0.1  # Require manual review if more than 10% of data is affected
            }
        }
    }

def test_calculate_quality_metrics(db_session, test_user, test_data, test_quality_config):
    """Test quality metrics calculation"""
    # Calculate quality metrics
    result = calculate_quality_metrics(
        data=test_data,
        metrics_config=test_quality_config["metrics"],
        db_session=db_session
    )
    
    # Verify metrics result
    assert isinstance(result, Dict)
    assert "metrics_id" in result
    assert "timestamp" in result
    assert "metrics" in result
    
    # Verify individual metrics
    metrics = result["metrics"]
    assert "completeness" in metrics
    assert "accuracy" in metrics
    assert "consistency" in metrics
    assert "validity" in metrics
    assert "timeliness" in metrics
    
    # Verify completeness metrics
    completeness = metrics["completeness"]
    assert isinstance(completeness, Dict)
    assert "overall_score" in completeness
    assert "field_scores" in completeness
    assert all(field in completeness["field_scores"] 
              for field in test_quality_config["metrics"]["completeness"]["fields"])
    
    # Verify accuracy metrics
    accuracy = metrics["accuracy"]
    assert isinstance(accuracy, Dict)
    assert "overall_score" in accuracy
    assert "field_scores" in accuracy
    assert all(field in accuracy["field_scores"] 
              for field in test_quality_config["metrics"]["accuracy"]["fields"])
    
    # Verify quality record
    quality_record = db_session.query(QualityRecord).filter_by(
        metrics_id=result["metrics_id"]
    ).first()
    assert quality_record is not None
    assert quality_record.status == "COMPLETED"
    assert quality_record.error is None

def test_validate_data_rules(db_session, test_user, test_data, test_quality_config):
    """Test data validation rules"""
    # Validate data rules
    result = validate_data_rules(
        data=test_data,
        rules_config=test_quality_config["validation_rules"],
        db_session=db_session
    )
    
    # Verify validation result
    assert isinstance(result, Dict)
    assert "validation_id" in result
    assert "timestamp" in result
    assert "validation_results" in result
    
    # Verify validation results
    validation_results = result["validation_results"]
    assert "trading_performance" in validation_results
    assert "risk_score" in validation_results
    assert "transaction_amount" in validation_results
    assert "email" in validation_results
    assert "age" in validation_results
    
    # Verify individual field validations
    trading_performance = validation_results["trading_performance"]
    assert "valid" in trading_performance
    assert "invalid_count" in trading_performance
    assert "invalid_indices" in trading_performance
    assert "error_types" in trading_performance
    
    # Verify quality record
    quality_record = db_session.query(QualityRecord).filter_by(
        validation_id=result["validation_id"]
    ).first()
    assert quality_record is not None
    assert quality_record.status == "COMPLETED"
    assert quality_record.error is None

def test_profile_data(db_session, test_user, test_data, test_quality_config):
    """Test data profiling"""
    # Profile data
    result = profile_data(
        data=test_data,
        profiling_config=test_quality_config["profiling"],
        db_session=db_session
    )
    
    # Verify profiling result
    assert isinstance(result, Dict)
    assert "profile_id" in result
    assert "timestamp" in result
    assert "profile" in result
    
    # Verify profile content
    profile = result["profile"]
    assert "statistics" in profile
    assert "visualizations" in profile
    assert "correlations" in profile
    
    # Verify statistics
    statistics = profile["statistics"]
    for column in test_data.columns:
        assert column in statistics
        stats = statistics[column]
        assert "count" in stats
        assert "missing_count" in stats
        assert "unique_count" in stats
        if np.issubdtype(test_data[column].dtype, np.number):
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats
            assert "quartiles" in stats
    
    # Verify quality record
    quality_record = db_session.query(QualityRecord).filter_by(
        profile_id=result["profile_id"]
    ).first()
    assert quality_record is not None
    assert quality_record.status == "COMPLETED"
    assert quality_record.error is None

def test_monitor_quality(db_session, test_user, test_data, test_quality_config):
    """Test quality monitoring"""
    # Monitor quality
    result = monitor_quality(
        data=test_data,
        monitoring_config=test_quality_config["monitoring"],
        db_session=db_session
    )
    
    # Verify monitoring result
    assert isinstance(result, Dict)
    assert "monitoring_id" in result
    assert "timestamp" in result
    assert "metrics" in result
    assert "anomalies" in result
    assert "alerts" in result
    
    # Verify metrics
    metrics = result["metrics"]
    assert all(metric in metrics for metric in test_quality_config["monitoring"]["metrics_to_monitor"])
    
    # Verify anomalies
    anomalies = result["anomalies"]
    assert isinstance(anomalies, List)
    for anomaly in anomalies:
        assert "metric" in anomaly
        assert "value" in anomaly
        assert "threshold" in anomaly
        assert "zscore" in anomaly
    
    # Verify quality record
    quality_record = db_session.query(QualityRecord).filter_by(
        monitoring_id=result["monitoring_id"]
    ).first()
    assert quality_record is not None
    assert quality_record.status == "COMPLETED"
    assert quality_record.error is None

def test_detect_anomalies(db_session, test_user, test_data, test_quality_config):
    """Test anomaly detection"""
    # Detect anomalies
    result = detect_anomalies(
        data=test_data,
        anomaly_config=test_quality_config["monitoring"]["anomaly_detection"],
        db_session=db_session
    )
    
    # Verify anomaly detection result
    assert isinstance(result, Dict)
    assert "detection_id" in result
    assert "timestamp" in result
    assert "anomalies" in result
    
    # Verify anomalies
    anomalies = result["anomalies"]
    assert isinstance(anomalies, List)
    for anomaly in anomalies:
        assert "field" in anomaly
        assert "index" in anomaly
        assert "value" in anomaly
        assert "score" in anomaly
        assert "threshold" in anomaly
    
    # Verify quality record
    quality_record = db_session.query(QualityRecord).filter_by(
        detection_id=result["detection_id"]
    ).first()
    assert quality_record is not None
    assert quality_record.status == "COMPLETED"
    assert quality_record.error is None

def test_track_quality_issues(db_session, test_user, test_data, test_quality_config):
    """Test quality issue tracking"""
    # Create quality issue
    issue_data = {
        "type": "data_quality",
        "severity": "high",
        "description": "Multiple quality issues detected",
        "affected_fields": ["trading_performance", "risk_score"],
        "affected_records": 50,
        "timestamp": datetime.utcnow(),
        "metrics": {
            "completeness": 0.92,
            "accuracy": 0.94
        }
    }
    
    result = track_quality_issues(
        issue_data=issue_data,
        db_session=db_session
    )
    
    # Verify issue tracking result
    assert isinstance(result, Dict)
    assert "issue_id" in result
    assert "status" in result
    assert "timestamp" in result
    assert "details" in result
    
    # Verify issue record
    issue_record = db_session.query(QualityIssueRecord).filter_by(
        issue_id=result["issue_id"]
    ).first()
    assert issue_record is not None
    assert issue_record.status == "OPEN"
    assert issue_record.severity == "high"
    assert issue_record.error is None

def test_remediate_issues(db_session, test_user, test_data, test_quality_config):
    """Test issue remediation"""
    # Remediate issues
    result = remediate_issues(
        data=test_data,
        remediation_config=test_quality_config["remediation"],
        issues=[{
            "field": "trading_performance",
            "type": "missing_values",
            "indices": list(range(11))
        }, {
            "field": "risk_score",
            "type": "invalid_values",
            "indices": list(range(21, 31))
        }],
        db_session=db_session
    )
    
    # Verify remediation result
    assert isinstance(result, Dict)
    assert "remediation_id" in result
    assert "timestamp" in result
    assert "remediated_data" in result
    assert "summary" in result
    
    # Verify remediated data
    remediated_data = result["remediated_data"]
    assert isinstance(remediated_data, pd.DataFrame)
    assert len(remediated_data) == len(test_data)
    
    # Verify remediation summary
    summary = result["summary"]
    assert "total_issues" in summary
    assert "remediated_issues" in summary
    assert "manual_review_required" in summary
    
    # Verify quality record
    quality_record = db_session.query(QualityRecord).filter_by(
        remediation_id=result["remediation_id"]
    ).first()
    assert quality_record is not None
    assert quality_record.status == "COMPLETED"
    assert quality_record.error is None

def test_generate_quality_report(db_session, test_user, test_data, test_quality_config):
    """Test quality report generation"""
    # Generate quality report
    result = generate_quality_report(
        data=test_data,
        config=test_quality_config,
        start_time=datetime.utcnow() - timedelta(days=7),
        end_time=datetime.utcnow(),
        db_session=db_session
    )
    
    # Verify report result
    assert isinstance(result, Dict)
    assert "report_id" in result
    assert "timestamp" in result
    assert "report" in result
    
    # Verify report content
    report = result["report"]
    assert "summary" in report
    assert "metrics" in report
    assert "issues" in report
    assert "remediation" in report
    assert "recommendations" in report
    
    # Verify quality record
    quality_record = db_session.query(QualityRecord).filter_by(
        report_id=result["report_id"]
    ).first()
    assert quality_record is not None
    assert quality_record.status == "GENERATED"
    assert quality_record.error is None

def test_quality_error_handling(db_session, test_user):
    """Test quality error handling"""
    # Invalid metrics configuration
    with pytest.raises(QualityError) as excinfo:
        calculate_quality_metrics(
            data=pd.DataFrame(),
            metrics_config={},
            db_session=db_session
        )
    assert "Invalid metrics configuration" in str(excinfo.value)
    
    # Invalid validation rules
    with pytest.raises(QualityError) as excinfo:
        validate_data_rules(
            data=pd.DataFrame(),
            rules_config={},
            db_session=db_session
        )
    assert "Invalid validation rules" in str(excinfo.value)
    
    # Invalid profiling configuration
    with pytest.raises(QualityError) as excinfo:
        profile_data(
            data=pd.DataFrame(),
            profiling_config={},
            db_session=db_session
        )
    assert "Invalid profiling configuration" in str(excinfo.value)
    
    # Invalid monitoring configuration
    with pytest.raises(QualityError) as excinfo:
        monitor_quality(
            data=pd.DataFrame(),
            monitoring_config={},
            db_session=db_session
        )
    assert "Invalid monitoring configuration" in str(excinfo.value)
    
    # Invalid remediation configuration
    with pytest.raises(QualityError) as excinfo:
        remediate_issues(
            data=pd.DataFrame(),
            remediation_config={},
            issues=[],
            db_session=db_session
        )
    assert "Invalid remediation configuration" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBQualityError) as excinfo:
        calculate_quality_metrics(
            data=pd.DataFrame(),
            metrics_config={},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 