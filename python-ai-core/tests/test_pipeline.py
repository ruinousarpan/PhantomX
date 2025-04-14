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

from core.pipeline import (
    create_pipeline,
    execute_pipeline,
    get_pipeline_info,
    update_pipeline,
    delete_pipeline,
    get_execution_history,
    PipelineError
)
from database.models import User, PipelineRecord, PipelineExecutionRecord
from database.exceptions import PipelineError as DBPipelineError

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
def test_pipeline_config():
    """Create test pipeline configuration"""
    return {
        "name": "performance_analysis_pipeline",
        "description": "Pipeline for performance data analysis",
        "version": "1.0.0",
        "steps": [
            {
                "name": "data_validation",
                "type": "validation",
                "config": {
                    "validation_type": "performance",
                    "validation_level": "strict",
                    "metrics": ["mining_performance", "staking_performance", "trading_performance"]
                }
            },
            {
                "name": "data_normalization",
                "type": "normalization",
                "config": {
                    "method": "min_max",
                    "columns": ["mining_performance", "staking_performance", "trading_performance"]
                }
            },
            {
                "name": "trend_analysis",
                "type": "analysis",
                "config": {
                    "analysis_type": "trend",
                    "window_size": 7,
                    "metrics": ["mining_performance", "staking_performance", "trading_performance"]
                }
            },
            {
                "name": "report_generation",
                "type": "reporting",
                "config": {
                    "report_type": "performance_summary",
                    "format": "pdf",
                    "sections": ["summary", "trends", "recommendations"]
                }
            }
        ],
        "dependencies": {
            "data_validation": [],
            "data_normalization": ["data_validation"],
            "trend_analysis": ["data_normalization"],
            "report_generation": ["trend_analysis"]
        },
        "error_handling": {
            "retry": {
                "enabled": True,
                "max_attempts": 3,
                "delay": 300  # 5 minutes
            },
            "fallback": {
                "enabled": True,
                "steps": ["data_validation", "data_normalization"]
            }
        },
        "notification": {
            "enabled": True,
            "channels": ["email"],
            "recipients": ["test@example.com"],
            "on_success": True,
            "on_failure": True
        },
        "metadata": {
            "tags": ["performance", "analysis", "automated"],
            "priority": "high",
            "owner": "test_user"
        }
    }

def test_create_performance_pipeline(db_session, test_user, test_performance_data, test_pipeline_config):
    """Test creating a performance analysis pipeline"""
    # Create pipeline
    result = create_pipeline(
        user_id=test_user.user_id,
        pipeline_config=test_pipeline_config,
        db_session=db_session
    )
    
    # Verify pipeline result
    assert isinstance(result, Dict)
    assert "pipeline_id" in result
    assert "pipeline_details" in result
    
    # Verify pipeline metadata
    assert result["name"] == "performance_analysis_pipeline"
    assert result["description"] == "Pipeline for performance data analysis"
    assert result["version"] == "1.0.0"
    
    # Verify pipeline steps
    assert "steps" in result
    assert len(result["steps"]) == 4
    step_names = [step["name"] for step in result["steps"]]
    assert all(name in step_names for name in ["data_validation", "data_normalization", "trend_analysis", "report_generation"])
    
    # Verify dependencies
    assert "dependencies" in result
    dependencies = result["dependencies"]
    assert "data_validation" in dependencies
    assert "data_normalization" in dependencies
    assert "trend_analysis" in dependencies
    assert "report_generation" in dependencies
    assert dependencies["data_validation"] == []
    assert "data_validation" in dependencies["data_normalization"]
    assert "data_normalization" in dependencies["trend_analysis"]
    assert "trend_analysis" in dependencies["report_generation"]
    
    # Verify error handling
    assert "error_handling" in result
    error_handling = result["error_handling"]
    assert "retry" in error_handling
    assert "fallback" in error_handling
    assert error_handling["retry"]["enabled"] is True
    assert error_handling["retry"]["max_attempts"] == 3
    assert error_handling["retry"]["delay"] == 300
    assert error_handling["fallback"]["enabled"] is True
    assert all(step in error_handling["fallback"]["steps"] for step in ["data_validation", "data_normalization"])
    
    # Verify notification settings
    assert "notification" in result
    notification = result["notification"]
    assert notification["enabled"] is True
    assert "email" in notification["channels"]
    assert "test@example.com" in notification["recipients"]
    assert notification["on_success"] is True
    assert notification["on_failure"] is True
    
    # Verify metadata
    assert "metadata" in result
    metadata = result["metadata"]
    assert all(tag in metadata["tags"] for tag in ["performance", "analysis", "automated"])
    assert metadata["priority"] == "high"
    assert metadata["owner"] == "test_user"
    
    # Verify database entry
    db_record = db_session.query(PipelineRecord).filter_by(
        user_id=test_user.user_id,
        pipeline_id=result["pipeline_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_create_risk_pipeline(db_session, test_user, test_risk_data):
    """Test creating a risk monitoring pipeline"""
    # Create pipeline config for risk monitoring
    risk_config = {
        "name": "risk_monitoring_pipeline",
        "description": "Pipeline for risk data monitoring",
        "version": "1.0.0",
        "steps": [
            {
                "name": "data_validation",
                "type": "validation",
                "config": {
                    "validation_type": "risk",
                    "validation_level": "strict",
                    "metrics": ["mining_risk", "staking_risk", "trading_risk"]
                }
            },
            {
                "name": "threshold_check",
                "type": "monitoring",
                "config": {
                    "monitoring_type": "threshold",
                    "thresholds": {
                        "mining_risk": 0.3,
                        "staking_risk": 0.2,
                        "trading_risk": 0.4
                    }
                }
            },
            {
                "name": "anomaly_detection",
                "type": "analysis",
                "config": {
                    "analysis_type": "anomaly",
                    "method": "isolation_forest",
                    "metrics": ["mining_risk", "staking_risk", "trading_risk"]
                }
            },
            {
                "name": "alert_generation",
                "type": "notification",
                "config": {
                    "alert_type": "risk_alert",
                    "channels": ["email", "webhook"],
                    "severity_levels": ["high", "medium", "low"]
                }
            }
        ],
        "dependencies": {
            "data_validation": [],
            "threshold_check": ["data_validation"],
            "anomaly_detection": ["data_validation"],
            "alert_generation": ["threshold_check", "anomaly_detection"]
        },
        "error_handling": {
            "retry": {
                "enabled": True,
                "max_attempts": 5,
                "delay": 600  # 10 minutes
            },
            "fallback": {
                "enabled": True,
                "steps": ["data_validation"]
            }
        },
        "notification": {
            "enabled": True,
            "channels": ["email", "webhook"],
            "recipients": ["test@example.com"],
            "webhook_url": "https://example.com/webhook",
            "on_success": True,
            "on_failure": True
        },
        "metadata": {
            "tags": ["risk", "monitoring", "real-time"],
            "priority": "critical",
            "owner": "test_user"
        }
    }
    
    # Create pipeline
    result = create_pipeline(
        user_id=test_user.user_id,
        pipeline_config=risk_config,
        db_session=db_session
    )
    
    # Verify pipeline result
    assert isinstance(result, Dict)
    assert "pipeline_id" in result
    assert "pipeline_details" in result
    
    # Verify pipeline metadata
    assert result["name"] == "risk_monitoring_pipeline"
    assert result["description"] == "Pipeline for risk data monitoring"
    assert result["version"] == "1.0.0"
    
    # Verify pipeline steps
    assert "steps" in result
    assert len(result["steps"]) == 4
    step_names = [step["name"] for step in result["steps"]]
    assert all(name in step_names for name in ["data_validation", "threshold_check", "anomaly_detection", "alert_generation"])
    
    # Verify dependencies
    assert "dependencies" in result
    dependencies = result["dependencies"]
    assert "data_validation" in dependencies
    assert "threshold_check" in dependencies
    assert "anomaly_detection" in dependencies
    assert "alert_generation" in dependencies
    assert dependencies["data_validation"] == []
    assert "data_validation" in dependencies["threshold_check"]
    assert "data_validation" in dependencies["anomaly_detection"]
    assert all(step in dependencies["alert_generation"] for step in ["threshold_check", "anomaly_detection"])
    
    # Verify error handling
    assert "error_handling" in result
    error_handling = result["error_handling"]
    assert "retry" in error_handling
    assert "fallback" in error_handling
    assert error_handling["retry"]["enabled"] is True
    assert error_handling["retry"]["max_attempts"] == 5
    assert error_handling["retry"]["delay"] == 600
    assert error_handling["fallback"]["enabled"] is True
    assert "data_validation" in error_handling["fallback"]["steps"]
    
    # Verify notification settings
    assert "notification" in result
    notification = result["notification"]
    assert notification["enabled"] is True
    assert all(channel in notification["channels"] for channel in ["email", "webhook"])
    assert "test@example.com" in notification["recipients"]
    assert notification["webhook_url"] == "https://example.com/webhook"
    assert notification["on_success"] is True
    assert notification["on_failure"] is True
    
    # Verify metadata
    assert "metadata" in result
    metadata = result["metadata"]
    assert all(tag in metadata["tags"] for tag in ["risk", "monitoring", "real-time"])
    assert metadata["priority"] == "critical"
    assert metadata["owner"] == "test_user"
    
    # Verify database entry
    db_record = db_session.query(PipelineRecord).filter_by(
        user_id=test_user.user_id,
        pipeline_id=result["pipeline_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_create_reward_pipeline(db_session, test_user, test_reward_data):
    """Test creating a reward analysis pipeline"""
    # Create pipeline config for reward analysis
    reward_config = {
        "name": "reward_analysis_pipeline",
        "description": "Pipeline for reward data analysis",
        "version": "1.0.0",
        "steps": [
            {
                "name": "data_validation",
                "type": "validation",
                "config": {
                    "validation_type": "reward",
                    "validation_level": "strict",
                    "metrics": ["mining_reward", "staking_reward", "trading_reward"]
                }
            },
            {
                "name": "data_aggregation",
                "type": "aggregation",
                "config": {
                    "aggregation_type": "time_series",
                    "window": "1D",
                    "metrics": ["mining_reward", "staking_reward", "trading_reward"],
                    "functions": ["sum", "mean", "max"]
                }
            },
            {
                "name": "performance_analysis",
                "type": "analysis",
                "config": {
                    "analysis_type": "performance",
                    "metrics": ["mining_reward", "staking_reward", "trading_reward"],
                    "benchmarks": {
                        "mining_reward": 0.8,
                        "staking_reward": 0.9,
                        "trading_reward": 0.7
                    }
                }
            },
            {
                "name": "report_generation",
                "type": "reporting",
                "config": {
                    "report_type": "reward_summary",
                    "format": "excel",
                    "sections": ["summary", "performance", "recommendations"]
                }
            }
        ],
        "dependencies": {
            "data_validation": [],
            "data_aggregation": ["data_validation"],
            "performance_analysis": ["data_aggregation"],
            "report_generation": ["performance_analysis"]
        },
        "error_handling": {
            "retry": {
                "enabled": True,
                "max_attempts": 3,
                "delay": 300  # 5 minutes
            },
            "fallback": {
                "enabled": True,
                "steps": ["data_validation", "data_aggregation"]
            }
        },
        "notification": {
            "enabled": True,
            "channels": ["email"],
            "recipients": ["test@example.com"],
            "on_success": True,
            "on_failure": True
        },
        "metadata": {
            "tags": ["reward", "analysis", "daily"],
            "priority": "medium",
            "owner": "test_user"
        }
    }
    
    # Create pipeline
    result = create_pipeline(
        user_id=test_user.user_id,
        pipeline_config=reward_config,
        db_session=db_session
    )
    
    # Verify pipeline result
    assert isinstance(result, Dict)
    assert "pipeline_id" in result
    assert "pipeline_details" in result
    
    # Verify pipeline metadata
    assert result["name"] == "reward_analysis_pipeline"
    assert result["description"] == "Pipeline for reward data analysis"
    assert result["version"] == "1.0.0"
    
    # Verify pipeline steps
    assert "steps" in result
    assert len(result["steps"]) == 4
    step_names = [step["name"] for step in result["steps"]]
    assert all(name in step_names for name in ["data_validation", "data_aggregation", "performance_analysis", "report_generation"])
    
    # Verify dependencies
    assert "dependencies" in result
    dependencies = result["dependencies"]
    assert "data_validation" in dependencies
    assert "data_aggregation" in dependencies
    assert "performance_analysis" in dependencies
    assert "report_generation" in dependencies
    assert dependencies["data_validation"] == []
    assert "data_validation" in dependencies["data_aggregation"]
    assert "data_aggregation" in dependencies["performance_analysis"]
    assert "performance_analysis" in dependencies["report_generation"]
    
    # Verify error handling
    assert "error_handling" in result
    error_handling = result["error_handling"]
    assert "retry" in error_handling
    assert "fallback" in error_handling
    assert error_handling["retry"]["enabled"] is True
    assert error_handling["retry"]["max_attempts"] == 3
    assert error_handling["retry"]["delay"] == 300
    assert error_handling["fallback"]["enabled"] is True
    assert all(step in error_handling["fallback"]["steps"] for step in ["data_validation", "data_aggregation"])
    
    # Verify notification settings
    assert "notification" in result
    notification = result["notification"]
    assert notification["enabled"] is True
    assert "email" in notification["channels"]
    assert "test@example.com" in notification["recipients"]
    assert notification["on_success"] is True
    assert notification["on_failure"] is True
    
    # Verify metadata
    assert "metadata" in result
    metadata = result["metadata"]
    assert all(tag in metadata["tags"] for tag in ["reward", "analysis", "daily"])
    assert metadata["priority"] == "medium"
    assert metadata["owner"] == "test_user"
    
    # Verify database entry
    db_record = db_session.query(PipelineRecord).filter_by(
        user_id=test_user.user_id,
        pipeline_id=result["pipeline_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_create_activity_pipeline(db_session, test_user, test_activity_data):
    """Test creating an activity monitoring pipeline"""
    # Create pipeline config for activity monitoring
    activity_config = {
        "name": "activity_monitoring_pipeline",
        "description": "Pipeline for activity data monitoring",
        "version": "1.0.0",
        "steps": [
            {
                "name": "data_validation",
                "type": "validation",
                "config": {
                    "validation_type": "activity",
                    "validation_level": "strict",
                    "metrics": ["mining_activity", "staking_activity", "trading_activity"]
                }
            },
            {
                "name": "activity_check",
                "type": "monitoring",
                "config": {
                    "monitoring_type": "activity",
                    "thresholds": {
                        "mining_activity": 0.7,
                        "staking_activity": 0.8,
                        "trading_activity": 0.6
                    },
                    "window": "1H"
                }
            },
            {
                "name": "pattern_analysis",
                "type": "analysis",
                "config": {
                    "analysis_type": "pattern",
                    "method": "sequence_mining",
                    "metrics": ["mining_activity", "staking_activity", "trading_activity"]
                }
            },
            {
                "name": "alert_generation",
                "type": "notification",
                "config": {
                    "alert_type": "activity_alert",
                    "channels": ["email"],
                    "conditions": {
                        "mining_activity": {"operator": "<", "value": 0.7},
                        "trading_activity": {"operator": "<", "value": 0.6}
                    }
                }
            }
        ],
        "dependencies": {
            "data_validation": [],
            "activity_check": ["data_validation"],
            "pattern_analysis": ["data_validation"],
            "alert_generation": ["activity_check", "pattern_analysis"]
        },
        "error_handling": {
            "retry": {
                "enabled": True,
                "max_attempts": 3,
                "delay": 300  # 5 minutes
            },
            "fallback": {
                "enabled": True,
                "steps": ["data_validation"]
            }
        },
        "notification": {
            "enabled": True,
            "channels": ["email"],
            "recipients": ["test@example.com"],
            "on_success": True,
            "on_failure": True
        },
        "metadata": {
            "tags": ["activity", "monitoring", "real-time"],
            "priority": "high",
            "owner": "test_user"
        }
    }
    
    # Create pipeline
    result = create_pipeline(
        user_id=test_user.user_id,
        pipeline_config=activity_config,
        db_session=db_session
    )
    
    # Verify pipeline result
    assert isinstance(result, Dict)
    assert "pipeline_id" in result
    assert "pipeline_details" in result
    
    # Verify pipeline metadata
    assert result["name"] == "activity_monitoring_pipeline"
    assert result["description"] == "Pipeline for activity data monitoring"
    assert result["version"] == "1.0.0"
    
    # Verify pipeline steps
    assert "steps" in result
    assert len(result["steps"]) == 4
    step_names = [step["name"] for step in result["steps"]]
    assert all(name in step_names for name in ["data_validation", "activity_check", "pattern_analysis", "alert_generation"])
    
    # Verify dependencies
    assert "dependencies" in result
    dependencies = result["dependencies"]
    assert "data_validation" in dependencies
    assert "activity_check" in dependencies
    assert "pattern_analysis" in dependencies
    assert "alert_generation" in dependencies
    assert dependencies["data_validation"] == []
    assert "data_validation" in dependencies["activity_check"]
    assert "data_validation" in dependencies["pattern_analysis"]
    assert all(step in dependencies["alert_generation"] for step in ["activity_check", "pattern_analysis"])
    
    # Verify error handling
    assert "error_handling" in result
    error_handling = result["error_handling"]
    assert "retry" in error_handling
    assert "fallback" in error_handling
    assert error_handling["retry"]["enabled"] is True
    assert error_handling["retry"]["max_attempts"] == 3
    assert error_handling["retry"]["delay"] == 300
    assert error_handling["fallback"]["enabled"] is True
    assert "data_validation" in error_handling["fallback"]["steps"]
    
    # Verify notification settings
    assert "notification" in result
    notification = result["notification"]
    assert notification["enabled"] is True
    assert "email" in notification["channels"]
    assert "test@example.com" in notification["recipients"]
    assert notification["on_success"] is True
    assert notification["on_failure"] is True
    
    # Verify metadata
    assert "metadata" in result
    metadata = result["metadata"]
    assert all(tag in metadata["tags"] for tag in ["activity", "monitoring", "real-time"])
    assert metadata["priority"] == "high"
    assert metadata["owner"] == "test_user"
    
    # Verify database entry
    db_record = db_session.query(PipelineRecord).filter_by(
        user_id=test_user.user_id,
        pipeline_id=result["pipeline_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_execute_pipeline(db_session, test_user, test_performance_data, test_pipeline_config):
    """Test executing a pipeline"""
    # First, create a pipeline
    pipeline_result = create_pipeline(
        user_id=test_user.user_id,
        pipeline_config=test_pipeline_config,
        db_session=db_session
    )
    
    pipeline_id = pipeline_result["pipeline_id"]
    
    # Execute pipeline
    result = execute_pipeline(
        user_id=test_user.user_id,
        pipeline_id=pipeline_id,
        input_data=test_performance_data,
        db_session=db_session
    )
    
    # Verify execution result
    assert isinstance(result, Dict)
    assert "execution_id" in result
    assert "pipeline_id" in result
    assert result["pipeline_id"] == pipeline_id
    assert "status" in result
    assert "start_time" in result
    assert "end_time" in result
    assert "duration" in result
    
    # Verify step results
    assert "step_results" in result
    step_results = result["step_results"]
    assert len(step_results) == 4
    step_names = [step["name"] for step in step_results]
    assert all(name in step_names for name in ["data_validation", "data_normalization", "trend_analysis", "report_generation"])
    
    # Verify each step result
    for step in step_results:
        assert "name" in step
        assert "status" in step
        assert "start_time" in step
        assert "end_time" in step
        assert "duration" in step
        assert step["status"] == "COMPLETED"
        assert "output" in step
    
    # Verify database entry
    db_record = db_session.query(PipelineExecutionRecord).filter_by(
        user_id=test_user.user_id,
        pipeline_id=pipeline_id,
        execution_id=result["execution_id"]
    ).first()
    assert db_record is not None
    assert db_record.status == "COMPLETED"
    assert db_record.error is None

def test_update_pipeline(db_session, test_user, test_performance_data, test_pipeline_config):
    """Test updating a pipeline"""
    # First, create a pipeline
    pipeline_result = create_pipeline(
        user_id=test_user.user_id,
        pipeline_config=test_pipeline_config,
        db_session=db_session
    )
    
    pipeline_id = pipeline_result["pipeline_id"]
    
    # Create updated pipeline config
    updated_config = test_pipeline_config.copy()
    updated_config["version"] = "1.1.0"
    updated_config["steps"][2]["config"]["window_size"] = 14  # Change window size
    updated_config["notification"]["recipients"].append("updated@example.com")
    
    # Update pipeline
    result = update_pipeline(
        user_id=test_user.user_id,
        pipeline_id=pipeline_id,
        pipeline_config=updated_config,
        db_session=db_session
    )
    
    # Verify update result
    assert isinstance(result, Dict)
    assert "pipeline_id" in result
    assert result["pipeline_id"] == pipeline_id
    assert "pipeline_details" in result
    
    # Verify updated pipeline
    assert result["version"] == "1.1.0"
    assert result["steps"][2]["config"]["window_size"] == 14
    assert "updated@example.com" in result["notification"]["recipients"]
    
    # Verify database entry
    db_record = db_session.query(PipelineRecord).filter_by(
        user_id=test_user.user_id,
        pipeline_id=pipeline_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_delete_pipeline(db_session, test_user, test_performance_data, test_pipeline_config):
    """Test deleting a pipeline"""
    # First, create a pipeline
    pipeline_result = create_pipeline(
        user_id=test_user.user_id,
        pipeline_config=test_pipeline_config,
        db_session=db_session
    )
    
    pipeline_id = pipeline_result["pipeline_id"]
    
    # Delete pipeline
    result = delete_pipeline(
        user_id=test_user.user_id,
        pipeline_id=pipeline_id,
        db_session=db_session
    )
    
    # Verify delete result
    assert isinstance(result, Dict)
    assert "pipeline_id" in result
    assert result["pipeline_id"] == pipeline_id
    assert "status" in result
    assert result["status"] == "DELETED"
    
    # Verify database entry
    db_record = db_session.query(PipelineRecord).filter_by(
        user_id=test_user.user_id,
        pipeline_id=pipeline_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is False
    assert db_record.error is None

def test_get_pipeline_info(db_session, test_user, test_performance_data, test_pipeline_config):
    """Test retrieving pipeline information"""
    # First, create a pipeline
    pipeline_result = create_pipeline(
        user_id=test_user.user_id,
        pipeline_config=test_pipeline_config,
        db_session=db_session
    )
    
    pipeline_id = pipeline_result["pipeline_id"]
    
    # Get pipeline info
    result = get_pipeline_info(
        user_id=test_user.user_id,
        pipeline_id=pipeline_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "pipeline_id" in result
    assert result["pipeline_id"] == pipeline_id
    
    # Verify pipeline metadata
    assert result["name"] == "performance_analysis_pipeline"
    assert result["description"] == "Pipeline for performance data analysis"
    assert result["version"] == "1.0.0"
    
    # Verify pipeline details
    assert "pipeline_details" in result
    assert isinstance(result["pipeline_details"], Dict)
    assert "created_at" in result["pipeline_details"]
    assert "last_modified" in result["pipeline_details"]
    assert "execution_count" in result["pipeline_details"]
    
    # Verify database entry
    db_record = db_session.query(PipelineRecord).filter_by(
        user_id=test_user.user_id,
        pipeline_id=pipeline_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_get_execution_history(db_session, test_user, test_performance_data, test_pipeline_config):
    """Test retrieving execution history"""
    # First, create a pipeline
    pipeline_result = create_pipeline(
        user_id=test_user.user_id,
        pipeline_config=test_pipeline_config,
        db_session=db_session
    )
    
    pipeline_id = pipeline_result["pipeline_id"]
    
    # Create mock execution records
    execution_records = []
    for i in range(5):
        execution_record = PipelineExecutionRecord(
            execution_id=str(uuid.uuid4()),
            user_id=test_user.user_id,
            pipeline_id=pipeline_id,
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
        pipeline_id=pipeline_id,
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

def test_pipeline_error_handling(db_session, test_user):
    """Test pipeline error handling"""
    # Invalid user ID
    with pytest.raises(PipelineError) as excinfo:
        create_pipeline(
            user_id=None,
            pipeline_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid pipeline name
    with pytest.raises(PipelineError) as excinfo:
        create_pipeline(
            user_id=test_user.user_id,
            pipeline_config={"name": ""},
            db_session=db_session
        )
    assert "Invalid pipeline name" in str(excinfo.value)
    
    # Invalid pipeline steps
    with pytest.raises(PipelineError) as excinfo:
        create_pipeline(
            user_id=test_user.user_id,
            pipeline_config={"name": "test_pipeline", "steps": []},
            db_session=db_session
        )
    assert "Invalid pipeline steps" in str(excinfo.value)
    
    # Invalid pipeline ID
    with pytest.raises(PipelineError) as excinfo:
        get_pipeline_info(
            user_id=test_user.user_id,
            pipeline_id="invalid_pipeline_id",
            db_session=db_session
        )
    assert "Invalid pipeline ID" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBPipelineError) as excinfo:
        create_pipeline(
            user_id=test_user.user_id,
            pipeline_config={"name": "test_pipeline"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 