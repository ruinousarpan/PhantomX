import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import uuid
from unittest.mock import patch, MagicMock

from core.orchestration import (
    create_workflow,
    execute_workflow,
    get_workflow_info,
    update_workflow,
    delete_workflow,
    get_execution_history,
    schedule_workflow,
    pause_workflow,
    resume_workflow,
    OrchestrationError
)
from database.models import User, WorkflowRecord, WorkflowExecutionRecord
from database.exceptions import OrchestrationError as DBOrchestrationError

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
def test_workflow_config():
    """Create test workflow configuration"""
    return {
        "name": "data_processing_workflow",
        "description": "End-to-end data processing workflow",
        "version": "1.0.0",
        "tasks": [
            {
                "id": "data_validation",
                "type": "validation",
                "config": {
                    "validation_type": "performance",
                    "validation_level": "strict",
                    "metrics": ["mining_performance", "staking_performance", "trading_performance"]
                },
                "retry": {
                    "max_attempts": 3,
                    "delay": 300
                }
            },
            {
                "id": "data_normalization",
                "type": "normalization",
                "config": {
                    "method": "min_max",
                    "columns": ["mining_performance", "staking_performance", "trading_performance"]
                }
            },
            {
                "id": "data_aggregation",
                "type": "aggregation",
                "config": {
                    "aggregation_type": "time_series",
                    "window": "1D",
                    "metrics": ["mining_performance", "staking_performance", "trading_performance"],
                    "functions": ["sum", "mean", "max"]
                }
            },
            {
                "id": "risk_analysis",
                "type": "analysis",
                "config": {
                    "analysis_type": "risk",
                    "metrics": ["mining_performance", "staking_performance", "trading_performance"],
                    "risk_metrics": ["var", "volatility", "sharpe_ratio"]
                }
            },
            {
                "id": "report_generation",
                "type": "reporting",
                "config": {
                    "report_type": "performance_summary",
                    "format": "pdf",
                    "sections": ["summary", "analysis", "recommendations"]
                }
            }
        ],
        "dependencies": {
            "data_validation": [],
            "data_normalization": ["data_validation"],
            "data_aggregation": ["data_normalization"],
            "risk_analysis": ["data_aggregation"],
            "report_generation": ["risk_analysis"]
        },
        "schedule": {
            "type": "cron",
            "expression": "0 0 * * *",  # Daily at midnight
            "timezone": "UTC",
            "start_date": "2024-01-01",
            "end_date": None
        },
        "parallelism": {
            "max_parallel_tasks": 3,
            "resource_limits": {
                "cpu": "2",
                "memory": "4Gi"
            }
        },
        "error_handling": {
            "retry": {
                "enabled": True,
                "max_attempts": 3,
                "delay": 300  # 5 minutes
            },
            "fallback": {
                "enabled": True,
                "tasks": ["data_validation", "data_normalization"]
            }
        },
        "notification": {
            "enabled": True,
            "channels": ["email"],
            "recipients": ["test@example.com"],
            "events": ["start", "complete", "fail"]
        },
        "metadata": {
            "tags": ["data-processing", "automated", "daily"],
            "priority": "high",
            "owner": "test_user",
            "department": "data-science"
        }
    }

def test_create_workflow(db_session, test_user, test_workflow_config):
    """Test creating a workflow"""
    # Create workflow
    result = create_workflow(
        user_id=test_user.user_id,
        workflow_config=test_workflow_config,
        db_session=db_session
    )
    
    # Verify workflow result
    assert isinstance(result, Dict)
    assert "workflow_id" in result
    assert "workflow_details" in result
    
    # Verify workflow metadata
    assert result["name"] == "data_processing_workflow"
    assert result["description"] == "End-to-end data processing workflow"
    assert result["version"] == "1.0.0"
    
    # Verify tasks
    assert "tasks" in result
    assert len(result["tasks"]) == 5
    task_ids = [task["id"] for task in result["tasks"]]
    assert all(task_id in task_ids for task_id in [
        "data_validation", "data_normalization", "data_aggregation",
        "risk_analysis", "report_generation"
    ])
    
    # Verify dependencies
    assert "dependencies" in result
    dependencies = result["dependencies"]
    assert "data_validation" in dependencies
    assert dependencies["data_validation"] == []
    assert "data_normalization" in dependencies
    assert "data_validation" in dependencies["data_normalization"]
    
    # Verify schedule
    assert "schedule" in result
    schedule = result["schedule"]
    assert schedule["type"] == "cron"
    assert schedule["expression"] == "0 0 * * *"
    assert schedule["timezone"] == "UTC"
    
    # Verify parallelism
    assert "parallelism" in result
    parallelism = result["parallelism"]
    assert parallelism["max_parallel_tasks"] == 3
    assert "resource_limits" in parallelism
    
    # Verify error handling
    assert "error_handling" in result
    error_handling = result["error_handling"]
    assert error_handling["retry"]["enabled"] is True
    assert error_handling["fallback"]["enabled"] is True
    
    # Verify notification settings
    assert "notification" in result
    notification = result["notification"]
    assert notification["enabled"] is True
    assert "email" in notification["channels"]
    assert "test@example.com" in notification["recipients"]
    
    # Verify metadata
    assert "metadata" in result
    metadata = result["metadata"]
    assert all(tag in metadata["tags"] for tag in ["data-processing", "automated", "daily"])
    assert metadata["priority"] == "high"
    assert metadata["owner"] == "test_user"
    
    # Verify database entry
    db_record = db_session.query(WorkflowRecord).filter_by(
        user_id=test_user.user_id,
        workflow_id=result["workflow_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_execute_workflow(db_session, test_user, test_workflow_config):
    """Test executing a workflow"""
    # First, create a workflow
    workflow_result = create_workflow(
        user_id=test_user.user_id,
        workflow_config=test_workflow_config,
        db_session=db_session
    )
    
    workflow_id = workflow_result["workflow_id"]
    
    # Create test input data
    input_data = pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="H"),
        "mining_performance": np.random.uniform(0.8, 0.9, 100),
        "staking_performance": np.random.uniform(0.85, 0.95, 100),
        "trading_performance": np.random.uniform(0.7, 0.8, 100)
    })
    
    # Execute workflow
    result = execute_workflow(
        user_id=test_user.user_id,
        workflow_id=workflow_id,
        input_data=input_data,
        db_session=db_session
    )
    
    # Verify execution result
    assert isinstance(result, Dict)
    assert "execution_id" in result
    assert "workflow_id" in result
    assert result["workflow_id"] == workflow_id
    assert "status" in result
    assert "start_time" in result
    assert "end_time" in result
    assert "duration" in result
    
    # Verify task results
    assert "task_results" in result
    task_results = result["task_results"]
    assert len(task_results) == 5
    
    # Verify each task result
    for task in task_results:
        assert "id" in task
        assert "status" in task
        assert "start_time" in task
        assert "end_time" in task
        assert "duration" in task
        assert task["status"] == "COMPLETED"
        assert "output" in task
    
    # Verify execution order follows dependencies
    execution_order = [task["id"] for task in task_results]
    assert execution_order.index("data_validation") < execution_order.index("data_normalization")
    assert execution_order.index("data_normalization") < execution_order.index("data_aggregation")
    assert execution_order.index("data_aggregation") < execution_order.index("risk_analysis")
    assert execution_order.index("risk_analysis") < execution_order.index("report_generation")
    
    # Verify database entry
    db_record = db_session.query(WorkflowExecutionRecord).filter_by(
        user_id=test_user.user_id,
        workflow_id=workflow_id,
        execution_id=result["execution_id"]
    ).first()
    assert db_record is not None
    assert db_record.status == "COMPLETED"
    assert db_record.error is None

def test_schedule_workflow(db_session, test_user, test_workflow_config):
    """Test scheduling a workflow"""
    # First, create a workflow
    workflow_result = create_workflow(
        user_id=test_user.user_id,
        workflow_config=test_workflow_config,
        db_session=db_session
    )
    
    workflow_id = workflow_result["workflow_id"]
    
    # Schedule workflow
    result = schedule_workflow(
        user_id=test_user.user_id,
        workflow_id=workflow_id,
        schedule_config={
            "type": "cron",
            "expression": "0 */4 * * *",  # Every 4 hours
            "timezone": "UTC",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31"
        },
        db_session=db_session
    )
    
    # Verify schedule result
    assert isinstance(result, Dict)
    assert "workflow_id" in result
    assert result["workflow_id"] == workflow_id
    assert "schedule_id" in result
    assert "next_execution" in result
    
    # Verify schedule details
    schedule = result["schedule"]
    assert schedule["type"] == "cron"
    assert schedule["expression"] == "0 */4 * * *"
    assert schedule["timezone"] == "UTC"
    assert schedule["start_date"] == "2024-01-01"
    assert schedule["end_date"] == "2024-12-31"
    
    # Verify database entry
    db_record = db_session.query(WorkflowRecord).filter_by(
        user_id=test_user.user_id,
        workflow_id=workflow_id
    ).first()
    assert db_record is not None
    assert db_record.is_scheduled is True
    assert db_record.error is None

def test_pause_resume_workflow(db_session, test_user, test_workflow_config):
    """Test pausing and resuming a workflow"""
    # First, create and schedule a workflow
    workflow_result = create_workflow(
        user_id=test_user.user_id,
        workflow_config=test_workflow_config,
        db_session=db_session
    )
    
    workflow_id = workflow_result["workflow_id"]
    
    schedule_result = schedule_workflow(
        user_id=test_user.user_id,
        workflow_id=workflow_id,
        schedule_config=test_workflow_config["schedule"],
        db_session=db_session
    )
    
    # Pause workflow
    pause_result = pause_workflow(
        user_id=test_user.user_id,
        workflow_id=workflow_id,
        db_session=db_session
    )
    
    # Verify pause result
    assert isinstance(pause_result, Dict)
    assert "workflow_id" in pause_result
    assert pause_result["workflow_id"] == workflow_id
    assert "status" in pause_result
    assert pause_result["status"] == "PAUSED"
    
    # Verify database entry after pause
    db_record = db_session.query(WorkflowRecord).filter_by(
        user_id=test_user.user_id,
        workflow_id=workflow_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.is_paused is True
    
    # Resume workflow
    resume_result = resume_workflow(
        user_id=test_user.user_id,
        workflow_id=workflow_id,
        db_session=db_session
    )
    
    # Verify resume result
    assert isinstance(resume_result, Dict)
    assert "workflow_id" in resume_result
    assert resume_result["workflow_id"] == workflow_id
    assert "status" in resume_result
    assert resume_result["status"] == "ACTIVE"
    
    # Verify database entry after resume
    db_record = db_session.query(WorkflowRecord).filter_by(
        user_id=test_user.user_id,
        workflow_id=workflow_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.is_paused is False

def test_update_workflow(db_session, test_user, test_workflow_config):
    """Test updating a workflow"""
    # First, create a workflow
    workflow_result = create_workflow(
        user_id=test_user.user_id,
        workflow_config=test_workflow_config,
        db_session=db_session
    )
    
    workflow_id = workflow_result["workflow_id"]
    
    # Create updated workflow config
    updated_config = test_workflow_config.copy()
    updated_config["version"] = "1.1.0"
    updated_config["tasks"].append({
        "id": "data_export",
        "type": "export",
        "config": {
            "format": "parquet",
            "compression": "snappy",
            "partition_by": ["day"]
        }
    })
    updated_config["dependencies"]["data_export"] = ["report_generation"]
    updated_config["notification"]["recipients"].append("updated@example.com")
    
    # Update workflow
    result = update_workflow(
        user_id=test_user.user_id,
        workflow_id=workflow_id,
        workflow_config=updated_config,
        db_session=db_session
    )
    
    # Verify update result
    assert isinstance(result, Dict)
    assert "workflow_id" in result
    assert result["workflow_id"] == workflow_id
    assert "workflow_details" in result
    
    # Verify updated workflow
    assert result["version"] == "1.1.0"
    assert len(result["tasks"]) == 6
    assert "data_export" in [task["id"] for task in result["tasks"]]
    assert "data_export" in result["dependencies"]
    assert "report_generation" in result["dependencies"]["data_export"]
    assert "updated@example.com" in result["notification"]["recipients"]
    
    # Verify database entry
    db_record = db_session.query(WorkflowRecord).filter_by(
        user_id=test_user.user_id,
        workflow_id=workflow_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_delete_workflow(db_session, test_user, test_workflow_config):
    """Test deleting a workflow"""
    # First, create a workflow
    workflow_result = create_workflow(
        user_id=test_user.user_id,
        workflow_config=test_workflow_config,
        db_session=db_session
    )
    
    workflow_id = workflow_result["workflow_id"]
    
    # Delete workflow
    result = delete_workflow(
        user_id=test_user.user_id,
        workflow_id=workflow_id,
        db_session=db_session
    )
    
    # Verify delete result
    assert isinstance(result, Dict)
    assert "workflow_id" in result
    assert result["workflow_id"] == workflow_id
    assert "status" in result
    assert result["status"] == "DELETED"
    
    # Verify database entry
    db_record = db_session.query(WorkflowRecord).filter_by(
        user_id=test_user.user_id,
        workflow_id=workflow_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is False
    assert db_record.error is None

def test_get_workflow_info(db_session, test_user, test_workflow_config):
    """Test retrieving workflow information"""
    # First, create a workflow
    workflow_result = create_workflow(
        user_id=test_user.user_id,
        workflow_config=test_workflow_config,
        db_session=db_session
    )
    
    workflow_id = workflow_result["workflow_id"]
    
    # Get workflow info
    result = get_workflow_info(
        user_id=test_user.user_id,
        workflow_id=workflow_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "workflow_id" in result
    assert result["workflow_id"] == workflow_id
    
    # Verify workflow metadata
    assert result["name"] == "data_processing_workflow"
    assert result["description"] == "End-to-end data processing workflow"
    assert result["version"] == "1.0.0"
    
    # Verify workflow details
    assert "workflow_details" in result
    assert isinstance(result["workflow_details"], Dict)
    assert "created_at" in result["workflow_details"]
    assert "last_modified" in result["workflow_details"]
    assert "execution_count" in result["workflow_details"]
    
    # Verify database entry
    db_record = db_session.query(WorkflowRecord).filter_by(
        user_id=test_user.user_id,
        workflow_id=workflow_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_get_execution_history(db_session, test_user, test_workflow_config):
    """Test retrieving execution history"""
    # First, create a workflow
    workflow_result = create_workflow(
        user_id=test_user.user_id,
        workflow_config=test_workflow_config,
        db_session=db_session
    )
    
    workflow_id = workflow_result["workflow_id"]
    
    # Create mock execution records
    execution_records = []
    for i in range(5):
        execution_record = WorkflowExecutionRecord(
            execution_id=str(uuid.uuid4()),
            user_id=test_user.user_id,
            workflow_id=workflow_id,
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
        workflow_id=workflow_id,
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

def test_orchestration_error_handling(db_session, test_user):
    """Test orchestration error handling"""
    # Invalid user ID
    with pytest.raises(OrchestrationError) as excinfo:
        create_workflow(
            user_id=None,
            workflow_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid workflow name
    with pytest.raises(OrchestrationError) as excinfo:
        create_workflow(
            user_id=test_user.user_id,
            workflow_config={"name": ""},
            db_session=db_session
        )
    assert "Invalid workflow name" in str(excinfo.value)
    
    # Invalid workflow tasks
    with pytest.raises(OrchestrationError) as excinfo:
        create_workflow(
            user_id=test_user.user_id,
            workflow_config={"name": "test_workflow", "tasks": []},
            db_session=db_session
        )
    assert "Invalid workflow tasks" in str(excinfo.value)
    
    # Invalid workflow ID
    with pytest.raises(OrchestrationError) as excinfo:
        get_workflow_info(
            user_id=test_user.user_id,
            workflow_id="invalid_workflow_id",
            db_session=db_session
        )
    assert "Invalid workflow ID" in str(excinfo.value)
    
    # Invalid schedule configuration
    with pytest.raises(OrchestrationError) as excinfo:
        schedule_workflow(
            user_id=test_user.user_id,
            workflow_id="test_workflow_id",
            schedule_config={"type": "invalid"},
            db_session=db_session
        )
    assert "Invalid schedule configuration" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBOrchestrationError) as excinfo:
        create_workflow(
            user_id=test_user.user_id,
            workflow_config={"name": "test_workflow"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 