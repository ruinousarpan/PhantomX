import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import uuid
from unittest.mock import patch, MagicMock

from core.integration import (
    create_data_source,
    connect_data_source,
    ingest_data,
    transform_data,
    sync_data,
    get_source_info,
    update_data_source,
    delete_data_source,
    get_sync_history,
    monitor_integration,
    IntegrationError
)
from database.models import User, DataSourceRecord, SyncRecord
from database.exceptions import IntegrationError as DBIntegrationError

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
def test_data_source_config():
    """Create test data source configuration"""
    return {
        "name": "crypto_exchange_source",
        "description": "Cryptocurrency exchange data source",
        "type": "exchange_api",
        "version": "1.0.0",
        "connection": {
            "type": "rest_api",
            "url": "https://api.exchange.com/v1",
            "auth_type": "api_key",
            "credentials": {
                "api_key": "${API_KEY}",
                "api_secret": "${API_SECRET}"
            },
            "rate_limit": {
                "requests_per_second": 10,
                "burst_limit": 20
            }
        },
        "data_schema": {
            "type": "structured",
            "format": "json",
            "fields": [
                {
                    "name": "timestamp",
                    "type": "datetime",
                    "required": True
                },
                {
                    "name": "mining_performance",
                    "type": "float",
                    "required": True,
                    "validation": {
                        "min": 0.0,
                        "max": 1.0
                    }
                },
                {
                    "name": "staking_performance",
                    "type": "float",
                    "required": True,
                    "validation": {
                        "min": 0.0,
                        "max": 1.0
                    }
                },
                {
                    "name": "trading_performance",
                    "type": "float",
                    "required": True,
                    "validation": {
                        "min": 0.0,
                        "max": 1.0
                    }
                }
            ]
        },
        "ingestion": {
            "method": "pull",
            "schedule": {
                "type": "cron",
                "expression": "*/5 * * * *"  # Every 5 minutes
            },
            "batch_size": 1000,
            "retry": {
                "max_attempts": 3,
                "delay": 60  # 1 minute
            }
        },
        "transformation": {
            "steps": [
                {
                    "name": "timestamp_conversion",
                    "type": "datetime",
                    "config": {
                        "input_format": "unix_timestamp",
                        "output_format": "iso8601",
                        "timezone": "UTC"
                    }
                },
                {
                    "name": "performance_normalization",
                    "type": "normalization",
                    "config": {
                        "method": "min_max",
                        "columns": ["mining_performance", "staking_performance", "trading_performance"]
                    }
                }
            ]
        },
        "sync": {
            "strategy": "incremental",
            "key_field": "timestamp",
            "batch_size": 1000,
            "conflict_resolution": "newer_wins"
        },
        "monitoring": {
            "enabled": True,
            "metrics": [
                "latency",
                "throughput",
                "error_rate",
                "data_quality"
            ],
            "alerts": {
                "latency_threshold_ms": 1000,
                "error_rate_threshold": 0.01,
                "data_quality_threshold": 0.95
            }
        },
        "metadata": {
            "tags": ["crypto", "exchange", "performance"],
            "owner": "test_user",
            "department": "trading"
        }
    }

def test_create_data_source(db_session, test_user, test_data_source_config):
    """Test creating a data source"""
    # Create data source
    result = create_data_source(
        user_id=test_user.user_id,
        source_config=test_data_source_config,
        db_session=db_session
    )
    
    # Verify result
    assert isinstance(result, Dict)
    assert "source_id" in result
    assert "source_details" in result
    
    # Verify source metadata
    assert result["name"] == "crypto_exchange_source"
    assert result["description"] == "Cryptocurrency exchange data source"
    assert result["type"] == "exchange_api"
    assert result["version"] == "1.0.0"
    
    # Verify connection config
    assert "connection" in result
    connection = result["connection"]
    assert connection["type"] == "rest_api"
    assert connection["url"] == "https://api.exchange.com/v1"
    assert connection["auth_type"] == "api_key"
    assert "rate_limit" in connection
    
    # Verify data schema
    assert "data_schema" in result
    schema = result["data_schema"]
    assert schema["type"] == "structured"
    assert schema["format"] == "json"
    assert len(schema["fields"]) == 4
    
    # Verify ingestion config
    assert "ingestion" in result
    ingestion = result["ingestion"]
    assert ingestion["method"] == "pull"
    assert ingestion["batch_size"] == 1000
    
    # Verify transformation config
    assert "transformation" in result
    transformation = result["transformation"]
    assert len(transformation["steps"]) == 2
    
    # Verify sync config
    assert "sync" in result
    sync = result["sync"]
    assert sync["strategy"] == "incremental"
    assert sync["key_field"] == "timestamp"
    
    # Verify monitoring config
    assert "monitoring" in result
    monitoring = result["monitoring"]
    assert monitoring["enabled"] is True
    assert len(monitoring["metrics"]) == 4
    
    # Verify database entry
    db_record = db_session.query(DataSourceRecord).filter_by(
        user_id=test_user.user_id,
        source_id=result["source_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_connect_data_source(db_session, test_user, test_data_source_config):
    """Test connecting to a data source"""
    # First, create a data source
    source_result = create_data_source(
        user_id=test_user.user_id,
        source_config=test_data_source_config,
        db_session=db_session
    )
    
    source_id = source_result["source_id"]
    
    # Mock API response
    mock_response = {
        "status": "connected",
        "connection_id": str(uuid.uuid4()),
        "capabilities": ["read", "write", "delete"],
        "rate_limits": {
            "remaining": 9999,
            "reset_at": (datetime.utcnow() + timedelta(hours=1)).isoformat()
        }
    }
    
    # Connect to data source
    with patch('core.integration.connect_to_api', return_value=mock_response):
        result = connect_data_source(
            user_id=test_user.user_id,
            source_id=source_id,
            db_session=db_session
        )
    
    # Verify connection result
    assert isinstance(result, Dict)
    assert "source_id" in result
    assert result["source_id"] == source_id
    assert "status" in result
    assert result["status"] == "connected"
    assert "connection_id" in result
    assert "capabilities" in result
    assert "rate_limits" in result
    
    # Verify database entry
    db_record = db_session.query(DataSourceRecord).filter_by(
        user_id=test_user.user_id,
        source_id=source_id
    ).first()
    assert db_record is not None
    assert db_record.is_connected is True
    assert db_record.error is None

def test_ingest_data(db_session, test_user, test_data_source_config):
    """Test ingesting data from a source"""
    # First, create and connect to a data source
    source_result = create_data_source(
        user_id=test_user.user_id,
        source_config=test_data_source_config,
        db_session=db_session
    )
    
    source_id = source_result["source_id"]
    
    # Mock API data
    mock_data = pd.DataFrame({
        "timestamp": range(1640995200, 1641081600, 300),  # 5-minute intervals
        "mining_performance": np.random.uniform(0.8, 0.9, 288),
        "staking_performance": np.random.uniform(0.85, 0.95, 288),
        "trading_performance": np.random.uniform(0.7, 0.8, 288)
    })
    
    # Ingest data
    with patch('core.integration.fetch_data', return_value=mock_data):
        result = ingest_data(
            user_id=test_user.user_id,
            source_id=source_id,
            start_time="2022-01-01T00:00:00Z",
            end_time="2022-01-02T00:00:00Z",
            db_session=db_session
        )
    
    # Verify ingestion result
    assert isinstance(result, Dict)
    assert "source_id" in result
    assert result["source_id"] == source_id
    assert "ingestion_id" in result
    assert "status" in result
    assert "stats" in result
    
    # Verify ingestion stats
    stats = result["stats"]
    assert stats["total_records"] == 288
    assert stats["processed_records"] == 288
    assert stats["failed_records"] == 0
    assert "duration" in stats
    
    # Verify data quality
    assert "data_quality" in result
    quality = result["data_quality"]
    assert quality["completeness"] > 0.95
    assert quality["validity"] > 0.95
    assert quality["consistency"] > 0.95

def test_transform_data(db_session, test_user, test_data_source_config):
    """Test transforming ingested data"""
    # First, create a data source and ingest data
    source_result = create_data_source(
        user_id=test_user.user_id,
        source_config=test_data_source_config,
        db_session=db_session
    )
    
    source_id = source_result["source_id"]
    
    # Mock ingested data
    mock_data = pd.DataFrame({
        "timestamp": range(1640995200, 1641081600, 300),
        "mining_performance": np.random.uniform(0.8, 0.9, 288),
        "staking_performance": np.random.uniform(0.85, 0.95, 288),
        "trading_performance": np.random.uniform(0.7, 0.8, 288)
    })
    
    # Transform data
    result = transform_data(
        user_id=test_user.user_id,
        source_id=source_id,
        data=mock_data,
        db_session=db_session
    )
    
    # Verify transformation result
    assert isinstance(result, Dict)
    assert "source_id" in result
    assert result["source_id"] == source_id
    assert "transformation_id" in result
    assert "status" in result
    assert "data" in result
    
    # Verify transformed data
    transformed_data = result["data"]
    assert isinstance(transformed_data, pd.DataFrame)
    assert "timestamp" in transformed_data.columns
    assert all(col in transformed_data.columns for col in [
        "mining_performance", "staking_performance", "trading_performance"
    ])
    
    # Verify timestamps are in ISO8601 format
    assert pd.api.types.is_datetime64_any_dtype(transformed_data["timestamp"])
    
    # Verify performance metrics are normalized
    for col in ["mining_performance", "staking_performance", "trading_performance"]:
        assert transformed_data[col].min() >= 0.0
        assert transformed_data[col].max() <= 1.0

def test_sync_data(db_session, test_user, test_data_source_config):
    """Test synchronizing transformed data"""
    # First, create a data source and transform data
    source_result = create_data_source(
        user_id=test_user.user_id,
        source_config=test_data_source_config,
        db_session=db_session
    )
    
    source_id = source_result["source_id"]
    
    # Mock transformed data
    mock_data = pd.DataFrame({
        "timestamp": pd.date_range(start="2022-01-01", periods=288, freq="5min"),
        "mining_performance": np.random.uniform(0.8, 0.9, 288),
        "staking_performance": np.random.uniform(0.85, 0.95, 288),
        "trading_performance": np.random.uniform(0.7, 0.8, 288)
    })
    
    # Sync data
    result = sync_data(
        user_id=test_user.user_id,
        source_id=source_id,
        data=mock_data,
        db_session=db_session
    )
    
    # Verify sync result
    assert isinstance(result, Dict)
    assert "source_id" in result
    assert result["source_id"] == source_id
    assert "sync_id" in result
    assert "status" in result
    assert "stats" in result
    
    # Verify sync stats
    stats = result["stats"]
    assert stats["total_records"] == 288
    assert stats["synced_records"] == 288
    assert stats["conflict_records"] == 0
    assert stats["failed_records"] == 0
    assert "duration" in stats
    
    # Verify database entry
    db_record = db_session.query(SyncRecord).filter_by(
        user_id=test_user.user_id,
        source_id=source_id,
        sync_id=result["sync_id"]
    ).first()
    assert db_record is not None
    assert db_record.status == "COMPLETED"
    assert db_record.error is None

def test_monitor_integration(db_session, test_user, test_data_source_config):
    """Test monitoring integration metrics"""
    # First, create a data source and perform operations
    source_result = create_data_source(
        user_id=test_user.user_id,
        source_config=test_data_source_config,
        db_session=db_session
    )
    
    source_id = source_result["source_id"]
    
    # Monitor integration
    result = monitor_integration(
        user_id=test_user.user_id,
        source_id=source_id,
        db_session=db_session
    )
    
    # Verify monitoring result
    assert isinstance(result, Dict)
    assert "source_id" in result
    assert result["source_id"] == source_id
    assert "metrics" in result
    
    # Verify metrics
    metrics = result["metrics"]
    assert "latency" in metrics
    assert "throughput" in metrics
    assert "error_rate" in metrics
    assert "data_quality" in metrics
    
    # Verify alerts
    assert "alerts" in result
    alerts = result["alerts"]
    assert isinstance(alerts, List)
    for alert in alerts:
        assert "type" in alert
        assert "threshold" in alert
        assert "current_value" in alert
        assert "status" in alert

def test_update_data_source(db_session, test_user, test_data_source_config):
    """Test updating a data source"""
    # First, create a data source
    source_result = create_data_source(
        user_id=test_user.user_id,
        source_config=test_data_source_config,
        db_session=db_session
    )
    
    source_id = source_result["source_id"]
    
    # Create updated config
    updated_config = test_data_source_config.copy()
    updated_config["version"] = "1.1.0"
    updated_config["ingestion"]["batch_size"] = 2000
    updated_config["monitoring"]["alerts"]["latency_threshold_ms"] = 2000
    
    # Update data source
    result = update_data_source(
        user_id=test_user.user_id,
        source_id=source_id,
        source_config=updated_config,
        db_session=db_session
    )
    
    # Verify update result
    assert isinstance(result, Dict)
    assert "source_id" in result
    assert result["source_id"] == source_id
    assert "source_details" in result
    
    # Verify updated config
    assert result["version"] == "1.1.0"
    assert result["ingestion"]["batch_size"] == 2000
    assert result["monitoring"]["alerts"]["latency_threshold_ms"] == 2000
    
    # Verify database entry
    db_record = db_session.query(DataSourceRecord).filter_by(
        user_id=test_user.user_id,
        source_id=source_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_delete_data_source(db_session, test_user, test_data_source_config):
    """Test deleting a data source"""
    # First, create a data source
    source_result = create_data_source(
        user_id=test_user.user_id,
        source_config=test_data_source_config,
        db_session=db_session
    )
    
    source_id = source_result["source_id"]
    
    # Delete data source
    result = delete_data_source(
        user_id=test_user.user_id,
        source_id=source_id,
        db_session=db_session
    )
    
    # Verify delete result
    assert isinstance(result, Dict)
    assert "source_id" in result
    assert result["source_id"] == source_id
    assert "status" in result
    assert result["status"] == "DELETED"
    
    # Verify database entry
    db_record = db_session.query(DataSourceRecord).filter_by(
        user_id=test_user.user_id,
        source_id=source_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is False
    assert db_record.error is None

def test_get_source_info(db_session, test_user, test_data_source_config):
    """Test retrieving data source information"""
    # First, create a data source
    source_result = create_data_source(
        user_id=test_user.user_id,
        source_config=test_data_source_config,
        db_session=db_session
    )
    
    source_id = source_result["source_id"]
    
    # Get source info
    result = get_source_info(
        user_id=test_user.user_id,
        source_id=source_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "source_id" in result
    assert result["source_id"] == source_id
    
    # Verify source metadata
    assert result["name"] == "crypto_exchange_source"
    assert result["description"] == "Cryptocurrency exchange data source"
    assert result["type"] == "exchange_api"
    assert result["version"] == "1.0.0"
    
    # Verify source details
    assert "source_details" in result
    assert isinstance(result["source_details"], Dict)
    assert "created_at" in result["source_details"]
    assert "last_modified" in result["source_details"]
    assert "last_synced" in result["source_details"]
    
    # Verify database entry
    db_record = db_session.query(DataSourceRecord).filter_by(
        user_id=test_user.user_id,
        source_id=source_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_get_sync_history(db_session, test_user, test_data_source_config):
    """Test retrieving sync history"""
    # First, create a data source
    source_result = create_data_source(
        user_id=test_user.user_id,
        source_config=test_data_source_config,
        db_session=db_session
    )
    
    source_id = source_result["source_id"]
    
    # Create mock sync records
    sync_records = []
    for i in range(5):
        sync_record = SyncRecord(
            sync_id=str(uuid.uuid4()),
            user_id=test_user.user_id,
            source_id=source_id,
            status="COMPLETED" if i % 2 == 0 else "FAILED",
            started_at=datetime.utcnow() - timedelta(hours=i),
            completed_at=datetime.utcnow() - timedelta(hours=i) + timedelta(minutes=5),
            error=None if i % 2 == 0 else "Test error",
            stats={
                "total_records": 1000,
                "synced_records": 1000 if i % 2 == 0 else 500,
                "failed_records": 0 if i % 2 == 0 else 500
            }
        )
        sync_records.append(sync_record)
        db_session.add(sync_record)
    
    db_session.commit()
    
    # Get sync history
    result = get_sync_history(
        user_id=test_user.user_id,
        source_id=source_id,
        db_session=db_session
    )
    
    # Verify history result
    assert isinstance(result, List)
    assert len(result) == 5
    
    # Verify sync records
    for i, record in enumerate(result):
        assert "sync_id" in record
        assert "status" in record
        assert "started_at" in record
        assert "completed_at" in record
        assert "duration" in record
        assert "stats" in record
        
        if i % 2 == 0:
            assert record["status"] == "COMPLETED"
            assert record["error"] is None
            assert record["stats"]["synced_records"] == 1000
            assert record["stats"]["failed_records"] == 0
        else:
            assert record["status"] == "FAILED"
            assert record["error"] == "Test error"
            assert record["stats"]["synced_records"] == 500
            assert record["stats"]["failed_records"] == 500

def test_integration_error_handling(db_session, test_user):
    """Test integration error handling"""
    # Invalid user ID
    with pytest.raises(IntegrationError) as excinfo:
        create_data_source(
            user_id=None,
            source_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid source name
    with pytest.raises(IntegrationError) as excinfo:
        create_data_source(
            user_id=test_user.user_id,
            source_config={"name": ""},
            db_session=db_session
        )
    assert "Invalid source name" in str(excinfo.value)
    
    # Invalid source type
    with pytest.raises(IntegrationError) as excinfo:
        create_data_source(
            user_id=test_user.user_id,
            source_config={"name": "test_source", "type": "invalid"},
            db_session=db_session
        )
    assert "Invalid source type" in str(excinfo.value)
    
    # Invalid source ID
    with pytest.raises(IntegrationError) as excinfo:
        get_source_info(
            user_id=test_user.user_id,
            source_id="invalid_source_id",
            db_session=db_session
        )
    assert "Invalid source ID" in str(excinfo.value)
    
    # Invalid connection configuration
    with pytest.raises(IntegrationError) as excinfo:
        create_data_source(
            user_id=test_user.user_id,
            source_config={
                "name": "test_source",
                "type": "exchange_api",
                "connection": {"type": "invalid"}
            },
            db_session=db_session
        )
    assert "Invalid connection configuration" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBIntegrationError) as excinfo:
        create_data_source(
            user_id=test_user.user_id,
            source_config={"name": "test_source"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 