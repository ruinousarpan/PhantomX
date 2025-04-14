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
import csv
import xlsxwriter
from openpyxl import load_workbook
import pyarrow as pa
import pyarrow.parquet as pq
from unittest.mock import patch, MagicMock

from core.export import (
    export_data,
    get_export_info,
    ExportError,
    create_export,
    schedule_export,
    transform_data,
    validate_export,
    track_export_history,
    list_exports,
    delete_export
)
from database.models import User, ExportRecord, ExportSchedule, ExportHistory
from database.exceptions import ExportError as DBExportError

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
def test_export_config():
    """Create test export configuration"""
    return {
        "export_formats": {
            "enabled": True,
            "formats": ["csv", "json", "excel", "parquet", "avro"],
            "compression": ["none", "gzip", "zip", "snappy"],
            "encoding": ["utf-8", "utf-16", "ascii"],
            "date_format": "ISO8601",
            "file_options": {
                "csv": {
                    "separator": ",",
                    "quote_char": '"',
                    "header": True,
                    "index": False
                },
                "json": {
                    "orient": "records",
                    "lines": False,
                    "indent": 2
                },
                "excel": {
                    "sheet_name": "Data",
                    "index": False,
                    "engine": "openpyxl"
                }
            }
        },
        "export_scheduling": {
            "enabled": True,
            "schedule_types": ["once", "hourly", "daily", "weekly", "monthly"],
            "max_retries": 3,
            "retry_delay": 300,
            "timeout": 3600,
            "concurrent_exports": 5
        },
        "data_transformation": {
            "enabled": True,
            "operations": ["filter", "sort", "aggregate", "pivot", "merge"],
            "column_operations": ["rename", "cast", "derive", "drop"],
            "aggregation_functions": ["sum", "mean", "min", "max", "count"],
            "filter_operators": ["eq", "ne", "gt", "lt", "ge", "le", "in", "not_in"]
        },
        "export_validation": {
            "enabled": True,
            "schema_validation": True,
            "data_quality_checks": True,
            "size_limits": True,
            "format_validation": True,
            "max_file_size": 1073741824,  # 1GB
            "max_rows": 1000000
        },
        "history_tracking": {
            "enabled": True,
            "track_metadata": True,
            "track_performance": True,
            "track_errors": True,
            "retention_period": 90,
            "max_history_records": 1000
        }
    }

def test_export_performance_data_csv(db_session, test_user, test_performance_data, test_export_config):
    """Test exporting performance data to CSV"""
    # Export performance data
    result = export_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        export_config=test_export_config["export_formats"],
        db_session=db_session
    )
    
    # Verify export result
    assert isinstance(result, Dict)
    assert "export_id" in result
    assert "export_data" in result
    assert "export_details" in result
    
    # Verify export metadata
    assert result["export_type"] == "performance"
    assert result["format"] == "csv"
    assert "options" in result
    assert "compression" in result
    assert "metadata" in result
    
    # Verify data columns
    assert "data_columns" in result
    assert all(col in result["data_columns"] for col in test_export_config["export_formats"]["file_options"]["csv"]["columns"])
    
    # Verify export options
    options = result["options"]
    assert options["separator"] == ","
    assert options["quote_char"] == '"'
    assert options["header"] is True
    assert options["index"] is False
    assert options["date_format"] == "ISO8601"
    
    # Verify compression settings
    compression = result["compression"]
    assert compression["enabled"] is False
    assert compression["type"] is None
    
    # Verify metadata
    metadata = result["metadata"]
    assert metadata["title"] == "Performance Data Export"
    assert metadata["description"] == "Export of performance metrics"
    assert metadata["author"] == "Test User"
    assert "created_at" in metadata
    
    # Verify export data
    assert "file_data" in result["export_data"]
    assert "file_size" in result["export_data"]
    assert "checksum" in result["export_data"]
    
    # Verify database entry
    db_record = db_session.query(ExportRecord).filter_by(
        user_id=test_user.user_id,
        export_id=result["export_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_export_risk_data_excel(db_session, test_user, test_risk_data):
    """Test exporting risk data to Excel"""
    # Create export config for risk data
    risk_config = {
        "export_type": "risk",
        "format": "excel",
        "data_columns": ["mining_risk", "staking_risk", "trading_risk", "overall_risk"],
        "time_column": "timestamp",
        "options": {
            "sheet_name": "Risk Metrics",
            "index": False,
            "date_format": "%Y-%m-%d %H:%M:%S"
        },
        "compression": {
            "enabled": False,
            "type": None
        },
        "metadata": {
            "title": "Risk Data Export",
            "description": "Export of risk metrics",
            "author": "Test User",
            "created_at": datetime.utcnow().isoformat()
        }
    }
    
    # Export risk data
    result = export_data(
        user_id=test_user.user_id,
        data=test_risk_data,
        export_config=risk_config,
        db_session=db_session
    )
    
    # Verify export result
    assert isinstance(result, Dict)
    assert "export_id" in result
    assert "export_data" in result
    assert "export_details" in result
    
    # Verify export metadata
    assert result["export_type"] == "risk"
    assert result["format"] == "excel"
    assert "options" in result
    assert "compression" in result
    assert "metadata" in result
    
    # Verify data columns
    assert "data_columns" in result
    assert all(col in result["data_columns"] for col in risk_config["data_columns"])
    
    # Verify export options
    options = result["options"]
    assert options["sheet_name"] == "Risk Metrics"
    assert options["index"] is False
    assert options["date_format"] == "%Y-%m-%d %H:%M:%S"
    
    # Verify compression settings
    compression = result["compression"]
    assert compression["enabled"] is False
    assert compression["type"] is None
    
    # Verify metadata
    metadata = result["metadata"]
    assert metadata["title"] == "Risk Data Export"
    assert metadata["description"] == "Export of risk metrics"
    assert metadata["author"] == "Test User"
    assert "created_at" in metadata
    
    # Verify export data
    assert "file_data" in result["export_data"]
    assert "file_size" in result["export_data"]
    assert "checksum" in result["export_data"]
    
    # Verify database entry
    db_record = db_session.query(ExportRecord).filter_by(
        user_id=test_user.user_id,
        export_id=result["export_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_export_reward_data_json(db_session, test_user, test_reward_data):
    """Test exporting reward data to JSON"""
    # Create export config for reward data
    reward_config = {
        "export_type": "reward",
        "format": "json",
        "data_columns": ["mining_reward", "staking_reward", "trading_reward", "overall_reward"],
        "time_column": "timestamp",
        "options": {
            "orient": "records",
            "date_format": "iso",
            "indent": 2
        },
        "compression": {
            "enabled": False,
            "type": None
        },
        "metadata": {
            "title": "Reward Data Export",
            "description": "Export of reward metrics",
            "author": "Test User",
            "created_at": datetime.utcnow().isoformat()
        }
    }
    
    # Export reward data
    result = export_data(
        user_id=test_user.user_id,
        data=test_reward_data,
        export_config=reward_config,
        db_session=db_session
    )
    
    # Verify export result
    assert isinstance(result, Dict)
    assert "export_id" in result
    assert "export_data" in result
    assert "export_details" in result
    
    # Verify export metadata
    assert result["export_type"] == "reward"
    assert result["format"] == "json"
    assert "options" in result
    assert "compression" in result
    assert "metadata" in result
    
    # Verify data columns
    assert "data_columns" in result
    assert all(col in result["data_columns"] for col in reward_config["data_columns"])
    
    # Verify export options
    options = result["options"]
    assert options["orient"] == "records"
    assert options["date_format"] == "iso"
    assert options["indent"] == 2
    
    # Verify compression settings
    compression = result["compression"]
    assert compression["enabled"] is False
    assert compression["type"] is None
    
    # Verify metadata
    metadata = result["metadata"]
    assert metadata["title"] == "Reward Data Export"
    assert metadata["description"] == "Export of reward metrics"
    assert metadata["author"] == "Test User"
    assert "created_at" in metadata
    
    # Verify export data
    assert "file_data" in result["export_data"]
    assert "file_size" in result["export_data"]
    assert "checksum" in result["export_data"]
    
    # Verify database entry
    db_record = db_session.query(ExportRecord).filter_by(
        user_id=test_user.user_id,
        export_id=result["export_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_export_activity_data_parquet(db_session, test_user, test_activity_data):
    """Test exporting activity data to Parquet"""
    # Create export config for activity data
    activity_config = {
        "export_type": "activity",
        "format": "parquet",
        "data_columns": ["mining_activity", "staking_activity", "trading_activity", "overall_activity"],
        "time_column": "timestamp",
        "options": {
            "compression": "snappy",
            "row_group_size": 1000
        },
        "compression": {
            "enabled": False,
            "type": None
        },
        "metadata": {
            "title": "Activity Data Export",
            "description": "Export of activity metrics",
            "author": "Test User",
            "created_at": datetime.utcnow().isoformat()
        }
    }
    
    # Export activity data
    result = export_data(
        user_id=test_user.user_id,
        data=test_activity_data,
        export_config=activity_config,
        db_session=db_session
    )
    
    # Verify export result
    assert isinstance(result, Dict)
    assert "export_id" in result
    assert "export_data" in result
    assert "export_details" in result
    
    # Verify export metadata
    assert result["export_type"] == "activity"
    assert result["format"] == "parquet"
    assert "options" in result
    assert "compression" in result
    assert "metadata" in result
    
    # Verify data columns
    assert "data_columns" in result
    assert all(col in result["data_columns"] for col in activity_config["data_columns"])
    
    # Verify export options
    options = result["options"]
    assert options["compression"] == "snappy"
    assert options["row_group_size"] == 1000
    
    # Verify compression settings
    compression = result["compression"]
    assert compression["enabled"] is False
    assert compression["type"] is None
    
    # Verify metadata
    metadata = result["metadata"]
    assert metadata["title"] == "Activity Data Export"
    assert metadata["description"] == "Export of activity metrics"
    assert metadata["author"] == "Test User"
    assert "created_at" in metadata
    
    # Verify export data
    assert "file_data" in result["export_data"]
    assert "file_size" in result["export_data"]
    assert "checksum" in result["export_data"]
    
    # Verify database entry
    db_record = db_session.query(ExportRecord).filter_by(
        user_id=test_user.user_id,
        export_id=result["export_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_export_compressed_data(db_session, test_user, test_performance_data, test_export_config):
    """Test exporting compressed data"""
    # Modify export config to enable compression
    compressed_config = test_export_config["export_formats"].copy()
    compressed_config["compression"] = {
        "enabled": True,
        "type": "gzip"
    }
    
    # Export compressed data
    result = export_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        export_config=compressed_config,
        db_session=db_session
    )
    
    # Verify export result
    assert isinstance(result, Dict)
    assert "export_id" in result
    assert "export_data" in result
    assert "export_details" in result
    
    # Verify compression settings
    compression = result["compression"]
    assert compression["enabled"] is True
    assert compression["type"] == "gzip"
    
    # Verify export data
    assert "file_data" in result["export_data"]
    assert "file_size" in result["export_data"]
    assert "checksum" in result["export_data"]
    
    # Verify database entry
    db_record = db_session.query(ExportRecord).filter_by(
        user_id=test_user.user_id,
        export_id=result["export_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_export_filtered_data(db_session, test_user, test_performance_data, test_export_config):
    """Test exporting filtered data"""
    # Modify export config to include filtering
    filtered_config = test_export_config["export_formats"].copy()
    filtered_config["filter"] = {
        "column": "mining_performance",
        "operator": ">",
        "value": 0.85
    }
    
    # Export filtered data
    result = export_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        export_config=filtered_config,
        db_session=db_session
    )
    
    # Verify export result
    assert isinstance(result, Dict)
    assert "export_id" in result
    assert "export_data" in result
    assert "export_details" in result
    
    # Verify filter settings
    assert "filter" in result
    filter_config = result["filter"]
    assert filter_config["column"] == "mining_performance"
    assert filter_config["operator"] == ">"
    assert filter_config["value"] == 0.85
    
    # Verify export data
    assert "file_data" in result["export_data"]
    assert "file_size" in result["export_data"]
    assert "checksum" in result["export_data"]
    
    # Verify database entry
    db_record = db_session.query(ExportRecord).filter_by(
        user_id=test_user.user_id,
        export_id=result["export_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_create_export(db_session, test_user, test_data, test_export_config):
    """Test export creation"""
    # Create export
    result = create_export(
        data=test_data,
        export_config=test_export_config["export_formats"],
        user_id=test_user.user_id,
        message="Create data export",
        db_session=db_session
    )
    
    # Verify export result
    assert isinstance(result, Dict)
    assert "export_id" in result
    assert "timestamp" in result
    assert "export" in result
    
    # Verify export details
    export = result["export"]
    assert "format" in export
    assert "data" in export
    assert "metadata" in export
    
    # Verify export format
    assert export["format"] in test_export_config["export_formats"]["formats"]
    
    # Verify export metadata
    metadata = export["metadata"]
    assert "rows" in metadata
    assert "columns" in metadata
    assert "size" in metadata
    assert "created_at" in metadata
    
    # Verify export record
    export_record = db_session.query(ExportRecord).filter_by(
        export_id=result["export_id"]
    ).first()
    assert export_record is not None
    assert export_record.status == "CREATED"
    assert export_record.error is None

def test_schedule_export(db_session, test_user, test_data, test_export_config):
    """Test export scheduling"""
    # Schedule export
    result = schedule_export(
        data=test_data,
        schedule_config=test_export_config["export_scheduling"],
        user_id=test_user.user_id,
        message="Schedule data export",
        db_session=db_session
    )
    
    # Verify schedule result
    assert isinstance(result, Dict)
    assert "schedule_id" in result
    assert "timestamp" in result
    assert "schedule" in result
    
    # Verify schedule details
    schedule = result["schedule"]
    assert "type" in schedule
    assert "start_time" in schedule
    assert "end_time" in schedule
    assert "interval" in schedule
    assert "status" in schedule
    
    # Verify schedule type
    assert schedule["type"] in test_export_config["export_scheduling"]["schedule_types"]
    
    # Verify schedule record
    schedule_record = db_session.query(ExportSchedule).filter_by(
        schedule_id=result["schedule_id"]
    ).first()
    assert schedule_record is not None
    assert schedule_record.status == "SCHEDULED"
    assert schedule_record.error is None

def test_transform_data(db_session, test_user, test_data, test_export_config):
    """Test data transformation"""
    # Transform data
    result = transform_data(
        data=test_data,
        transform_config=test_export_config["data_transformation"],
        user_id=test_user.user_id,
        message="Transform export data",
        db_session=db_session
    )
    
    # Verify transformation result
    assert isinstance(result, Dict)
    assert "transform_id" in result
    assert "timestamp" in result
    assert "data" in result
    
    # Verify transformed data
    transformed_data = result["data"]
    assert isinstance(transformed_data, pd.DataFrame)
    assert len(transformed_data) > 0
    
    # Verify transformation metadata
    assert "metadata" in result
    metadata = result["metadata"]
    assert "operations" in metadata
    assert "columns" in metadata
    assert "rows" in metadata
    
    # Verify transformation record
    export_record = db_session.query(ExportRecord).filter_by(
        transform_id=result["transform_id"]
    ).first()
    assert export_record is not None
    assert export_record.status == "TRANSFORMED"
    assert export_record.error is None

def test_validate_export(db_session, test_user, test_data, test_export_config):
    """Test export validation"""
    # Validate export
    result = validate_export(
        data=test_data,
        validation_config=test_export_config["export_validation"],
        user_id=test_user.user_id,
        message="Validate export data",
        db_session=db_session
    )
    
    # Verify validation result
    assert isinstance(result, Dict)
    assert "validation_id" in result
    assert "timestamp" in result
    assert "validation" in result
    
    # Verify validation details
    validation = result["validation"]
    assert "is_valid" in validation
    assert "checks" in validation
    assert "issues" in validation
    
    # Verify validation checks
    checks = validation["checks"]
    assert "schema_validation" in checks
    assert "data_quality" in checks
    assert "size_limits" in checks
    assert "format_validation" in checks
    
    # Verify validation record
    export_record = db_session.query(ExportRecord).filter_by(
        validation_id=result["validation_id"]
    ).first()
    assert export_record is not None
    assert export_record.status == "VALIDATED"
    assert export_record.error is None

def test_track_export_history(db_session, test_user, test_data, test_export_config):
    """Test export history tracking"""
    # Create export first
    export = create_export(
        data=test_data,
        export_config=test_export_config["export_formats"],
        user_id=test_user.user_id,
        message="Test export",
        db_session=db_session
    )
    
    # Track export history
    result = track_export_history(
        export_id=export["export_id"],
        history_config=test_export_config["history_tracking"],
        user_id=test_user.user_id,
        message="Track export history",
        db_session=db_session
    )
    
    # Verify history result
    assert isinstance(result, Dict)
    assert "history_id" in result
    assert "timestamp" in result
    assert "history" in result
    
    # Verify history details
    history = result["history"]
    assert "export_id" in history
    assert "metadata" in history
    assert "performance" in history
    assert "errors" in history
    
    # Verify history record
    history_record = db_session.query(ExportHistory).filter_by(
        history_id=result["history_id"]
    ).first()
    assert history_record is not None
    assert history_record.status == "TRACKED"
    assert history_record.error is None

def test_get_export_info(db_session, test_user, test_data, test_export_config):
    """Test export information retrieval"""
    # Create export first
    export = create_export(
        data=test_data,
        export_config=test_export_config["export_formats"],
        user_id=test_user.user_id,
        message="Test export",
        db_session=db_session
    )
    
    # Get export info
    result = get_export_info(
        export_id=export["export_id"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "export_id" in result
    assert "timestamp" in result
    assert "info" in result
    
    # Verify export info
    info = result["info"]
    assert "format" in info
    assert "status" in info
    assert "metadata" in info
    assert "history" in info
    
    # Verify export record
    export_record = db_session.query(ExportRecord).filter_by(
        export_id=result["export_id"]
    ).first()
    assert export_record is not None
    assert export_record.status == "RETRIEVED"
    assert export_record.error is None

def test_list_exports(db_session, test_user, test_data, test_export_config):
    """Test export listing"""
    # Create multiple exports
    for i in range(5):
        create_export(
            data=test_data,
            export_config=test_export_config["export_formats"],
            user_id=test_user.user_id,
            message=f"Export {i+1}",
            db_session=db_session
        )
    
    # List exports
    result = list_exports(
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify listing result
    assert isinstance(result, Dict)
    assert "timestamp" in result
    assert "exports" in result
    
    # Verify exports list
    exports = result["exports"]
    assert isinstance(exports, List)
    assert len(exports) == 5
    
    # Verify export details
    for export in exports:
        assert "export_id" in export
        assert "timestamp" in export
        assert "format" in export
        assert "status" in export

def test_delete_export(db_session, test_user, test_data, test_export_config):
    """Test export deletion"""
    # Create export first
    export = create_export(
        data=test_data,
        export_config=test_export_config["export_formats"],
        user_id=test_user.user_id,
        message="Test export",
        db_session=db_session
    )
    
    # Delete export
    result = delete_export(
        export_id=export["export_id"],
        user_id=test_user.user_id,
        message="Delete test export",
        db_session=db_session
    )
    
    # Verify deletion result
    assert isinstance(result, Dict)
    assert "deletion_id" in result
    assert "timestamp" in result
    assert "status" in result
    
    # Verify status
    assert result["status"] == "DELETED"
    
    # Verify export record
    export_record = db_session.query(ExportRecord).filter_by(
        export_id=export["export_id"]
    ).first()
    assert export_record is not None
    assert export_record.status == "DELETED"
    assert export_record.error is None

def test_export_error_handling(db_session, test_user):
    """Test export error handling"""
    # Invalid export configuration
    with pytest.raises(ExportError) as excinfo:
        create_export(
            data=pd.DataFrame(),
            export_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid export configuration" in str(excinfo.value)
    
    # Invalid schedule configuration
    with pytest.raises(ExportError) as excinfo:
        schedule_export(
            data=pd.DataFrame(),
            schedule_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid schedule configuration" in str(excinfo.value)
    
    # Invalid transformation configuration
    with pytest.raises(ExportError) as excinfo:
        transform_data(
            data=pd.DataFrame(),
            transform_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid transformation configuration" in str(excinfo.value)
    
    # Invalid validation configuration
    with pytest.raises(ExportError) as excinfo:
        validate_export(
            data=pd.DataFrame(),
            validation_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid validation configuration" in str(excinfo.value)
    
    # Invalid export ID
    with pytest.raises(ExportError) as excinfo:
        track_export_history(
            export_id="invalid_id",
            history_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid export ID" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBExportError) as excinfo:
        create_export(
            data=pd.DataFrame(),
            export_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 