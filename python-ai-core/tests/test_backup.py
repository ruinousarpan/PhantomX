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
import zipfile
import tarfile
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.backup import (
    backup_performance_data,
    backup_risk_data,
    backup_reward_data,
    backup_activity_data,
    backup_analytics_data,
    restore_backup,
    manage_backup,
    BackupError,
    create_backup,
    schedule_backup,
    verify_backup,
    manage_backups,
    get_backup_info,
    list_backups,
    delete_backup
)
from database.models import User, Backup, BackupEntry, BackupRecord, BackupSchedule
from database.exceptions import BackupError as DBBackupError

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
def test_backup_config():
    """Create test backup configuration"""
    return {
        "format": "zip",
        "compression": "gzip",
        "encryption": "aes-256-gcm",
        "encryption_key": "test_encryption_key",
        "storage": {
            "type": "local",
            "path": tempfile.mkdtemp(),
            "retention": 30,  # days
            "max_size": 1073741824,  # 1GB
            "max_files": 10
        },
        "options": {
            "verify": True,
            "checksum": "sha256",
            "metadata": True,
            "notification": {
                "on_success": True,
                "on_failure": True,
                "recipients": ["test@example.com"]
            }
        },
        "backup_creation": {
            "enabled": True,
            "compression": True,
            "encryption": True,
            "incremental": True,
            "verify_after_backup": True
        },
        "backup_restoration": {
            "enabled": True,
            "verify_before_restore": True,
            "backup_before_restore": True,
            "notify_on_restore": True
        },
        "backup_scheduling": {
            "enabled": True,
            "schedule_type": "cron",
            "schedule": "0 0 * * *",  # Daily at midnight
            "retry_on_failure": True,
            "max_retries": 3
        },
        "backup_verification": {
            "enabled": True,
            "verify_data": True,
            "verify_metadata": True,
            "verify_integrity": True,
            "checksum_verification": True
        },
        "backup_retention": {
            "enabled": True,
            "max_backups": 10,
            "retention_period": "30_days",
            "cleanup_strategy": "oldest_first"
        }
    }

@pytest.fixture
def cleanup_backup_files(test_backup_config):
    """Cleanup backup files after tests"""
    yield
    if os.path.exists(test_backup_config["storage"]["path"]):
        shutil.rmtree(test_backup_config["storage"]["path"])

def test_backup_performance_data(db_session, test_user, test_performance_data, test_backup_config, cleanup_backup_files):
    """Test performance data backup"""
    # Backup performance data
    result = backup_performance_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        backup_config=test_backup_config,
        db_session=db_session
    )
    
    # Verify backup result
    assert isinstance(result, Dict)
    assert "backup_id" in result
    assert "file_path" in result
    
    # Verify backup metadata
    assert result["backup_type"] == "PERFORMANCE_BACKUP"
    assert result["backup_config"]["format"] == "zip"
    
    # Verify backup details
    assert "backup_details" in result
    assert isinstance(result["backup_details"], Dict)
    assert "timestamp" in result["backup_details"]
    assert "size" in result["backup_details"]
    assert "checksum" in result["backup_details"]
    
    # Verify backup file exists
    assert os.path.exists(result["file_path"])
    
    # Verify backup file is a valid zip file
    with zipfile.ZipFile(result["file_path"], 'r') as zip_ref:
        assert "performance_data.csv" in zip_ref.namelist()
    
    # Verify database entry
    db_backup = db_session.query(Backup).filter_by(
        user_id=test_user.user_id,
        backup_type="PERFORMANCE_BACKUP"
    ).first()
    assert db_backup is not None
    assert db_backup.is_active is True
    assert db_backup.error is None
    
    # Verify backup entry
    db_entry = db_session.query(BackupEntry).filter_by(
        backup_id=db_backup.backup_id
    ).first()
    assert db_entry is not None
    assert db_entry.status == "COMPLETED"
    assert "Performance data backed up" in db_entry.message

def test_backup_risk_data(db_session, test_user, test_risk_data, test_backup_config, cleanup_backup_files):
    """Test risk data backup"""
    # Backup risk data
    result = backup_risk_data(
        user_id=test_user.user_id,
        data=test_risk_data,
        backup_config=test_backup_config,
        db_session=db_session
    )
    
    # Verify backup result
    assert isinstance(result, Dict)
    assert "backup_id" in result
    assert "file_path" in result
    
    # Verify backup metadata
    assert result["backup_type"] == "RISK_BACKUP"
    assert result["backup_config"]["format"] == "zip"
    
    # Verify backup details
    assert "backup_details" in result
    assert isinstance(result["backup_details"], Dict)
    assert "timestamp" in result["backup_details"]
    assert "size" in result["backup_details"]
    assert "checksum" in result["backup_details"]
    
    # Verify backup file exists
    assert os.path.exists(result["file_path"])
    
    # Verify backup file is a valid zip file
    with zipfile.ZipFile(result["file_path"], 'r') as zip_ref:
        assert "risk_data.csv" in zip_ref.namelist()
    
    # Verify database entry
    db_backup = db_session.query(Backup).filter_by(
        user_id=test_user.user_id,
        backup_type="RISK_BACKUP"
    ).first()
    assert db_backup is not None
    assert db_backup.is_active is True
    assert db_backup.error is None
    
    # Verify backup entry
    db_entry = db_session.query(BackupEntry).filter_by(
        backup_id=db_backup.backup_id
    ).first()
    assert db_entry is not None
    assert db_entry.status == "COMPLETED"
    assert "Risk data backed up" in db_entry.message

def test_backup_reward_data(db_session, test_user, test_reward_data, test_backup_config, cleanup_backup_files):
    """Test reward data backup"""
    # Backup reward data
    result = backup_reward_data(
        user_id=test_user.user_id,
        data=test_reward_data,
        backup_config=test_backup_config,
        db_session=db_session
    )
    
    # Verify backup result
    assert isinstance(result, Dict)
    assert "backup_id" in result
    assert "file_path" in result
    
    # Verify backup metadata
    assert result["backup_type"] == "REWARD_BACKUP"
    assert result["backup_config"]["format"] == "zip"
    
    # Verify backup details
    assert "backup_details" in result
    assert isinstance(result["backup_details"], Dict)
    assert "timestamp" in result["backup_details"]
    assert "size" in result["backup_details"]
    assert "checksum" in result["backup_details"]
    
    # Verify backup file exists
    assert os.path.exists(result["file_path"])
    
    # Verify backup file is a valid zip file
    with zipfile.ZipFile(result["file_path"], 'r') as zip_ref:
        assert "reward_data.csv" in zip_ref.namelist()
    
    # Verify database entry
    db_backup = db_session.query(Backup).filter_by(
        user_id=test_user.user_id,
        backup_type="REWARD_BACKUP"
    ).first()
    assert db_backup is not None
    assert db_backup.is_active is True
    assert db_backup.error is None
    
    # Verify backup entry
    db_entry = db_session.query(BackupEntry).filter_by(
        backup_id=db_backup.backup_id
    ).first()
    assert db_entry is not None
    assert db_entry.status == "COMPLETED"
    assert "Reward data backed up" in db_entry.message

def test_backup_activity_data(db_session, test_user, test_activity_data, test_backup_config, cleanup_backup_files):
    """Test activity data backup"""
    # Backup activity data
    result = backup_activity_data(
        user_id=test_user.user_id,
        data=test_activity_data,
        backup_config=test_backup_config,
        db_session=db_session
    )
    
    # Verify backup result
    assert isinstance(result, Dict)
    assert "backup_id" in result
    assert "file_path" in result
    
    # Verify backup metadata
    assert result["backup_type"] == "ACTIVITY_BACKUP"
    assert result["backup_config"]["format"] == "zip"
    
    # Verify backup details
    assert "backup_details" in result
    assert isinstance(result["backup_details"], Dict)
    assert "timestamp" in result["backup_details"]
    assert "size" in result["backup_details"]
    assert "checksum" in result["backup_details"]
    
    # Verify backup file exists
    assert os.path.exists(result["file_path"])
    
    # Verify backup file is a valid zip file
    with zipfile.ZipFile(result["file_path"], 'r') as zip_ref:
        assert "activity_data.csv" in zip_ref.namelist()
    
    # Verify database entry
    db_backup = db_session.query(Backup).filter_by(
        user_id=test_user.user_id,
        backup_type="ACTIVITY_BACKUP"
    ).first()
    assert db_backup is not None
    assert db_backup.is_active is True
    assert db_backup.error is None
    
    # Verify backup entry
    db_entry = db_session.query(BackupEntry).filter_by(
        backup_id=db_backup.backup_id
    ).first()
    assert db_entry is not None
    assert db_entry.status == "COMPLETED"
    assert "Activity data backed up" in db_entry.message

def test_backup_analytics_data(db_session, test_user, test_analytics_data, test_backup_config, cleanup_backup_files):
    """Test analytics data backup"""
    # Backup analytics data
    result = backup_analytics_data(
        user_id=test_user.user_id,
        data=test_analytics_data,
        backup_config=test_backup_config,
        db_session=db_session
    )
    
    # Verify backup result
    assert isinstance(result, Dict)
    assert "backup_id" in result
    assert "file_path" in result
    
    # Verify backup metadata
    assert result["backup_type"] == "ANALYTICS_BACKUP"
    assert result["backup_config"]["format"] == "zip"
    
    # Verify backup details
    assert "backup_details" in result
    assert isinstance(result["backup_details"], Dict)
    assert "timestamp" in result["backup_details"]
    assert "size" in result["backup_details"]
    assert "checksum" in result["backup_details"]
    
    # Verify backup file exists
    assert os.path.exists(result["file_path"])
    
    # Verify backup file is a valid zip file
    with zipfile.ZipFile(result["file_path"], 'r') as zip_ref:
        assert "analytics_data.json" in zip_ref.namelist()
    
    # Verify database entry
    db_backup = db_session.query(Backup).filter_by(
        user_id=test_user.user_id,
        backup_type="ANALYTICS_BACKUP"
    ).first()
    assert db_backup is not None
    assert db_backup.is_active is True
    assert db_backup.error is None
    
    # Verify backup entry
    db_entry = db_session.query(BackupEntry).filter_by(
        backup_id=db_backup.backup_id
    ).first()
    assert db_entry is not None
    assert db_entry.status == "COMPLETED"
    assert "Analytics data backed up" in db_entry.message

def test_restore_backup(db_session, test_user, test_performance_data, test_backup_config, cleanup_backup_files):
    """Test backup restoration"""
    # First, create a backup to get a backup ID
    backup_result = backup_performance_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        backup_config=test_backup_config,
        db_session=db_session
    )
    
    backup_id = backup_result["backup_id"]
    
    # Restore backup
    result = restore_backup(
        user_id=test_user.user_id,
        backup_id=backup_id,
        db_session=db_session
    )
    
    # Verify restoration result
    assert isinstance(result, Dict)
    assert "backup_id" in result
    assert "restored_data" in result
    
    # Verify restored data
    assert isinstance(result["restored_data"], pd.DataFrame)
    assert "timestamp" in result["restored_data"].columns
    assert "mining_performance" in result["restored_data"].columns
    assert "staking_performance" in result["restored_data"].columns
    assert "trading_performance" in result["restored_data"].columns
    assert "overall_performance" in result["restored_data"].columns
    
    # Verify database entry
    db_entry = db_session.query(BackupEntry).filter_by(
        backup_id=backup_id,
        action="RESTORE"
    ).first()
    assert db_entry is not None
    assert db_entry.status == "COMPLETED"
    assert "Backup restored" in db_entry.message

def test_manage_backup(db_session, test_user, test_performance_data, test_backup_config, cleanup_backup_files):
    """Test backup management"""
    # First, create a backup to get a backup ID
    backup_result = backup_performance_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        backup_config=test_backup_config,
        db_session=db_session
    )
    
    backup_id = backup_result["backup_id"]
    
    # Test backup management operations
    # 1. Archive backup
    archive_result = manage_backup(
        user_id=test_user.user_id,
        backup_id=backup_id,
        action="archive",
        db_session=db_session
    )
    
    assert isinstance(archive_result, Dict)
    assert "backup_id" in archive_result
    assert "status" in archive_result
    assert archive_result["status"] == "ARCHIVED"
    
    # 2. Restore backup
    restore_result = manage_backup(
        user_id=test_user.user_id,
        backup_id=backup_id,
        action="restore",
        db_session=db_session
    )
    
    assert isinstance(restore_result, Dict)
    assert "backup_id" in restore_result
    assert "status" in restore_result
    assert restore_result["status"] == "RESTORED"
    
    # 3. Modify backup
    modified_config = test_backup_config.copy()
    modified_config["format"] = "tar"
    
    modify_result = manage_backup(
        user_id=test_user.user_id,
        backup_id=backup_id,
        action="modify",
        backup_config=modified_config,
        db_session=db_session
    )
    
    assert isinstance(modify_result, Dict)
    assert "backup_id" in modify_result
    assert "status" in modify_result
    assert "backup_details" in modify_result
    assert modify_result["backup_details"]["format"] == "tar"
    
    # 4. Delete backup
    delete_result = manage_backup(
        user_id=test_user.user_id,
        backup_id=backup_id,
        action="delete",
        db_session=db_session
    )
    
    assert isinstance(delete_result, Dict)
    assert "backup_id" in delete_result
    assert "status" in delete_result
    assert delete_result["status"] == "DELETED"
    
    # Verify database entry
    db_backup = db_session.query(Backup).filter_by(
        user_id=test_user.user_id,
        backup_id=backup_id
    ).first()
    assert db_backup is not None
    assert db_backup.is_active is False
    assert db_backup.error is None

def test_backup_error_handling(db_session, test_user):
    """Test backup error handling"""
    # Invalid user ID
    with pytest.raises(BackupError) as excinfo:
        backup_performance_data(
            user_id=None,
            data=pd.DataFrame(),
            backup_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(BackupError) as excinfo:
        backup_risk_data(
            user_id=test_user.user_id,
            data=None,
            backup_config={},
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Empty data
    with pytest.raises(BackupError) as excinfo:
        backup_reward_data(
            user_id=test_user.user_id,
            data=pd.DataFrame(),
            backup_config={},
            db_session=db_session
        )
    assert "Empty data" in str(excinfo.value)
    
    # Invalid backup format
    with pytest.raises(BackupError) as excinfo:
        backup_activity_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            backup_config={"format": "invalid_format"},
            db_session=db_session
        )
    assert "Invalid backup format" in str(excinfo.value)
    
    # Invalid storage type
    with pytest.raises(BackupError) as excinfo:
        backup_analytics_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            backup_config={"storage": {"type": "invalid_storage"}},
            db_session=db_session
        )
    assert "Invalid storage type" in str(excinfo.value)
    
    # Invalid backup action
    with pytest.raises(BackupError) as excinfo:
        manage_backup(
            user_id=test_user.user_id,
            backup_id="test_backup_id",
            action="invalid_action",
            db_session=db_session
        )
    assert "Invalid backup action" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBBackupError) as excinfo:
        backup_performance_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            backup_config={"format": "zip"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value)

def test_create_backup(db_session, test_user, test_data, test_backup_config):
    """Test backup creation"""
    # Create backup
    result = create_backup(
        data=test_data,
        backup_config=test_backup_config["backup_creation"],
        user_id=test_user.user_id,
        message="Initial backup",
        db_session=db_session
    )
    
    # Verify backup result
    assert isinstance(result, Dict)
    assert "backup_id" in result
    assert "timestamp" in result
    assert "backup_path" in result
    assert "metadata" in result
    
    # Verify backup details
    backup = result["backup"]
    assert "size" in backup
    assert "compressed" in backup
    assert "encrypted" in backup
    assert "incremental" in backup
    
    # Verify metadata
    metadata = result["metadata"]
    assert "user_id" in metadata
    assert "data_shape" in metadata
    assert "checksum" in metadata
    
    # Verify backup record
    backup_record = db_session.query(BackupRecord).filter_by(
        backup_id=result["backup_id"]
    ).first()
    assert backup_record is not None
    assert backup_record.status == "CREATED"
    assert backup_record.error is None

def test_schedule_backup(db_session, test_user, test_data, test_backup_config):
    """Test backup scheduling"""
    # Schedule backup
    result = schedule_backup(
        data_id="test_data",
        schedule_config=test_backup_config["backup_scheduling"],
        user_id=test_user.user_id,
        message="Schedule daily backup",
        db_session=db_session
    )
    
    # Verify scheduling result
    assert isinstance(result, Dict)
    assert "schedule_id" in result
    assert "timestamp" in result
    assert "schedule" in result
    
    # Verify schedule details
    schedule = result["schedule"]
    assert "type" in schedule
    assert "cron" in schedule
    assert "next_run" in schedule
    assert "status" in schedule
    
    # Verify schedule record
    schedule_record = db_session.query(BackupSchedule).filter_by(
        schedule_id=result["schedule_id"]
    ).first()
    assert schedule_record is not None
    assert schedule_record.status == "SCHEDULED"
    assert schedule_record.error is None

def test_verify_backup(db_session, test_user, test_data, test_backup_config):
    """Test backup verification"""
    # Create backup
    backup = create_backup(
        data=test_data,
        backup_config=test_backup_config["backup_creation"],
        user_id=test_user.user_id,
        message="Test backup",
        db_session=db_session
    )
    
    # Verify backup
    result = verify_backup(
        backup_id=backup["backup_id"],
        verification_config=test_backup_config["backup_verification"],
        db_session=db_session
    )
    
    # Verify verification result
    assert isinstance(result, Dict)
    assert "verification_id" in result
    assert "timestamp" in result
    assert "verification" in result
    
    # Verify verification details
    verification = result["verification"]
    assert "data_verified" in verification
    assert "metadata_verified" in verification
    assert "integrity_verified" in verification
    assert "checksum_verified" in verification
    
    # Verify backup record
    backup_record = db_session.query(BackupRecord).filter_by(
        backup_id=backup["backup_id"]
    ).first()
    assert backup_record is not None
    assert backup_record.status == "VERIFIED"
    assert backup_record.error is None

def test_manage_backups(db_session, test_user, test_data, test_backup_config):
    """Test backup management"""
    # Create multiple backups
    backups = []
    for i in range(5):
        backup = create_backup(
            data=test_data,
            backup_config=test_backup_config["backup_creation"],
            user_id=test_user.user_id,
            message=f"Backup {i+1}",
            db_session=db_session
        )
        backups.append(backup)
    
    # Manage backups
    result = manage_backups(
        data_id="test_data",
        management_config=test_backup_config["backup_retention"],
        db_session=db_session
    )
    
    # Verify backup management result
    assert isinstance(result, Dict)
    assert "management_id" in result
    assert "timestamp" in result
    assert "managed_backups" in result
    
    # Verify managed backups
    managed_backups = result["managed_backups"]
    assert "active" in managed_backups
    assert "archived" in managed_backups
    assert "deleted" in managed_backups
    
    # Verify active backups
    active = managed_backups["active"]
    assert isinstance(active, List)
    assert len(active) <= test_backup_config["backup_retention"]["max_backups"]
    
    # Verify backup record
    backup_record = db_session.query(BackupRecord).filter_by(
        management_id=result["management_id"]
    ).first()
    assert backup_record is not None
    assert backup_record.status == "MANAGED"
    assert backup_record.error is None

def test_get_backup_info(db_session, test_user, test_data, test_backup_config):
    """Test backup information retrieval"""
    # Create backup
    backup = create_backup(
        data=test_data,
        backup_config=test_backup_config["backup_creation"],
        user_id=test_user.user_id,
        message="Test backup",
        db_session=db_session
    )
    
    # Get backup info
    result = get_backup_info(
        backup_id=backup["backup_id"],
        db_session=db_session
    )
    
    # Verify backup info result
    assert isinstance(result, Dict)
    assert "backup_id" in result
    assert "timestamp" in result
    assert "info" in result
    
    # Verify info content
    info = result["info"]
    assert "backup" in info
    assert "metadata" in info
    assert "verification" in info
    assert "user" in info
    
    # Verify backup details
    backup_info = info["backup"]
    assert "size" in backup_info
    assert "compressed" in backup_info
    assert "encrypted" in backup_info
    assert "incremental" in backup_info
    
    # Verify backup record
    backup_record = db_session.query(BackupRecord).filter_by(
        backup_id=result["backup_id"]
    ).first()
    assert backup_record is not None
    assert backup_record.status == "RETRIEVED"
    assert backup_record.error is None

def test_list_backups(db_session, test_user, test_data, test_backup_config):
    """Test backup listing"""
    # Create multiple backups
    for i in range(5):
        create_backup(
            data=test_data,
            backup_config=test_backup_config["backup_creation"],
            user_id=test_user.user_id,
            message=f"Backup {i+1}",
            db_session=db_session
        )
    
    # List backups
    result = list_backups(
        data_id="test_data",
        db_session=db_session
    )
    
    # Verify backup listing result
    assert isinstance(result, Dict)
    assert "timestamp" in result
    assert "backups" in result
    
    # Verify backups list
    backups = result["backups"]
    assert isinstance(backups, List)
    assert len(backups) == 5
    
    # Verify backup details
    for backup in backups:
        assert "backup_id" in backup
        assert "timestamp" in backup
        assert "size" in backup
        assert "status" in backup
        assert "user_id" in backup

def test_delete_backup(db_session, test_user, test_data, test_backup_config):
    """Test backup deletion"""
    # Create backup
    backup = create_backup(
        data=test_data,
        backup_config=test_backup_config["backup_creation"],
        user_id=test_user.user_id,
        message="Test backup",
        db_session=db_session
    )
    
    # Delete backup
    result = delete_backup(
        backup_id=backup["backup_id"],
        user_id=test_user.user_id,
        message="Delete test backup",
        db_session=db_session
    )
    
    # Verify deletion result
    assert isinstance(result, Dict)
    assert "deletion_id" in result
    assert "timestamp" in result
    assert "status" in result
    
    # Verify status
    assert result["status"] == "DELETED"
    
    # Verify backup record
    backup_record = db_session.query(BackupRecord).filter_by(
        backup_id=backup["backup_id"]
    ).first()
    assert backup_record is not None
    assert backup_record.status == "DELETED"
    assert backup_record.error is None

def test_backup_error_handling(db_session, test_user):
    """Test backup error handling"""
    # Invalid backup configuration
    with pytest.raises(BackupError) as excinfo:
        create_backup(
            data=pd.DataFrame(),
            backup_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid backup configuration" in str(excinfo.value)
    
    # Invalid restore configuration
    with pytest.raises(BackupError) as excinfo:
        restore_backup(
            backup_id="invalid",
            restore_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid restore configuration" in str(excinfo.value)
    
    # Invalid schedule configuration
    with pytest.raises(BackupError) as excinfo:
        schedule_backup(
            data_id="invalid",
            schedule_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid schedule configuration" in str(excinfo.value)
    
    # Invalid verification configuration
    with pytest.raises(BackupError) as excinfo:
        verify_backup(
            backup_id="invalid",
            verification_config={},
            db_session=db_session
        )
    assert "Invalid verification configuration" in str(excinfo.value)
    
    # Invalid management configuration
    with pytest.raises(BackupError) as excinfo:
        manage_backups(
            data_id="invalid",
            management_config={},
            db_session=db_session
        )
    assert "Invalid management configuration" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBBackupError) as excinfo:
        create_backup(
            data=pd.DataFrame(),
            backup_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 