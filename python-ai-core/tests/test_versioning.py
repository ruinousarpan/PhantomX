import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import uuid
from unittest.mock import patch, MagicMock

from core.versioning import (
    create_version,
    track_changes,
    manage_versions,
    rollback_version,
    compare_versions,
    get_version_info,
    list_versions,
    delete_version,
    VersioningError
)
from database.models import User, VersionRecord, ChangeRecord
from database.exceptions import VersioningError as DBVersioningError

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
def test_version_config():
    """Create test version configuration"""
    return {
        "version_control": {
            "enabled": True,
            "auto_version": True,
            "version_format": "semantic",
            "major_threshold": 0.8,
            "minor_threshold": 0.5,
            "patch_threshold": 0.2
        },
        "change_tracking": {
            "enabled": True,
            "track_additions": True,
            "track_deletions": True,
            "track_modifications": True,
            "track_metadata": True
        },
        "version_management": {
            "enabled": True,
            "max_versions": 10,
            "retention_period": "1_year",
            "cleanup_strategy": "oldest_first"
        },
        "rollback": {
            "enabled": True,
            "verify_rollback": True,
            "backup_before_rollback": True,
            "notify_on_rollback": True
        },
        "comparison": {
            "enabled": True,
            "compare_data": True,
            "compare_metadata": True,
            "compare_schema": True,
            "highlight_changes": True
        }
    }

def test_create_version(db_session, test_user, test_data, test_version_config):
    """Test version creation"""
    # Create version
    result = create_version(
        data=test_data,
        version_config=test_version_config["version_control"],
        user_id=test_user.user_id,
        message="Initial version",
        db_session=db_session
    )
    
    # Verify version result
    assert isinstance(result, Dict)
    assert "version_id" in result
    assert "timestamp" in result
    assert "version" in result
    assert "metadata" in result
    
    # Verify version details
    version = result["version"]
    assert "major" in version
    assert "minor" in version
    assert "patch" in version
    assert "message" in version
    
    # Verify metadata
    metadata = result["metadata"]
    assert "user_id" in metadata
    assert "data_shape" in metadata
    assert "checksum" in metadata
    
    # Verify version record
    version_record = db_session.query(VersionRecord).filter_by(
        version_id=result["version_id"]
    ).first()
    assert version_record is not None
    assert version_record.status == "CREATED"
    assert version_record.error is None

def test_track_changes(db_session, test_user, test_data, test_version_config):
    """Test change tracking"""
    # Create initial version
    initial_version = create_version(
        data=test_data,
        version_config=test_version_config["version_control"],
        user_id=test_user.user_id,
        message="Initial version",
        db_session=db_session
    )
    
    # Modify data
    modified_data = test_data.copy()
    modified_data.loc[0:10, "trading_performance"] = np.random.uniform(0.8, 1.0, 11)
    modified_data.loc[11:20, "risk_score"] = np.random.uniform(0.2, 0.6, 10)
    
    # Track changes
    result = track_changes(
        data=modified_data,
        previous_version=initial_version["version_id"],
        change_config=test_version_config["change_tracking"],
        user_id=test_user.user_id,
        message="Modified performance and risk scores",
        db_session=db_session
    )
    
    # Verify change tracking result
    assert isinstance(result, Dict)
    assert "change_id" in result
    assert "timestamp" in result
    assert "changes" in result
    
    # Verify changes
    changes = result["changes"]
    assert "additions" in changes
    assert "deletions" in changes
    assert "modifications" in changes
    assert "metadata" in changes
    
    # Verify modifications
    modifications = changes["modifications"]
    assert isinstance(modifications, List)
    for mod in modifications:
        assert "field" in mod
        assert "indices" in mod
        assert "old_values" in mod
        assert "new_values" in mod
    
    # Verify change record
    change_record = db_session.query(ChangeRecord).filter_by(
        change_id=result["change_id"]
    ).first()
    assert change_record is not None
    assert change_record.status == "TRACKED"
    assert change_record.error is None

def test_manage_versions(db_session, test_user, test_data, test_version_config):
    """Test version management"""
    # Create multiple versions
    versions = []
    for i in range(5):
        modified_data = test_data.copy()
        modified_data["trading_performance"] += np.random.uniform(-0.1, 0.1, len(test_data))
        version = create_version(
            data=modified_data,
            version_config=test_version_config["version_control"],
            user_id=test_user.user_id,
            message=f"Version {i+1}",
            db_session=db_session
        )
        versions.append(version)
    
    # Manage versions
    result = manage_versions(
        data_id="test_data",
        management_config=test_version_config["version_management"],
        db_session=db_session
    )
    
    # Verify version management result
    assert isinstance(result, Dict)
    assert "management_id" in result
    assert "timestamp" in result
    assert "managed_versions" in result
    
    # Verify managed versions
    managed_versions = result["managed_versions"]
    assert "active" in managed_versions
    assert "archived" in managed_versions
    assert "deleted" in managed_versions
    
    # Verify active versions
    active = managed_versions["active"]
    assert isinstance(active, List)
    assert len(active) <= test_version_config["version_management"]["max_versions"]
    
    # Verify version record
    version_record = db_session.query(VersionRecord).filter_by(
        management_id=result["management_id"]
    ).first()
    assert version_record is not None
    assert version_record.status == "MANAGED"
    assert version_record.error is None

def test_rollback_version(db_session, test_user, test_data, test_version_config):
    """Test version rollback"""
    # Create initial version
    initial_version = create_version(
        data=test_data,
        version_config=test_version_config["version_control"],
        user_id=test_user.user_id,
        message="Initial version",
        db_session=db_session
    )
    
    # Create modified version
    modified_data = test_data.copy()
    modified_data["trading_performance"] += 0.1
    modified_version = create_version(
        data=modified_data,
        version_config=test_version_config["version_control"],
        user_id=test_user.user_id,
        message="Modified version",
        db_session=db_session
    )
    
    # Rollback to initial version
    result = rollback_version(
        data_id="test_data",
        target_version=initial_version["version_id"],
        rollback_config=test_version_config["rollback"],
        user_id=test_user.user_id,
        message="Rollback to initial version",
        db_session=db_session
    )
    
    # Verify rollback result
    assert isinstance(result, Dict)
    assert "rollback_id" in result
    assert "timestamp" in result
    assert "rolled_back_data" in result
    
    # Verify rolled back data
    rolled_back_data = result["rolled_back_data"]
    assert isinstance(rolled_back_data, pd.DataFrame)
    assert len(rolled_back_data) == len(test_data)
    assert np.allclose(rolled_back_data["trading_performance"], test_data["trading_performance"])
    
    # Verify version record
    version_record = db_session.query(VersionRecord).filter_by(
        rollback_id=result["rollback_id"]
    ).first()
    assert version_record is not None
    assert version_record.status == "ROLLED_BACK"
    assert version_record.error is None

def test_compare_versions(db_session, test_user, test_data, test_version_config):
    """Test version comparison"""
    # Create initial version
    initial_version = create_version(
        data=test_data,
        version_config=test_version_config["version_control"],
        user_id=test_user.user_id,
        message="Initial version",
        db_session=db_session
    )
    
    # Create modified version
    modified_data = test_data.copy()
    modified_data.loc[0:10, "trading_performance"] = np.random.uniform(0.8, 1.0, 11)
    modified_version = create_version(
        data=modified_data,
        version_config=test_version_config["version_control"],
        user_id=test_user.user_id,
        message="Modified version",
        db_session=db_session
    )
    
    # Compare versions
    result = compare_versions(
        version1=initial_version["version_id"],
        version2=modified_version["version_id"],
        comparison_config=test_version_config["comparison"],
        db_session=db_session
    )
    
    # Verify comparison result
    assert isinstance(result, Dict)
    assert "comparison_id" in result
    assert "timestamp" in result
    assert "comparison" in result
    
    # Verify comparison details
    comparison = result["comparison"]
    assert "data_diff" in comparison
    assert "metadata_diff" in comparison
    assert "schema_diff" in comparison
    
    # Verify data differences
    data_diff = comparison["data_diff"]
    assert "additions" in data_diff
    assert "deletions" in data_diff
    assert "modifications" in data_diff
    
    # Verify modifications
    modifications = data_diff["modifications"]
    assert isinstance(modifications, List)
    for mod in modifications:
        assert "field" in mod
        assert "indices" in mod
        assert "old_values" in mod
        assert "new_values" in mod
    
    # Verify version record
    version_record = db_session.query(VersionRecord).filter_by(
        comparison_id=result["comparison_id"]
    ).first()
    assert version_record is not None
    assert version_record.status == "COMPARED"
    assert version_record.error is None

def test_get_version_info(db_session, test_user, test_data, test_version_config):
    """Test version information retrieval"""
    # Create version
    version = create_version(
        data=test_data,
        version_config=test_version_config["version_control"],
        user_id=test_user.user_id,
        message="Test version",
        db_session=db_session
    )
    
    # Get version info
    result = get_version_info(
        version_id=version["version_id"],
        db_session=db_session
    )
    
    # Verify version info result
    assert isinstance(result, Dict)
    assert "version_id" in result
    assert "timestamp" in result
    assert "info" in result
    
    # Verify info content
    info = result["info"]
    assert "version" in info
    assert "metadata" in info
    assert "changes" in info
    assert "user" in info
    
    # Verify version details
    version_info = info["version"]
    assert "major" in version_info
    assert "minor" in version_info
    assert "patch" in version_info
    assert "message" in version_info
    
    # Verify version record
    version_record = db_session.query(VersionRecord).filter_by(
        version_id=result["version_id"]
    ).first()
    assert version_record is not None
    assert version_record.status == "RETRIEVED"
    assert version_record.error is None

def test_list_versions(db_session, test_user, test_data, test_version_config):
    """Test version listing"""
    # Create multiple versions
    for i in range(5):
        modified_data = test_data.copy()
        modified_data["trading_performance"] += np.random.uniform(-0.1, 0.1, len(test_data))
        create_version(
            data=modified_data,
            version_config=test_version_config["version_control"],
            user_id=test_user.user_id,
            message=f"Version {i+1}",
            db_session=db_session
        )
    
    # List versions
    result = list_versions(
        data_id="test_data",
        db_session=db_session
    )
    
    # Verify version listing result
    assert isinstance(result, Dict)
    assert "timestamp" in result
    assert "versions" in result
    
    # Verify versions list
    versions = result["versions"]
    assert isinstance(versions, List)
    assert len(versions) == 5
    
    # Verify version details
    for version in versions:
        assert "version_id" in version
        assert "version" in version
        assert "timestamp" in version
        assert "message" in version
        assert "user_id" in version

def test_delete_version(db_session, test_user, test_data, test_version_config):
    """Test version deletion"""
    # Create version
    version = create_version(
        data=test_data,
        version_config=test_version_config["version_control"],
        user_id=test_user.user_id,
        message="Test version",
        db_session=db_session
    )
    
    # Delete version
    result = delete_version(
        version_id=version["version_id"],
        user_id=test_user.user_id,
        message="Delete test version",
        db_session=db_session
    )
    
    # Verify deletion result
    assert isinstance(result, Dict)
    assert "deletion_id" in result
    assert "timestamp" in result
    assert "status" in result
    
    # Verify status
    assert result["status"] == "DELETED"
    
    # Verify version record
    version_record = db_session.query(VersionRecord).filter_by(
        version_id=version["version_id"]
    ).first()
    assert version_record is not None
    assert version_record.status == "DELETED"
    assert version_record.error is None

def test_versioning_error_handling(db_session, test_user):
    """Test versioning error handling"""
    # Invalid version configuration
    with pytest.raises(VersioningError) as excinfo:
        create_version(
            data=pd.DataFrame(),
            version_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid version configuration" in str(excinfo.value)
    
    # Invalid change configuration
    with pytest.raises(VersioningError) as excinfo:
        track_changes(
            data=pd.DataFrame(),
            previous_version="invalid",
            change_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid change configuration" in str(excinfo.value)
    
    # Invalid management configuration
    with pytest.raises(VersioningError) as excinfo:
        manage_versions(
            data_id="invalid",
            management_config={},
            db_session=db_session
        )
    assert "Invalid management configuration" in str(excinfo.value)
    
    # Invalid rollback configuration
    with pytest.raises(VersioningError) as excinfo:
        rollback_version(
            data_id="invalid",
            target_version="invalid",
            rollback_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid rollback configuration" in str(excinfo.value)
    
    # Invalid comparison configuration
    with pytest.raises(VersioningError) as excinfo:
        compare_versions(
            version1="invalid",
            version2="invalid",
            comparison_config={},
            db_session=db_session
        )
    assert "Invalid comparison configuration" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBVersioningError) as excinfo:
        create_version(
            data=pd.DataFrame(),
            version_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 