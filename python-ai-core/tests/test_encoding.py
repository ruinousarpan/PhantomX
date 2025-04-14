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

from core.encoding import (
    encode_data,
    get_encoding_info,
    EncodingError
)
from database.models import User, EncodingRecord
from database.exceptions import EncodingError as DBEncodingError

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
def test_categorical_data():
    """Create test categorical data"""
    # Create data with categorical variables
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
    days = dates.date
    
    return pd.DataFrame({
        "timestamp": dates,
        "day": days,
        "mining_status": np.random.choice(["active", "inactive", "maintenance"], 100),
        "staking_status": np.random.choice(["active", "inactive", "pending"], 100),
        "trading_status": np.random.choice(["active", "inactive", "suspended"], 100),
        "overall_status": np.random.choice(["good", "fair", "poor"], 100)
    })

@pytest.fixture
def test_mixed_data():
    """Create test mixed data with categorical and numerical variables"""
    # Create data with mixed variables
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
    days = dates.date
    
    return pd.DataFrame({
        "timestamp": dates,
        "day": days,
        "mining_status": np.random.choice(["active", "inactive", "maintenance"], 100),
        "mining_performance": np.random.uniform(0.8, 0.9, 100),
        "staking_status": np.random.choice(["active", "inactive", "pending"], 100),
        "staking_performance": np.random.uniform(0.85, 0.95, 100),
        "trading_status": np.random.choice(["active", "inactive", "suspended"], 100),
        "trading_performance": np.random.uniform(0.7, 0.8, 100),
        "overall_status": np.random.choice(["good", "fair", "poor"], 100),
        "overall_performance": np.random.uniform(0.8, 0.9, 100)
    })

@pytest.fixture
def test_encoding_config():
    """Create test encoding configuration"""
    return {
        "encoding_type": "onehot",
        "columns": None,  # Encode all categorical columns
        "exclude_columns": ["timestamp", "day"],
        "handle_unknown": "ignore"
    }

def test_encode_categorical_data(db_session, test_user, test_categorical_data, test_encoding_config):
    """Test encoding categorical data"""
    # Encode categorical data
    result = encode_data(
        user_id=test_user.user_id,
        data=test_categorical_data,
        encoding_config=test_encoding_config,
        db_session=db_session
    )
    
    # Verify encoding result
    assert isinstance(result, Dict)
    assert "encoding_id" in result
    assert "encoded_data" in result
    
    # Verify encoding metadata
    assert result["encoding_type"] == "onehot"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["handle_unknown"] == "ignore"
    
    # Verify encoding details
    assert "encoding_details" in result
    assert isinstance(result["encoding_details"], Dict)
    assert "timestamp" in result["encoding_details"]
    assert "original_shape" in result["encoding_details"]
    assert "encoded_shape" in result["encoding_details"]
    assert "encoders" in result["encoding_details"]
    
    # Verify encoded data
    assert isinstance(result["encoded_data"], pd.DataFrame)
    assert "day" in result["encoded_data"].columns
    assert "mining_status_active" in result["encoded_data"].columns
    assert "mining_status_inactive" in result["encoded_data"].columns
    assert "mining_status_maintenance" in result["encoded_data"].columns
    assert "staking_status_active" in result["encoded_data"].columns
    assert "staking_status_inactive" in result["encoded_data"].columns
    assert "staking_status_pending" in result["encoded_data"].columns
    assert "trading_status_active" in result["encoded_data"].columns
    assert "trading_status_inactive" in result["encoded_data"].columns
    assert "trading_status_suspended" in result["encoded_data"].columns
    assert "overall_status_good" in result["encoded_data"].columns
    assert "overall_status_fair" in result["encoded_data"].columns
    assert "overall_status_poor" in result["encoded_data"].columns
    
    # Verify data encoding
    assert len(result["encoded_data"]) == len(test_categorical_data)
    # Check that one-hot encoding is correct (only one 1 per category)
    for i in range(len(result["encoded_data"])):
        assert result["encoded_data"].iloc[i][["mining_status_active", "mining_status_inactive", "mining_status_maintenance"]].sum() == 1
        assert result["encoded_data"].iloc[i][["staking_status_active", "staking_status_inactive", "staking_status_pending"]].sum() == 1
        assert result["encoded_data"].iloc[i][["trading_status_active", "trading_status_inactive", "trading_status_suspended"]].sum() == 1
        assert result["encoded_data"].iloc[i][["overall_status_good", "overall_status_fair", "overall_status_poor"]].sum() == 1
    
    # Verify database entry
    db_record = db_session.query(EncodingRecord).filter_by(
        user_id=test_user.user_id,
        encoding_id=result["encoding_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_encode_mixed_data(db_session, test_user, test_mixed_data, test_encoding_config):
    """Test encoding mixed data"""
    # Encode mixed data
    result = encode_data(
        user_id=test_user.user_id,
        data=test_mixed_data,
        encoding_config=test_encoding_config,
        db_session=db_session
    )
    
    # Verify encoding result
    assert isinstance(result, Dict)
    assert "encoding_id" in result
    assert "encoded_data" in result
    
    # Verify encoding metadata
    assert result["encoding_type"] == "onehot"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["handle_unknown"] == "ignore"
    
    # Verify encoding details
    assert "encoding_details" in result
    assert isinstance(result["encoding_details"], Dict)
    assert "timestamp" in result["encoding_details"]
    assert "original_shape" in result["encoding_details"]
    assert "encoded_shape" in result["encoding_details"]
    assert "encoders" in result["encoding_details"]
    
    # Verify encoded data
    assert isinstance(result["encoded_data"], pd.DataFrame)
    assert "day" in result["encoded_data"].columns
    assert "mining_status_active" in result["encoded_data"].columns
    assert "mining_status_inactive" in result["encoded_data"].columns
    assert "mining_status_maintenance" in result["encoded_data"].columns
    assert "mining_performance" in result["encoded_data"].columns
    assert "staking_status_active" in result["encoded_data"].columns
    assert "staking_status_inactive" in result["encoded_data"].columns
    assert "staking_status_pending" in result["encoded_data"].columns
    assert "staking_performance" in result["encoded_data"].columns
    assert "trading_status_active" in result["encoded_data"].columns
    assert "trading_status_inactive" in result["encoded_data"].columns
    assert "trading_status_suspended" in result["encoded_data"].columns
    assert "trading_performance" in result["encoded_data"].columns
    assert "overall_status_good" in result["encoded_data"].columns
    assert "overall_status_fair" in result["encoded_data"].columns
    assert "overall_status_poor" in result["encoded_data"].columns
    assert "overall_performance" in result["encoded_data"].columns
    
    # Verify data encoding
    assert len(result["encoded_data"]) == len(test_mixed_data)
    # Check that one-hot encoding is correct (only one 1 per category)
    for i in range(len(result["encoded_data"])):
        assert result["encoded_data"].iloc[i][["mining_status_active", "mining_status_inactive", "mining_status_maintenance"]].sum() == 1
        assert result["encoded_data"].iloc[i][["staking_status_active", "staking_status_inactive", "staking_status_pending"]].sum() == 1
        assert result["encoded_data"].iloc[i][["trading_status_active", "trading_status_inactive", "trading_status_suspended"]].sum() == 1
        assert result["encoded_data"].iloc[i][["overall_status_good", "overall_status_fair", "overall_status_poor"]].sum() == 1
    
    # Check that numerical columns are unchanged
    assert result["encoded_data"]["mining_performance"].equals(test_mixed_data["mining_performance"])
    assert result["encoded_data"]["staking_performance"].equals(test_mixed_data["staking_performance"])
    assert result["encoded_data"]["trading_performance"].equals(test_mixed_data["trading_performance"])
    assert result["encoded_data"]["overall_performance"].equals(test_mixed_data["overall_performance"])
    
    # Verify database entry
    db_record = db_session.query(EncodingRecord).filter_by(
        user_id=test_user.user_id,
        encoding_id=result["encoding_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_encode_with_label_encoding(db_session, test_user, test_categorical_data):
    """Test encoding with label encoding"""
    # Create encoding config with label encoding
    label_config = {
        "encoding_type": "label",
        "columns": None,  # Encode all categorical columns
        "exclude_columns": ["timestamp", "day"],
        "handle_unknown": "ignore"
    }
    
    # Encode categorical data
    result = encode_data(
        user_id=test_user.user_id,
        data=test_categorical_data,
        encoding_config=label_config,
        db_session=db_session
    )
    
    # Verify encoding result
    assert isinstance(result, Dict)
    assert "encoding_id" in result
    assert "encoded_data" in result
    
    # Verify encoding metadata
    assert result["encoding_type"] == "label"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["handle_unknown"] == "ignore"
    
    # Verify encoding details
    assert "encoding_details" in result
    assert isinstance(result["encoding_details"], Dict)
    assert "timestamp" in result["encoding_details"]
    assert "original_shape" in result["encoding_details"]
    assert "encoded_shape" in result["encoding_details"]
    assert "encoders" in result["encoding_details"]
    
    # Verify encoded data
    assert isinstance(result["encoded_data"], pd.DataFrame)
    assert "day" in result["encoded_data"].columns
    assert "mining_status" in result["encoded_data"].columns
    assert "staking_status" in result["encoded_data"].columns
    assert "trading_status" in result["encoded_data"].columns
    assert "overall_status" in result["encoded_data"].columns
    
    # Verify data encoding
    assert len(result["encoded_data"]) == len(test_categorical_data)
    # Check that label encoding is correct (values are integers)
    assert result["encoded_data"]["mining_status"].dtype in [np.int32, np.int64]
    assert result["encoded_data"]["staking_status"].dtype in [np.int32, np.int64]
    assert result["encoded_data"]["trading_status"].dtype in [np.int32, np.int64]
    assert result["encoded_data"]["overall_status"].dtype in [np.int32, np.int64]
    
    # Verify database entry
    db_record = db_session.query(EncodingRecord).filter_by(
        user_id=test_user.user_id,
        encoding_id=result["encoding_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_encode_with_ordinal_encoding(db_session, test_user, test_categorical_data):
    """Test encoding with ordinal encoding"""
    # Create encoding config with ordinal encoding
    ordinal_config = {
        "encoding_type": "ordinal",
        "columns": None,  # Encode all categorical columns
        "exclude_columns": ["timestamp", "day"],
        "handle_unknown": "ignore",
        "categories": {
            "mining_status": ["inactive", "maintenance", "active"],
            "staking_status": ["inactive", "pending", "active"],
            "trading_status": ["inactive", "suspended", "active"],
            "overall_status": ["poor", "fair", "good"]
        }
    }
    
    # Encode categorical data
    result = encode_data(
        user_id=test_user.user_id,
        data=test_categorical_data,
        encoding_config=ordinal_config,
        db_session=db_session
    )
    
    # Verify encoding result
    assert isinstance(result, Dict)
    assert "encoding_id" in result
    assert "encoded_data" in result
    
    # Verify encoding metadata
    assert result["encoding_type"] == "ordinal"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["handle_unknown"] == "ignore"
    assert result["categories"] == {
        "mining_status": ["inactive", "maintenance", "active"],
        "staking_status": ["inactive", "pending", "active"],
        "trading_status": ["inactive", "suspended", "active"],
        "overall_status": ["poor", "fair", "good"]
    }
    
    # Verify encoding details
    assert "encoding_details" in result
    assert isinstance(result["encoding_details"], Dict)
    assert "timestamp" in result["encoding_details"]
    assert "original_shape" in result["encoding_details"]
    assert "encoded_shape" in result["encoding_details"]
    assert "encoders" in result["encoding_details"]
    
    # Verify encoded data
    assert isinstance(result["encoded_data"], pd.DataFrame)
    assert "day" in result["encoded_data"].columns
    assert "mining_status" in result["encoded_data"].columns
    assert "staking_status" in result["encoded_data"].columns
    assert "trading_status" in result["encoded_data"].columns
    assert "overall_status" in result["encoded_data"].columns
    
    # Verify data encoding
    assert len(result["encoded_data"]) == len(test_categorical_data)
    # Check that ordinal encoding is correct (values are integers)
    assert result["encoded_data"]["mining_status"].dtype in [np.int32, np.int64]
    assert result["encoded_data"]["staking_status"].dtype in [np.int32, np.int64]
    assert result["encoded_data"]["trading_status"].dtype in [np.int32, np.int64]
    assert result["encoded_data"]["overall_status"].dtype in [np.int32, np.int64]
    
    # Verify database entry
    db_record = db_session.query(EncodingRecord).filter_by(
        user_id=test_user.user_id,
        encoding_id=result["encoding_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_encode_with_binary_encoding(db_session, test_user, test_categorical_data):
    """Test encoding with binary encoding"""
    # Create encoding config with binary encoding
    binary_config = {
        "encoding_type": "binary",
        "columns": None,  # Encode all categorical columns
        "exclude_columns": ["timestamp", "day"],
        "handle_unknown": "ignore"
    }
    
    # Encode categorical data
    result = encode_data(
        user_id=test_user.user_id,
        data=test_categorical_data,
        encoding_config=binary_config,
        db_session=db_session
    )
    
    # Verify encoding result
    assert isinstance(result, Dict)
    assert "encoding_id" in result
    assert "encoded_data" in result
    
    # Verify encoding metadata
    assert result["encoding_type"] == "binary"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["handle_unknown"] == "ignore"
    
    # Verify encoding details
    assert "encoding_details" in result
    assert isinstance(result["encoding_details"], Dict)
    assert "timestamp" in result["encoding_details"]
    assert "original_shape" in result["encoding_details"]
    assert "encoded_shape" in result["encoding_details"]
    assert "encoders" in result["encoding_details"]
    
    # Verify encoded data
    assert isinstance(result["encoded_data"], pd.DataFrame)
    assert "day" in result["encoded_data"].columns
    
    # Binary encoding creates fewer columns than one-hot encoding
    # For 3 categories, binary encoding uses 2 columns
    assert len([col for col in result["encoded_data"].columns if col.startswith("mining_status_")]) == 2
    assert len([col for col in result["encoded_data"].columns if col.startswith("staking_status_")]) == 2
    assert len([col for col in result["encoded_data"].columns if col.startswith("trading_status_")]) == 2
    assert len([col for col in result["encoded_data"].columns if col.startswith("overall_status_")]) == 2
    
    # Verify data encoding
    assert len(result["encoded_data"]) == len(test_categorical_data)
    
    # Verify database entry
    db_record = db_session.query(EncodingRecord).filter_by(
        user_id=test_user.user_id,
        encoding_id=result["encoding_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_encode_with_target_encoding(db_session, test_user, test_mixed_data):
    """Test encoding with target encoding"""
    # Create encoding config with target encoding
    target_config = {
        "encoding_type": "target",
        "columns": None,  # Encode all categorical columns
        "exclude_columns": ["timestamp", "day"],
        "handle_unknown": "ignore",
        "target_column": "overall_performance"
    }
    
    # Encode mixed data
    result = encode_data(
        user_id=test_user.user_id,
        data=test_mixed_data,
        encoding_config=target_config,
        db_session=db_session
    )
    
    # Verify encoding result
    assert isinstance(result, Dict)
    assert "encoding_id" in result
    assert "encoded_data" in result
    
    # Verify encoding metadata
    assert result["encoding_type"] == "target"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["handle_unknown"] == "ignore"
    assert result["target_column"] == "overall_performance"
    
    # Verify encoding details
    assert "encoding_details" in result
    assert isinstance(result["encoding_details"], Dict)
    assert "timestamp" in result["encoding_details"]
    assert "original_shape" in result["encoding_details"]
    assert "encoded_shape" in result["encoding_details"]
    assert "encoders" in result["encoding_details"]
    
    # Verify encoded data
    assert isinstance(result["encoded_data"], pd.DataFrame)
    assert "day" in result["encoded_data"].columns
    assert "mining_status" in result["encoded_data"].columns
    assert "staking_status" in result["encoded_data"].columns
    assert "trading_status" in result["encoded_data"].columns
    assert "overall_status" in result["encoded_data"].columns
    assert "mining_performance" in result["encoded_data"].columns
    assert "staking_performance" in result["encoded_data"].columns
    assert "trading_performance" in result["encoded_data"].columns
    assert "overall_performance" in result["encoded_data"].columns
    
    # Verify data encoding
    assert len(result["encoded_data"]) == len(test_mixed_data)
    # Check that target encoding is correct (values are floats)
    assert result["encoded_data"]["mining_status"].dtype in [np.float32, np.float64]
    assert result["encoded_data"]["staking_status"].dtype in [np.float32, np.float64]
    assert result["encoded_data"]["trading_status"].dtype in [np.float32, np.float64]
    assert result["encoded_data"]["overall_status"].dtype in [np.float32, np.float64]
    
    # Verify database entry
    db_record = db_session.query(EncodingRecord).filter_by(
        user_id=test_user.user_id,
        encoding_id=result["encoding_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_encode_with_specific_columns(db_session, test_user, test_categorical_data):
    """Test encoding with specific columns"""
    # Create encoding config with specific columns
    specific_columns_config = {
        "encoding_type": "onehot",
        "columns": ["mining_status", "trading_status"],  # Only encode these columns
        "exclude_columns": ["timestamp", "day"],
        "handle_unknown": "ignore"
    }
    
    # Encode categorical data
    result = encode_data(
        user_id=test_user.user_id,
        data=test_categorical_data,
        encoding_config=specific_columns_config,
        db_session=db_session
    )
    
    # Verify encoding result
    assert isinstance(result, Dict)
    assert "encoding_id" in result
    assert "encoded_data" in result
    
    # Verify encoding metadata
    assert result["encoding_type"] == "onehot"
    assert result["columns"] == ["mining_status", "trading_status"]
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["handle_unknown"] == "ignore"
    
    # Verify encoding details
    assert "encoding_details" in result
    assert isinstance(result["encoding_details"], Dict)
    assert "timestamp" in result["encoding_details"]
    assert "original_shape" in result["encoding_details"]
    assert "encoded_shape" in result["encoding_details"]
    assert "encoders" in result["encoding_details"]
    
    # Verify encoded data
    assert isinstance(result["encoded_data"], pd.DataFrame)
    assert "day" in result["encoded_data"].columns
    assert "mining_status_active" in result["encoded_data"].columns
    assert "mining_status_inactive" in result["encoded_data"].columns
    assert "mining_status_maintenance" in result["encoded_data"].columns
    assert "trading_status_active" in result["encoded_data"].columns
    assert "trading_status_inactive" in result["encoded_data"].columns
    assert "trading_status_suspended" in result["encoded_data"].columns
    assert "staking_status" in result["encoded_data"].columns
    assert "overall_status" in result["encoded_data"].columns
    
    # Verify data encoding
    assert len(result["encoded_data"]) == len(test_categorical_data)
    # Check that one-hot encoding is correct for encoded columns
    for i in range(len(result["encoded_data"])):
        assert result["encoded_data"].iloc[i][["mining_status_active", "mining_status_inactive", "mining_status_maintenance"]].sum() == 1
        assert result["encoded_data"].iloc[i][["trading_status_active", "trading_status_inactive", "trading_status_suspended"]].sum() == 1
    
    # Check that non-encoded columns are unchanged
    assert result["encoded_data"]["staking_status"].equals(test_categorical_data["staking_status"])
    assert result["encoded_data"]["overall_status"].equals(test_categorical_data["overall_status"])
    
    # Verify database entry
    db_record = db_session.query(EncodingRecord).filter_by(
        user_id=test_user.user_id,
        encoding_id=result["encoding_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_encode_with_unknown_handling(db_session, test_user, test_categorical_data):
    """Test encoding with unknown handling"""
    # Create encoding config with unknown handling
    unknown_config = {
        "encoding_type": "onehot",
        "columns": None,  # Encode all categorical columns
        "exclude_columns": ["timestamp", "day"],
        "handle_unknown": "error"  # Raise error for unknown categories
    }
    
    # Add an unknown category to the data
    test_data = test_categorical_data.copy()
    test_data.loc[0, "mining_status"] = "unknown_status"
    
    # Encode categorical data with unknown handling
    with pytest.raises(EncodingError) as excinfo:
        encode_data(
            user_id=test_user.user_id,
            data=test_data,
            encoding_config=unknown_config,
            db_session=db_session
        )
    assert "Unknown category" in str(excinfo.value)
    
    # Create encoding config with unknown handling set to "use_encoded_value"
    unknown_config = {
        "encoding_type": "onehot",
        "columns": None,  # Encode all categorical columns
        "exclude_columns": ["timestamp", "day"],
        "handle_unknown": "use_encoded_value",
        "unknown_value": 0  # Use 0 for unknown categories
    }
    
    # Encode categorical data with unknown handling
    result = encode_data(
        user_id=test_user.user_id,
        data=test_data,
        encoding_config=unknown_config,
        db_session=db_session
    )
    
    # Verify encoding result
    assert isinstance(result, Dict)
    assert "encoding_id" in result
    assert "encoded_data" in result
    
    # Verify encoding metadata
    assert result["encoding_type"] == "onehot"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["handle_unknown"] == "use_encoded_value"
    assert result["unknown_value"] == 0
    
    # Verify database entry
    db_record = db_session.query(EncodingRecord).filter_by(
        user_id=test_user.user_id,
        encoding_id=result["encoding_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_get_encoding_info(db_session, test_user, test_categorical_data, test_encoding_config):
    """Test encoding info retrieval"""
    # First, encode categorical data
    encoding_result = encode_data(
        user_id=test_user.user_id,
        data=test_categorical_data,
        encoding_config=test_encoding_config,
        db_session=db_session
    )
    
    encoding_id = encoding_result["encoding_id"]
    
    # Get encoding info
    result = get_encoding_info(
        user_id=test_user.user_id,
        encoding_id=encoding_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "encoding_id" in result
    assert result["encoding_id"] == encoding_id
    
    # Verify encoding metadata
    assert result["encoding_type"] == "onehot"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["handle_unknown"] == "ignore"
    
    # Verify encoding details
    assert "encoding_details" in result
    assert isinstance(result["encoding_details"], Dict)
    assert "timestamp" in result["encoding_details"]
    assert "original_shape" in result["encoding_details"]
    assert "encoded_shape" in result["encoding_details"]
    assert "encoders" in result["encoding_details"]
    
    # Verify database entry
    db_record = db_session.query(EncodingRecord).filter_by(
        user_id=test_user.user_id,
        encoding_id=encoding_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_encoding_error_handling(db_session, test_user):
    """Test encoding error handling"""
    # Invalid user ID
    with pytest.raises(EncodingError) as excinfo:
        encode_data(
            user_id=None,
            data=pd.DataFrame(),
            encoding_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(EncodingError) as excinfo:
        encode_data(
            user_id=test_user.user_id,
            data=None,
            encoding_config={},
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Invalid encoding type
    with pytest.raises(EncodingError) as excinfo:
        encode_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": ["a", "b", "c"]}),
            encoding_config={"encoding_type": "invalid_type"},
            db_session=db_session
        )
    assert "Invalid encoding type" in str(excinfo.value)
    
    # Invalid columns
    with pytest.raises(EncodingError) as excinfo:
        encode_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": ["a", "b", "c"]}),
            encoding_config={"encoding_type": "onehot", "columns": ["invalid_column"]},
            db_session=db_session
        )
    assert "Invalid columns" in str(excinfo.value)
    
    # Invalid encoding ID
    with pytest.raises(EncodingError) as excinfo:
        get_encoding_info(
            user_id=test_user.user_id,
            encoding_id="invalid_encoding_id",
            db_session=db_session
        )
    assert "Invalid encoding ID" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBEncodingError) as excinfo:
        encode_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": ["a", "b", "c"]}),
            encoding_config={"encoding_type": "onehot"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 