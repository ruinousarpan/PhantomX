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
import zlib
import gzip
import bz2
import lzma
import base64
import hashlib
import io

from core.compression import (
    compress_data,
    decompress_data,
    get_compression_info,
    CompressionError
)
from database.models import User, CompressionRecord
from database.exceptions import CompressionError as DBCompressionError

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
def test_compression_config():
    """Create test compression configuration"""
    return {
        "algorithm": "gzip",
        "level": 9,
        "options": {
            "verify": True,
            "checksum": "sha256",
            "metadata": True,
            "chunk_size": 1024 * 1024,  # 1MB
            "threads": 4
        }
    }

def test_compress_performance_data(db_session, test_user, test_performance_data, test_compression_config):
    """Test performance data compression"""
    # Compress performance data
    result = compress_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        compression_config=test_compression_config,
        db_session=db_session
    )
    
    # Verify compression result
    assert isinstance(result, Dict)
    assert "compression_id" in result
    assert "compressed_data" in result
    
    # Verify compression metadata
    assert result["algorithm"] == "gzip"
    assert result["level"] == 9
    
    # Verify compression details
    assert "compression_details" in result
    assert isinstance(result["compression_details"], Dict)
    assert "timestamp" in result["compression_details"]
    assert "original_size" in result["compression_details"]
    assert "compressed_size" in result["compression_details"]
    assert "ratio" in result["compression_details"]
    assert "checksum" in result["compression_details"]
    
    # Verify compression ratio is reasonable
    assert result["compression_details"]["ratio"] > 0
    assert result["compression_details"]["ratio"] < 1
    
    # Verify compressed data is not the same as original
    assert result["compressed_data"] != test_performance_data.to_json()
    
    # Verify database entry
    db_record = db_session.query(CompressionRecord).filter_by(
        user_id=test_user.user_id,
        compression_id=result["compression_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_compress_risk_data(db_session, test_user, test_risk_data, test_compression_config):
    """Test risk data compression"""
    # Compress risk data
    result = compress_data(
        user_id=test_user.user_id,
        data=test_risk_data,
        compression_config=test_compression_config,
        db_session=db_session
    )
    
    # Verify compression result
    assert isinstance(result, Dict)
    assert "compression_id" in result
    assert "compressed_data" in result
    
    # Verify compression metadata
    assert result["algorithm"] == "gzip"
    assert result["level"] == 9
    
    # Verify compression details
    assert "compression_details" in result
    assert isinstance(result["compression_details"], Dict)
    assert "timestamp" in result["compression_details"]
    assert "original_size" in result["compression_details"]
    assert "compressed_size" in result["compression_details"]
    assert "ratio" in result["compression_details"]
    assert "checksum" in result["compression_details"]
    
    # Verify compression ratio is reasonable
    assert result["compression_details"]["ratio"] > 0
    assert result["compression_details"]["ratio"] < 1
    
    # Verify compressed data is not the same as original
    assert result["compressed_data"] != test_risk_data.to_json()
    
    # Verify database entry
    db_record = db_session.query(CompressionRecord).filter_by(
        user_id=test_user.user_id,
        compression_id=result["compression_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_compress_reward_data(db_session, test_user, test_reward_data, test_compression_config):
    """Test reward data compression"""
    # Compress reward data
    result = compress_data(
        user_id=test_user.user_id,
        data=test_reward_data,
        compression_config=test_compression_config,
        db_session=db_session
    )
    
    # Verify compression result
    assert isinstance(result, Dict)
    assert "compression_id" in result
    assert "compressed_data" in result
    
    # Verify compression metadata
    assert result["algorithm"] == "gzip"
    assert result["level"] == 9
    
    # Verify compression details
    assert "compression_details" in result
    assert isinstance(result["compression_details"], Dict)
    assert "timestamp" in result["compression_details"]
    assert "original_size" in result["compression_details"]
    assert "compressed_size" in result["compression_details"]
    assert "ratio" in result["compression_details"]
    assert "checksum" in result["compression_details"]
    
    # Verify compression ratio is reasonable
    assert result["compression_details"]["ratio"] > 0
    assert result["compression_details"]["ratio"] < 1
    
    # Verify compressed data is not the same as original
    assert result["compressed_data"] != test_reward_data.to_json()
    
    # Verify database entry
    db_record = db_session.query(CompressionRecord).filter_by(
        user_id=test_user.user_id,
        compression_id=result["compression_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_compress_activity_data(db_session, test_user, test_activity_data, test_compression_config):
    """Test activity data compression"""
    # Compress activity data
    result = compress_data(
        user_id=test_user.user_id,
        data=test_activity_data,
        compression_config=test_compression_config,
        db_session=db_session
    )
    
    # Verify compression result
    assert isinstance(result, Dict)
    assert "compression_id" in result
    assert "compressed_data" in result
    
    # Verify compression metadata
    assert result["algorithm"] == "gzip"
    assert result["level"] == 9
    
    # Verify compression details
    assert "compression_details" in result
    assert isinstance(result["compression_details"], Dict)
    assert "timestamp" in result["compression_details"]
    assert "original_size" in result["compression_details"]
    assert "compressed_size" in result["compression_details"]
    assert "ratio" in result["compression_details"]
    assert "checksum" in result["compression_details"]
    
    # Verify compression ratio is reasonable
    assert result["compression_details"]["ratio"] > 0
    assert result["compression_details"]["ratio"] < 1
    
    # Verify compressed data is not the same as original
    assert result["compressed_data"] != test_activity_data.to_json()
    
    # Verify database entry
    db_record = db_session.query(CompressionRecord).filter_by(
        user_id=test_user.user_id,
        compression_id=result["compression_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_compress_analytics_data(db_session, test_user, test_analytics_data, test_compression_config):
    """Test analytics data compression"""
    # Compress analytics data
    result = compress_data(
        user_id=test_user.user_id,
        data=test_analytics_data,
        compression_config=test_compression_config,
        db_session=db_session
    )
    
    # Verify compression result
    assert isinstance(result, Dict)
    assert "compression_id" in result
    assert "compressed_data" in result
    
    # Verify compression metadata
    assert result["algorithm"] == "gzip"
    assert result["level"] == 9
    
    # Verify compression details
    assert "compression_details" in result
    assert isinstance(result["compression_details"], Dict)
    assert "timestamp" in result["compression_details"]
    assert "original_size" in result["compression_details"]
    assert "compressed_size" in result["compression_details"]
    assert "ratio" in result["compression_details"]
    assert "checksum" in result["compression_details"]
    
    # Verify compression ratio is reasonable
    assert result["compression_details"]["ratio"] > 0
    assert result["compression_details"]["ratio"] < 1
    
    # Verify compressed data is not the same as original
    assert result["compressed_data"] != json.dumps(test_analytics_data)
    
    # Verify database entry
    db_record = db_session.query(CompressionRecord).filter_by(
        user_id=test_user.user_id,
        compression_id=result["compression_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_decompress_data(db_session, test_user, test_performance_data, test_compression_config):
    """Test data decompression"""
    # First, compress performance data
    compress_result = compress_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        compression_config=test_compression_config,
        db_session=db_session
    )
    
    compression_id = compress_result["compression_id"]
    compressed_data = compress_result["compressed_data"]
    
    # Decompress data
    result = decompress_data(
        user_id=test_user.user_id,
        compressed_data=compressed_data,
        compression_config=test_compression_config,
        db_session=db_session
    )
    
    # Verify decompression result
    assert isinstance(result, Dict)
    assert "decompression_id" in result
    assert "decompressed_data" in result
    
    # Verify decompressed data
    assert isinstance(result["decompressed_data"], pd.DataFrame)
    assert "timestamp" in result["decompressed_data"].columns
    assert "mining_performance" in result["decompressed_data"].columns
    assert "staking_performance" in result["decompressed_data"].columns
    assert "trading_performance" in result["decompressed_data"].columns
    assert "overall_performance" in result["decompressed_data"].columns
    
    # Verify data integrity
    pd.testing.assert_frame_equal(result["decompressed_data"], test_performance_data)
    
    # Verify database entry
    db_record = db_session.query(CompressionRecord).filter_by(
        user_id=test_user.user_id,
        compression_id=compression_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_get_compression_info(db_session, test_user, test_performance_data, test_compression_config):
    """Test compression info retrieval"""
    # First, compress performance data
    compress_result = compress_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        compression_config=test_compression_config,
        db_session=db_session
    )
    
    compression_id = compress_result["compression_id"]
    
    # Get compression info
    result = get_compression_info(
        user_id=test_user.user_id,
        compression_id=compression_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "compression_id" in result
    assert result["compression_id"] == compression_id
    
    # Verify compression metadata
    assert result["algorithm"] == "gzip"
    assert result["level"] == 9
    
    # Verify compression details
    assert "compression_details" in result
    assert isinstance(result["compression_details"], Dict)
    assert "timestamp" in result["compression_details"]
    assert "original_size" in result["compression_details"]
    assert "compressed_size" in result["compression_details"]
    assert "ratio" in result["compression_details"]
    assert "checksum" in result["compression_details"]
    
    # Verify database entry
    db_record = db_session.query(CompressionRecord).filter_by(
        user_id=test_user.user_id,
        compression_id=compression_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_compression_error_handling(db_session, test_user):
    """Test compression error handling"""
    # Invalid user ID
    with pytest.raises(CompressionError) as excinfo:
        compress_data(
            user_id=None,
            data=pd.DataFrame(),
            compression_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(CompressionError) as excinfo:
        compress_data(
            user_id=test_user.user_id,
            data=None,
            compression_config={},
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Invalid compression algorithm
    with pytest.raises(CompressionError) as excinfo:
        compress_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            compression_config={"algorithm": "invalid_algorithm"},
            db_session=db_session
        )
    assert "Invalid compression algorithm" in str(excinfo.value)
    
    # Invalid compression level
    with pytest.raises(CompressionError) as excinfo:
        compress_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            compression_config={"algorithm": "gzip", "level": 10},
            db_session=db_session
        )
    assert "Invalid compression level" in str(excinfo.value)
    
    # Invalid compressed data
    with pytest.raises(CompressionError) as excinfo:
        decompress_data(
            user_id=test_user.user_id,
            compressed_data=None,
            compression_config={},
            db_session=db_session
        )
    assert "Invalid compressed data" in str(excinfo.value)
    
    # Invalid compression ID
    with pytest.raises(CompressionError) as excinfo:
        get_compression_info(
            user_id=test_user.user_id,
            compression_id="invalid_compression_id",
            db_session=db_session
        )
    assert "Invalid compression ID" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBCompressionError) as excinfo:
        compress_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            compression_config={"algorithm": "gzip"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 