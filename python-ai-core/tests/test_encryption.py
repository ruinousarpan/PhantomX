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
import base64
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from core.encryption import (
    encrypt_data,
    decrypt_data,
    generate_key,
    rotate_key,
    validate_key,
    EncryptionError
)
from database.models import User, EncryptionKey
from database.exceptions import EncryptionError as DBEncryptionError

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
def test_encryption_config():
    """Create test encryption configuration"""
    return {
        "algorithm": "aes-256-gcm",
        "key_derivation": "pbkdf2",
        "iterations": 100000,
        "salt_length": 16,
        "key_length": 32,
        "tag_length": 16,
        "nonce_length": 12,
        "options": {
            "verify": True,
            "checksum": "sha256",
            "metadata": True,
            "rotation": {
                "enabled": True,
                "interval": 30,  # days
                "grace_period": 7  # days
            }
        }
    }

def test_generate_key(db_session, test_user, test_encryption_config):
    """Test key generation"""
    # Generate encryption key
    result = generate_key(
        user_id=test_user.user_id,
        encryption_config=test_encryption_config,
        db_session=db_session
    )
    
    # Verify key result
    assert isinstance(result, Dict)
    assert "key_id" in result
    assert "key" in result
    
    # Verify key metadata
    assert result["algorithm"] == "aes-256-gcm"
    assert result["key_derivation"] == "pbkdf2"
    
    # Verify key details
    assert "key_details" in result
    assert isinstance(result["key_details"], Dict)
    assert "created_at" in result["key_details"]
    assert "expires_at" in result["key_details"]
    assert "status" in result["key_details"]
    assert result["key_details"]["status"] == "ACTIVE"
    
    # Verify database entry
    db_key = db_session.query(EncryptionKey).filter_by(
        user_id=test_user.user_id,
        key_id=result["key_id"]
    ).first()
    assert db_key is not None
    assert db_key.is_active is True
    assert db_key.error is None

def test_encrypt_performance_data(db_session, test_user, test_performance_data, test_encryption_config):
    """Test performance data encryption"""
    # First, generate a key
    key_result = generate_key(
        user_id=test_user.user_id,
        encryption_config=test_encryption_config,
        db_session=db_session
    )
    
    key_id = key_result["key_id"]
    
    # Encrypt performance data
    result = encrypt_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        key_id=key_id,
        encryption_config=test_encryption_config,
        db_session=db_session
    )
    
    # Verify encryption result
    assert isinstance(result, Dict)
    assert "encryption_id" in result
    assert "encrypted_data" in result
    
    # Verify encryption metadata
    assert result["algorithm"] == "aes-256-gcm"
    assert result["key_id"] == key_id
    
    # Verify encryption details
    assert "encryption_details" in result
    assert isinstance(result["encryption_details"], Dict)
    assert "timestamp" in result["encryption_details"]
    assert "size" in result["encryption_details"]
    assert "checksum" in result["encryption_details"]
    
    # Verify encrypted data is not the same as original
    assert result["encrypted_data"] != test_performance_data.to_json()
    
    # Verify database entry
    db_key = db_session.query(EncryptionKey).filter_by(
        user_id=test_user.user_id,
        key_id=key_id
    ).first()
    assert db_key is not None
    assert db_key.is_active is True
    assert db_key.error is None

def test_encrypt_risk_data(db_session, test_user, test_risk_data, test_encryption_config):
    """Test risk data encryption"""
    # First, generate a key
    key_result = generate_key(
        user_id=test_user.user_id,
        encryption_config=test_encryption_config,
        db_session=db_session
    )
    
    key_id = key_result["key_id"]
    
    # Encrypt risk data
    result = encrypt_data(
        user_id=test_user.user_id,
        data=test_risk_data,
        key_id=key_id,
        encryption_config=test_encryption_config,
        db_session=db_session
    )
    
    # Verify encryption result
    assert isinstance(result, Dict)
    assert "encryption_id" in result
    assert "encrypted_data" in result
    
    # Verify encryption metadata
    assert result["algorithm"] == "aes-256-gcm"
    assert result["key_id"] == key_id
    
    # Verify encryption details
    assert "encryption_details" in result
    assert isinstance(result["encryption_details"], Dict)
    assert "timestamp" in result["encryption_details"]
    assert "size" in result["encryption_details"]
    assert "checksum" in result["encryption_details"]
    
    # Verify encrypted data is not the same as original
    assert result["encrypted_data"] != test_risk_data.to_json()
    
    # Verify database entry
    db_key = db_session.query(EncryptionKey).filter_by(
        user_id=test_user.user_id,
        key_id=key_id
    ).first()
    assert db_key is not None
    assert db_key.is_active is True
    assert db_key.error is None

def test_encrypt_reward_data(db_session, test_user, test_reward_data, test_encryption_config):
    """Test reward data encryption"""
    # First, generate a key
    key_result = generate_key(
        user_id=test_user.user_id,
        encryption_config=test_encryption_config,
        db_session=db_session
    )
    
    key_id = key_result["key_id"]
    
    # Encrypt reward data
    result = encrypt_data(
        user_id=test_user.user_id,
        data=test_reward_data,
        key_id=key_id,
        encryption_config=test_encryption_config,
        db_session=db_session
    )
    
    # Verify encryption result
    assert isinstance(result, Dict)
    assert "encryption_id" in result
    assert "encrypted_data" in result
    
    # Verify encryption metadata
    assert result["algorithm"] == "aes-256-gcm"
    assert result["key_id"] == key_id
    
    # Verify encryption details
    assert "encryption_details" in result
    assert isinstance(result["encryption_details"], Dict)
    assert "timestamp" in result["encryption_details"]
    assert "size" in result["encryption_details"]
    assert "checksum" in result["encryption_details"]
    
    # Verify encrypted data is not the same as original
    assert result["encrypted_data"] != test_reward_data.to_json()
    
    # Verify database entry
    db_key = db_session.query(EncryptionKey).filter_by(
        user_id=test_user.user_id,
        key_id=key_id
    ).first()
    assert db_key is not None
    assert db_key.is_active is True
    assert db_key.error is None

def test_encrypt_activity_data(db_session, test_user, test_activity_data, test_encryption_config):
    """Test activity data encryption"""
    # First, generate a key
    key_result = generate_key(
        user_id=test_user.user_id,
        encryption_config=test_encryption_config,
        db_session=db_session
    )
    
    key_id = key_result["key_id"]
    
    # Encrypt activity data
    result = encrypt_data(
        user_id=test_user.user_id,
        data=test_activity_data,
        key_id=key_id,
        encryption_config=test_encryption_config,
        db_session=db_session
    )
    
    # Verify encryption result
    assert isinstance(result, Dict)
    assert "encryption_id" in result
    assert "encrypted_data" in result
    
    # Verify encryption metadata
    assert result["algorithm"] == "aes-256-gcm"
    assert result["key_id"] == key_id
    
    # Verify encryption details
    assert "encryption_details" in result
    assert isinstance(result["encryption_details"], Dict)
    assert "timestamp" in result["encryption_details"]
    assert "size" in result["encryption_details"]
    assert "checksum" in result["encryption_details"]
    
    # Verify encrypted data is not the same as original
    assert result["encrypted_data"] != test_activity_data.to_json()
    
    # Verify database entry
    db_key = db_session.query(EncryptionKey).filter_by(
        user_id=test_user.user_id,
        key_id=key_id
    ).first()
    assert db_key is not None
    assert db_key.is_active is True
    assert db_key.error is None

def test_encrypt_analytics_data(db_session, test_user, test_analytics_data, test_encryption_config):
    """Test analytics data encryption"""
    # First, generate a key
    key_result = generate_key(
        user_id=test_user.user_id,
        encryption_config=test_encryption_config,
        db_session=db_session
    )
    
    key_id = key_result["key_id"]
    
    # Encrypt analytics data
    result = encrypt_data(
        user_id=test_user.user_id,
        data=test_analytics_data,
        key_id=key_id,
        encryption_config=test_encryption_config,
        db_session=db_session
    )
    
    # Verify encryption result
    assert isinstance(result, Dict)
    assert "encryption_id" in result
    assert "encrypted_data" in result
    
    # Verify encryption metadata
    assert result["algorithm"] == "aes-256-gcm"
    assert result["key_id"] == key_id
    
    # Verify encryption details
    assert "encryption_details" in result
    assert isinstance(result["encryption_details"], Dict)
    assert "timestamp" in result["encryption_details"]
    assert "size" in result["encryption_details"]
    assert "checksum" in result["encryption_details"]
    
    # Verify encrypted data is not the same as original
    assert result["encrypted_data"] != json.dumps(test_analytics_data)
    
    # Verify database entry
    db_key = db_session.query(EncryptionKey).filter_by(
        user_id=test_user.user_id,
        key_id=key_id
    ).first()
    assert db_key is not None
    assert db_key.is_active is True
    assert db_key.error is None

def test_decrypt_data(db_session, test_user, test_performance_data, test_encryption_config):
    """Test data decryption"""
    # First, generate a key
    key_result = generate_key(
        user_id=test_user.user_id,
        encryption_config=test_encryption_config,
        db_session=db_session
    )
    
    key_id = key_result["key_id"]
    
    # Encrypt performance data
    encrypt_result = encrypt_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        key_id=key_id,
        encryption_config=test_encryption_config,
        db_session=db_session
    )
    
    encryption_id = encrypt_result["encryption_id"]
    encrypted_data = encrypt_result["encrypted_data"]
    
    # Decrypt data
    result = decrypt_data(
        user_id=test_user.user_id,
        encrypted_data=encrypted_data,
        key_id=key_id,
        encryption_config=test_encryption_config,
        db_session=db_session
    )
    
    # Verify decryption result
    assert isinstance(result, Dict)
    assert "decryption_id" in result
    assert "decrypted_data" in result
    
    # Verify decrypted data
    assert isinstance(result["decrypted_data"], pd.DataFrame)
    assert "timestamp" in result["decrypted_data"].columns
    assert "mining_performance" in result["decrypted_data"].columns
    assert "staking_performance" in result["decrypted_data"].columns
    assert "trading_performance" in result["decrypted_data"].columns
    assert "overall_performance" in result["decrypted_data"].columns
    
    # Verify data integrity
    pd.testing.assert_frame_equal(result["decrypted_data"], test_performance_data)
    
    # Verify database entry
    db_key = db_session.query(EncryptionKey).filter_by(
        user_id=test_user.user_id,
        key_id=key_id
    ).first()
    assert db_key is not None
    assert db_key.is_active is True
    assert db_key.error is None

def test_rotate_key(db_session, test_user, test_encryption_config):
    """Test key rotation"""
    # First, generate a key
    key_result = generate_key(
        user_id=test_user.user_id,
        encryption_config=test_encryption_config,
        db_session=db_session
    )
    
    key_id = key_result["key_id"]
    
    # Rotate key
    result = rotate_key(
        user_id=test_user.user_id,
        key_id=key_id,
        encryption_config=test_encryption_config,
        db_session=db_session
    )
    
    # Verify rotation result
    assert isinstance(result, Dict)
    assert "old_key_id" in result
    assert "new_key_id" in result
    assert result["old_key_id"] == key_id
    
    # Verify key metadata
    assert result["algorithm"] == "aes-256-gcm"
    assert result["key_derivation"] == "pbkdf2"
    
    # Verify key details
    assert "key_details" in result
    assert isinstance(result["key_details"], Dict)
    assert "created_at" in result["key_details"]
    assert "expires_at" in result["key_details"]
    assert "status" in result["key_details"]
    assert result["key_details"]["status"] == "ACTIVE"
    
    # Verify database entries
    old_key = db_session.query(EncryptionKey).filter_by(
        user_id=test_user.user_id,
        key_id=key_id
    ).first()
    assert old_key is not None
    assert old_key.is_active is False
    assert old_key.error is None
    
    new_key = db_session.query(EncryptionKey).filter_by(
        user_id=test_user.user_id,
        key_id=result["new_key_id"]
    ).first()
    assert new_key is not None
    assert new_key.is_active is True
    assert new_key.error is None

def test_validate_key(db_session, test_user, test_encryption_config):
    """Test key validation"""
    # First, generate a key
    key_result = generate_key(
        user_id=test_user.user_id,
        encryption_config=test_encryption_config,
        db_session=db_session
    )
    
    key_id = key_result["key_id"]
    
    # Validate key
    result = validate_key(
        user_id=test_user.user_id,
        key_id=key_id,
        db_session=db_session
    )
    
    # Verify validation result
    assert isinstance(result, Dict)
    assert "key_id" in result
    assert "is_valid" in result
    assert result["is_valid"] is True
    
    # Verify key metadata
    assert result["algorithm"] == "aes-256-gcm"
    assert result["key_derivation"] == "pbkdf2"
    
    # Verify key details
    assert "key_details" in result
    assert isinstance(result["key_details"], Dict)
    assert "created_at" in result["key_details"]
    assert "expires_at" in result["key_details"]
    assert "status" in result["key_details"]
    assert result["key_details"]["status"] == "ACTIVE"
    
    # Verify database entry
    db_key = db_session.query(EncryptionKey).filter_by(
        user_id=test_user.user_id,
        key_id=key_id
    ).first()
    assert db_key is not None
    assert db_key.is_active is True
    assert db_key.error is None

def test_encryption_error_handling(db_session, test_user):
    """Test encryption error handling"""
    # Invalid user ID
    with pytest.raises(EncryptionError) as excinfo:
        generate_key(
            user_id=None,
            encryption_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid encryption algorithm
    with pytest.raises(EncryptionError) as excinfo:
        generate_key(
            user_id=test_user.user_id,
            encryption_config={"algorithm": "invalid_algorithm"},
            db_session=db_session
        )
    assert "Invalid encryption algorithm" in str(excinfo.value)
    
    # Invalid key derivation
    with pytest.raises(EncryptionError) as excinfo:
        generate_key(
            user_id=test_user.user_id,
            encryption_config={"key_derivation": "invalid_derivation"},
            db_session=db_session
        )
    assert "Invalid key derivation" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(EncryptionError) as excinfo:
        encrypt_data(
            user_id=test_user.user_id,
            data=None,
            key_id="test_key_id",
            encryption_config={},
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Invalid key ID
    with pytest.raises(EncryptionError) as excinfo:
        encrypt_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            key_id="invalid_key_id",
            encryption_config={},
            db_session=db_session
        )
    assert "Invalid key ID" in str(excinfo.value)
    
    # Invalid encrypted data
    with pytest.raises(EncryptionError) as excinfo:
        decrypt_data(
            user_id=test_user.user_id,
            encrypted_data=None,
            key_id="test_key_id",
            encryption_config={},
            db_session=db_session
        )
    assert "Invalid encrypted data" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBEncryptionError) as excinfo:
        generate_key(
            user_id=test_user.user_id,
            encryption_config={"algorithm": "aes-256-gcm"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 