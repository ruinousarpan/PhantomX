import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import uuid
import bcrypt
import jwt
from cryptography.fernet import Fernet
from unittest.mock import patch, MagicMock

from core.security import (
    authenticate_user,
    authorize_access,
    encrypt_data,
    decrypt_data,
    create_access_policy,
    update_access_policy,
    delete_access_policy,
    get_access_policies,
    verify_access,
    log_security_event,
    get_audit_logs,
    SecurityError
)
from database.models import User, AccessPolicy, SecurityAuditLog
from database.exceptions import SecurityError as DBSecurityError

@pytest.fixture
def test_user(db_session):
    """Create test user"""
    # Hash password
    password = "test_password123"
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    
    user = User(
        user_id="test_user",
        username="testuser",
        email="test@example.com",
        password_hash=hashed_password,
        salt=salt,
        first_name="Test",
        last_name="User",
        role="USER",
        status="ACTIVE",
        email_verified=True,
        created_at=datetime.utcnow(),
        last_login=datetime.utcnow(),
        failed_login_attempts=0,
        last_password_change=datetime.utcnow()
    )
    db_session.add(user)
    db_session.commit()
    return user

@pytest.fixture
def test_access_policy():
    """Create test access policy configuration"""
    return {
        "name": "data_access_policy",
        "description": "Data access control policy",
        "version": "1.0.0",
        "resources": [
            {
                "type": "data_source",
                "id": "*",
                "actions": ["read", "write", "delete"]
            },
            {
                "type": "workflow",
                "id": "*",
                "actions": ["execute", "schedule", "monitor"]
            }
        ],
        "conditions": {
            "ip_range": ["192.168.0.0/16"],
            "time_window": {
                "start": "09:00:00",
                "end": "17:00:00",
                "timezone": "UTC"
            },
            "mfa_required": True
        },
        "permissions": {
            "roles": {
                "admin": ["*"],
                "data_scientist": ["read", "execute", "monitor"],
                "analyst": ["read", "monitor"]
            },
            "users": {
                "test_user": ["read", "write", "execute"]
            }
        },
        "encryption": {
            "algorithm": "AES-256-GCM",
            "key_rotation": {
                "enabled": True,
                "interval_days": 90
            },
            "data_classification": {
                "pii": "encrypt",
                "sensitive": "encrypt",
                "public": "none"
            }
        },
        "audit": {
            "enabled": True,
            "log_level": "INFO",
            "retention_days": 365,
            "events": [
                "authentication",
                "authorization",
                "access",
                "encryption",
                "policy_change"
            ]
        }
    }

def test_authenticate_user(db_session, test_user):
    """Test user authentication"""
    # Test successful authentication
    result = authenticate_user(
        username="testuser",
        password="test_password123",
        db_session=db_session
    )
    
    # Verify authentication result
    assert isinstance(result, Dict)
    assert "user_id" in result
    assert result["user_id"] == test_user.user_id
    assert "token" in result
    assert "expires_at" in result
    
    # Verify token
    token = result["token"]
    decoded = jwt.decode(token, "secret_key", algorithms=["HS256"])
    assert decoded["user_id"] == test_user.user_id
    assert decoded["role"] == "USER"
    
    # Verify audit log
    audit_log = db_session.query(SecurityAuditLog).filter_by(
        user_id=test_user.user_id,
        event_type="authentication",
        status="success"
    ).first()
    assert audit_log is not None
    
    # Test failed authentication
    with pytest.raises(SecurityError) as excinfo:
        authenticate_user(
            username="testuser",
            password="wrong_password",
            db_session=db_session
        )
    assert "Invalid credentials" in str(excinfo.value)
    
    # Verify failed login attempt is logged
    user = db_session.query(User).filter_by(user_id=test_user.user_id).first()
    assert user.failed_login_attempts == 1
    
    # Verify failed authentication audit log
    failed_audit = db_session.query(SecurityAuditLog).filter_by(
        user_id=test_user.user_id,
        event_type="authentication",
        status="failure"
    ).first()
    assert failed_audit is not None

def test_authorize_access(db_session, test_user, test_access_policy):
    """Test access authorization"""
    # First, create access policy
    policy_result = create_access_policy(
        policy_config=test_access_policy,
        db_session=db_session
    )
    
    # Test successful authorization
    result = authorize_access(
        user_id=test_user.user_id,
        resource_type="data_source",
        resource_id="test_source",
        action="read",
        db_session=db_session
    )
    
    # Verify authorization result
    assert isinstance(result, Dict)
    assert "authorized" in result
    assert result["authorized"] is True
    assert "policy_id" in result
    assert "permissions" in result
    
    # Verify audit log
    audit_log = db_session.query(SecurityAuditLog).filter_by(
        user_id=test_user.user_id,
        event_type="authorization",
        status="success"
    ).first()
    assert audit_log is not None
    
    # Test unauthorized access
    result = authorize_access(
        user_id=test_user.user_id,
        resource_type="workflow",
        resource_id="test_workflow",
        action="delete",
        db_session=db_session
    )
    
    # Verify unauthorized result
    assert isinstance(result, Dict)
    assert "authorized" in result
    assert result["authorized"] is False
    assert "reason" in result
    
    # Verify unauthorized audit log
    unauth_audit = db_session.query(SecurityAuditLog).filter_by(
        user_id=test_user.user_id,
        event_type="authorization",
        status="failure"
    ).first()
    assert unauth_audit is not None

def test_encrypt_decrypt_data(db_session, test_user):
    """Test data encryption and decryption"""
    # Generate encryption key
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)
    
    # Test data to encrypt
    sensitive_data = {
        "user_id": test_user.user_id,
        "credit_card": "4111-1111-1111-1111",
        "ssn": "123-45-6789"
    }
    
    # Encrypt data
    result = encrypt_data(
        data=sensitive_data,
        encryption_key=key,
        data_type="pii",
        db_session=db_session
    )
    
    # Verify encryption result
    assert isinstance(result, Dict)
    assert "encrypted_data" in result
    assert "encryption_id" in result
    assert "metadata" in result
    
    # Verify encrypted data is different from original
    assert result["encrypted_data"] != json.dumps(sensitive_data)
    
    # Verify encryption audit log
    encrypt_audit = db_session.query(SecurityAuditLog).filter_by(
        user_id=test_user.user_id,
        event_type="encryption",
        status="success"
    ).first()
    assert encrypt_audit is not None
    
    # Decrypt data
    decrypt_result = decrypt_data(
        encrypted_data=result["encrypted_data"],
        encryption_key=key,
        encryption_id=result["encryption_id"],
        db_session=db_session
    )
    
    # Verify decryption result
    assert isinstance(decrypt_result, Dict)
    assert decrypt_result == sensitive_data
    
    # Verify decryption audit log
    decrypt_audit = db_session.query(SecurityAuditLog).filter_by(
        user_id=test_user.user_id,
        event_type="decryption",
        status="success"
    ).first()
    assert decrypt_audit is not None

def test_access_policy_management(db_session, test_access_policy):
    """Test access policy management"""
    # Create access policy
    result = create_access_policy(
        policy_config=test_access_policy,
        db_session=db_session
    )
    
    # Verify creation result
    assert isinstance(result, Dict)
    assert "policy_id" in result
    assert "policy_details" in result
    
    policy_id = result["policy_id"]
    
    # Update access policy
    updated_policy = test_access_policy.copy()
    updated_policy["version"] = "1.1.0"
    updated_policy["permissions"]["roles"]["analyst"].append("execute")
    
    update_result = update_access_policy(
        policy_id=policy_id,
        policy_config=updated_policy,
        db_session=db_session
    )
    
    # Verify update result
    assert isinstance(update_result, Dict)
    assert update_result["policy_id"] == policy_id
    assert update_result["version"] == "1.1.0"
    assert "execute" in update_result["permissions"]["roles"]["analyst"]
    
    # Get access policies
    policies = get_access_policies(db_session=db_session)
    
    # Verify policies list
    assert isinstance(policies, List)
    assert len(policies) > 0
    assert any(p["policy_id"] == policy_id for p in policies)
    
    # Delete access policy
    delete_result = delete_access_policy(
        policy_id=policy_id,
        db_session=db_session
    )
    
    # Verify deletion result
    assert isinstance(delete_result, Dict)
    assert delete_result["policy_id"] == policy_id
    assert delete_result["status"] == "deleted"

def test_verify_access(db_session, test_user, test_access_policy):
    """Test access verification"""
    # First, create access policy
    policy_result = create_access_policy(
        policy_config=test_access_policy,
        db_session=db_session
    )
    
    # Test access verification with valid token
    auth_result = authenticate_user(
        username="testuser",
        password="test_password123",
        db_session=db_session
    )
    
    token = auth_result["token"]
    
    # Verify access
    result = verify_access(
        token=token,
        resource_type="data_source",
        resource_id="test_source",
        action="read",
        db_session=db_session
    )
    
    # Verify result
    assert isinstance(result, Dict)
    assert "verified" in result
    assert result["verified"] is True
    assert "user_id" in result
    assert result["user_id"] == test_user.user_id
    
    # Verify audit log
    audit_log = db_session.query(SecurityAuditLog).filter_by(
        user_id=test_user.user_id,
        event_type="access_verification",
        status="success"
    ).first()
    assert audit_log is not None
    
    # Test access verification with invalid token
    with pytest.raises(SecurityError) as excinfo:
        verify_access(
            token="invalid_token",
            resource_type="data_source",
            resource_id="test_source",
            action="read",
            db_session=db_session
        )
    assert "Invalid token" in str(excinfo.value)
    
    # Verify failed verification audit log
    failed_audit = db_session.query(SecurityAuditLog).filter_by(
        user_id=None,
        event_type="access_verification",
        status="failure"
    ).first()
    assert failed_audit is not None

def test_audit_logging(db_session, test_user):
    """Test security audit logging"""
    # Log security event
    event_data = {
        "event_type": "policy_change",
        "user_id": test_user.user_id,
        "resource_type": "access_policy",
        "resource_id": "test_policy",
        "action": "update",
        "status": "success",
        "details": {
            "changes": ["added_permission", "modified_condition"],
            "reason": "policy_update"
        }
    }
    
    result = log_security_event(
        event_data=event_data,
        db_session=db_session
    )
    
    # Verify logging result
    assert isinstance(result, Dict)
    assert "event_id" in result
    assert "timestamp" in result
    assert result["event_type"] == "policy_change"
    assert result["user_id"] == test_user.user_id
    
    # Get audit logs
    logs = get_audit_logs(
        user_id=test_user.user_id,
        start_time=datetime.utcnow() - timedelta(hours=1),
        end_time=datetime.utcnow(),
        event_types=["policy_change"],
        db_session=db_session
    )
    
    # Verify audit logs
    assert isinstance(logs, List)
    assert len(logs) > 0
    assert any(log["event_id"] == result["event_id"] for log in logs)
    
    # Verify log details
    log = next(log for log in logs if log["event_id"] == result["event_id"])
    assert log["event_type"] == "policy_change"
    assert log["status"] == "success"
    assert "changes" in log["details"]
    assert "reason" in log["details"]

def test_security_error_handling(db_session, test_user):
    """Test security error handling"""
    # Invalid authentication
    with pytest.raises(SecurityError) as excinfo:
        authenticate_user(
            username="nonexistent",
            password="wrong_password",
            db_session=db_session
        )
    assert "Invalid credentials" in str(excinfo.value)
    
    # Invalid authorization
    with pytest.raises(SecurityError) as excinfo:
        authorize_access(
            user_id=None,
            resource_type="data_source",
            resource_id="test_source",
            action="read",
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid encryption
    with pytest.raises(SecurityError) as excinfo:
        encrypt_data(
            data={},
            encryption_key="invalid_key",
            data_type="pii",
            db_session=db_session
        )
    assert "Invalid encryption key" in str(excinfo.value)
    
    # Invalid access policy
    with pytest.raises(SecurityError) as excinfo:
        create_access_policy(
            policy_config={},
            db_session=db_session
        )
    assert "Invalid policy configuration" in str(excinfo.value)
    
    # Invalid audit event
    with pytest.raises(SecurityError) as excinfo:
        log_security_event(
            event_data={},
            db_session=db_session
        )
    assert "Invalid event data" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBSecurityError) as excinfo:
        authenticate_user(
            username="testuser",
            password="test_password123",
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 