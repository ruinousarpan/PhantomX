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
import jsonschema
import cerberus
import voluptuous
import marshmallow
import pydantic
import io
import base64
import hashlib
import uuid

from src.validation import (
    validate_user_input,
    validate_mining_data,
    validate_staking_data,
    validate_trading_data,
    validate_performance_data,
    validate_risk_data,
    validate_reward_data,
    validate_activity_data,
    sanitize_input,
    validate_schema,
    ValidationError,
    ValidationLevel,
    validate_data,
    check_data_quality,
    verify_schema,
    get_validation_info
)
from database.models import User, ValidationRule, ValidationResult, ValidationRecord
from database.exceptions import ValidationError as DBValidationError, DatabaseError, ConnectionError  # etc.

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
def test_mining_data():
    """Create test mining data"""
    return {
        "hash_rate": 95.5,
        "power_usage": 1450.0,
        "temperature": 75.0,
        "uptime": 0.98,
        "efficiency": 0.85,
        "block_rewards": 0.5,
        "network_difficulty": 45.0,
        "profitability": 0.25
    }

@pytest.fixture
def test_staking_data():
    """Create test staking data"""
    return {
        "validator_uptime": 0.99,
        "missed_blocks": 2,
        "reward_rate": 0.12,
        "peer_count": 50,
        "network_participation": 0.85,
        "slashing_events": 0,
        "stake_amount": 1000.0,
        "validator_count": 100
    }

@pytest.fixture
def test_trading_data():
    """Create test trading data"""
    return {
        "position_size": 50000.0,
        "leverage_ratio": 2.0,
        "win_rate": 0.6,
        "profit_loss": 500.0,
        "drawdown": 0.15,
        "sharpe_ratio": 1.5,
        "volume": 100000.0,
        "execution_quality": 0.9
    }

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
def test_schema():
    """Create test schema"""
    return {
        "type": "object",
        "properties": {
            "timestamp": {"type": "string", "format": "date-time"},
            "mining_performance": {"type": "number", "minimum": 0, "maximum": 1},
            "staking_performance": {"type": "number", "minimum": 0, "maximum": 1},
            "trading_performance": {"type": "number", "minimum": 0, "maximum": 1},
            "overall_performance": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["timestamp", "mining_performance", "staking_performance", "trading_performance", "overall_performance"]
    }

@pytest.fixture
def test_quality_rules():
    """Create test quality rules"""
    return {
        "completeness": {
            "min_percentage": 0.95,
            "required_columns": ["timestamp", "mining_performance", "staking_performance", "trading_performance", "overall_performance"]
        },
        "accuracy": {
            "min_percentage": 0.9,
            "tolerance": 0.1
        },
        "consistency": {
            "min_percentage": 0.95,
            "check_duplicates": True,
            "check_anomalies": True
        }
    }

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
def test_validation_config():
    """Create test validation configuration"""
    return {
        "validation_type": "range",
        "columns": None,  # Validate all columns
        "exclude_columns": ["timestamp", "day"],
        "rules": {
            "mining_performance": {"min": 0.7, "max": 1.0},
            "staking_performance": {"min": 0.8, "max": 1.0},
            "trading_performance": {"min": 0.6, "max": 0.9},
            "overall_performance": {"min": 0.7, "max": 1.0}
        }
    }

def test_validate_user_input(db_session, test_user):
    """Test user input validation"""
    # Valid user input
    valid_input = {
        "username": "newuser",
        "email": "newuser@example.com",
        "password": "SecurePassword123!",
        "first_name": "New",
        "last_name": "User"
    }
    
    # Validate user input
    result = validate_user_input(
        input_data=valid_input,
        validation_level=ValidationLevel.STRICT,
        db_session=db_session
    )
    
    # Verify validation result
    assert isinstance(result, Dict)
    assert "is_valid" in result
    assert "errors" in result
    assert "warnings" in result
    
    # Verify validation details
    assert result["is_valid"] is True
    assert len(result["errors"]) == 0
    assert len(result["warnings"]) == 0
    
    # Invalid user input
    invalid_input = {
        "username": "a",  # Too short
        "email": "invalid_email",  # Invalid email format
        "password": "weak",  # Too weak
        "first_name": "",  # Empty
        "last_name": "User"
    }
    
    # Validate invalid user input
    result = validate_user_input(
        input_data=invalid_input,
        validation_level=ValidationLevel.STRICT,
        db_session=db_session
    )
    
    # Verify validation result
    assert result["is_valid"] is False
    assert len(result["errors"]) > 0
    assert "username" in result["errors"]
    assert "email" in result["errors"]
    assert "password" in result["errors"]
    assert "first_name" in result["errors"]
    
    # Verify database entry
    db_result = db_session.query(ValidationResult).filter_by(
        user_id=test_user.user_id,
        validation_type="USER_INPUT"
    ).first()
    assert db_result is not None
    assert db_result.is_valid == result["is_valid"]
    assert db_result.errors == result["errors"]
    assert db_result.warnings == result["warnings"]

def test_validate_mining_data(db_session, test_user, test_mining_data):
    """Test mining data validation"""
    # Valid mining data
    result = validate_mining_data(
        user_id=test_user.user_id,
        data=test_mining_data,
        validation_level=ValidationLevel.STRICT,
        db_session=db_session
    )
    
    # Verify validation result
    assert isinstance(result, Dict)
    assert "is_valid" in result
    assert "errors" in result
    assert "warnings" in result
    
    # Verify validation details
    assert result["is_valid"] is True
    assert len(result["errors"]) == 0
    assert len(result["warnings"]) == 0
    
    # Invalid mining data
    invalid_data = {
        "hash_rate": -1.0,  # Negative value
        "power_usage": 0.0,  # Zero value
        "temperature": 150.0,  # Too high
        "uptime": 1.5,  # Greater than 1
        "efficiency": 0.0,  # Zero value
        "block_rewards": -0.5,  # Negative value
        "network_difficulty": 0.0,  # Zero value
        "profitability": 2.0  # Greater than 1
    }
    
    # Validate invalid mining data
    result = validate_mining_data(
        user_id=test_user.user_id,
        data=invalid_data,
        validation_level=ValidationLevel.STRICT,
        db_session=db_session
    )
    
    # Verify validation result
    assert result["is_valid"] is False
    assert len(result["errors"]) > 0
    assert "hash_rate" in result["errors"]
    assert "power_usage" in result["errors"]
    assert "temperature" in result["errors"]
    assert "uptime" in result["errors"]
    assert "efficiency" in result["errors"]
    assert "block_rewards" in result["errors"]
    assert "network_difficulty" in result["errors"]
    assert "profitability" in result["errors"]
    
    # Verify database entry
    db_result = db_session.query(ValidationResult).filter_by(
        user_id=test_user.user_id,
        validation_type="MINING_DATA"
    ).first()
    assert db_result is not None
    assert db_result.is_valid == result["is_valid"]
    assert db_result.errors == result["errors"]
    assert db_result.warnings == result["warnings"]

def test_validate_staking_data(db_session, test_user, test_staking_data):
    """Test staking data validation"""
    # Valid staking data
    result = validate_staking_data(
        user_id=test_user.user_id,
        data=test_staking_data,
        validation_level=ValidationLevel.STRICT,
        db_session=db_session
    )
    
    # Verify validation result
    assert isinstance(result, Dict)
    assert "is_valid" in result
    assert "errors" in result
    assert "warnings" in result
    
    # Verify validation details
    assert result["is_valid"] is True
    assert len(result["errors"]) == 0
    assert len(result["warnings"]) == 0
    
    # Invalid staking data
    invalid_data = {
        "validator_uptime": 1.5,  # Greater than 1
        "missed_blocks": -1,  # Negative value
        "reward_rate": -0.12,  # Negative value
        "peer_count": 0,  # Zero value
        "network_participation": 2.0,  # Greater than 1
        "slashing_events": -1,  # Negative value
        "stake_amount": 0.0,  # Zero value
        "validator_count": 0  # Zero value
    }
    
    # Validate invalid staking data
    result = validate_staking_data(
        user_id=test_user.user_id,
        data=invalid_data,
        validation_level=ValidationLevel.STRICT,
        db_session=db_session
    )
    
    # Verify validation result
    assert result["is_valid"] is False
    assert len(result["errors"]) > 0
    assert "validator_uptime" in result["errors"]
    assert "missed_blocks" in result["errors"]
    assert "reward_rate" in result["errors"]
    assert "peer_count" in result["errors"]
    assert "network_participation" in result["errors"]
    assert "slashing_events" in result["errors"]
    assert "stake_amount" in result["errors"]
    assert "validator_count" in result["errors"]
    
    # Verify database entry
    db_result = db_session.query(ValidationResult).filter_by(
        user_id=test_user.user_id,
        validation_type="STAKING_DATA"
    ).first()
    assert db_result is not None
    assert db_result.is_valid == result["is_valid"]
    assert db_result.errors == result["errors"]
    assert db_result.warnings == result["warnings"]

def test_validate_trading_data(db_session, test_user, test_trading_data):
    """Test trading data validation"""
    # Valid trading data
    result = validate_trading_data(
        user_id=test_user.user_id,
        data=test_trading_data,
        validation_level=ValidationLevel.STRICT,
        db_session=db_session
    )
    
    # Verify validation result
    assert isinstance(result, Dict)
    assert "is_valid" in result
    assert "errors" in result
    assert "warnings" in result
    
    # Verify validation details
    assert result["is_valid"] is True
    assert len(result["errors"]) == 0
    assert len(result["warnings"]) == 0
    
    # Invalid trading data
    invalid_data = {
        "position_size": -50000.0,  # Negative value
        "leverage_ratio": 0.0,  # Zero value
        "win_rate": 2.0,  # Greater than 1
        "profit_loss": 0.0,  # Zero value
        "drawdown": 2.0,  # Greater than 1
        "sharpe_ratio": 0.0,  # Zero value
        "volume": -100000.0,  # Negative value
        "execution_quality": 2.0  # Greater than 1
    }
    
    # Validate invalid trading data
    result = validate_trading_data(
        user_id=test_user.user_id,
        data=invalid_data,
        validation_level=ValidationLevel.STRICT,
        db_session=db_session
    )
    
    # Verify validation result
    assert result["is_valid"] is False
    assert len(result["errors"]) > 0
    assert "position_size" in result["errors"]
    assert "leverage_ratio" in result["errors"]
    assert "win_rate" in result["errors"]
    assert "profit_loss" in result["errors"]
    assert "drawdown" in result["errors"]
    assert "sharpe_ratio" in result["errors"]
    assert "volume" in result["errors"]
    assert "execution_quality" in result["errors"]
    
    # Verify database entry
    db_result = db_session.query(ValidationResult).filter_by(
        user_id=test_user.user_id,
        validation_type="TRADING_DATA"
    ).first()
    assert db_result is not None
    assert db_result.is_valid == result["is_valid"]
    assert db_result.errors == result["errors"]
    assert db_result.warnings == result["warnings"]

def test_validate_data(db_session, test_user, test_performance_data, test_schema):
    """Test data validation"""
    # Validate data
    result = validate_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        schema=test_schema,
        validation_type="schema",
        db_session=db_session
    )
    
    # Verify validation result
    assert isinstance(result, Dict)
    assert "is_valid" in result
    assert "validation_type" in result
    
    # Verify validation metadata
    assert result["validation_type"] == "schema"
    assert result["is_valid"] is True
    
    # Verify validation details
    assert "validation_details" in result
    assert isinstance(result["validation_details"], Dict)
    
    # Verify database entry
    db_result = db_session.query(ValidationResult).filter_by(
        user_id=test_user.user_id,
        validation_type="SCHEMA_VALIDATION"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_validate_schema(db_session, test_user, test_performance_data, test_schema):
    """Test schema validation"""
    # Validate schema
    result = validate_schema(
        user_id=test_user.user_id,
        data=test_performance_data,
        schema=test_schema,
        db_session=db_session
    )
    
    # Verify validation result
    assert isinstance(result, Dict)
    assert "is_valid" in result
    assert "validation_type" in result
    
    # Verify validation metadata
    assert result["validation_type"] == "schema"
    assert result["is_valid"] is True
    
    # Verify validation details
    assert "validation_details" in result
    assert isinstance(result["validation_details"], Dict)
    
    # Verify database entry
    db_result = db_session.query(ValidationResult).filter_by(
        user_id=test_user.user_id,
        validation_type="SCHEMA_VALIDATION"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_check_data_quality(db_session, test_user, test_performance_data, test_quality_rules):
    """Test data quality check"""
    # Check data quality
    result = check_data_quality(
        user_id=test_user.user_id,
        data=test_performance_data,
        quality_rules=test_quality_rules,
        db_session=db_session
    )
    
    # Verify quality check result
    assert isinstance(result, Dict)
    assert "quality_score" in result
    assert "quality_rules" in result
    
    # Verify quality metadata
    assert result["quality_rules"] == test_quality_rules
    assert isinstance(result["quality_score"], float)
    assert 0 <= result["quality_score"] <= 1
    
    # Verify quality details
    assert "quality_details" in result
    assert isinstance(result["quality_details"], Dict)
    assert "completeness" in result["quality_details"]
    assert "accuracy" in result["quality_details"]
    assert "consistency" in result["quality_details"]
    
    # Verify database entry
    db_result = db_session.query(ValidationResult).filter_by(
        user_id=test_user.user_id,
        validation_type="QUALITY_CHECK"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_validate_performance_data(db_session, test_user, test_performance_data, test_validation_config):
    """Test validating performance data"""
    # Validate performance data
    result = validate_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        validation_config=test_validation_config,
        db_session=db_session
    )
    
    # Verify validation result
    assert isinstance(result, Dict)
    assert "validation_id" in result
    assert "is_valid" in result
    assert "validation_details" in result
    
    # Verify validation metadata
    assert result["validation_type"] == "range"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["rules"] == {
        "mining_performance": {"min": 0.7, "max": 1.0},
        "staking_performance": {"min": 0.8, "max": 1.0},
        "trading_performance": {"min": 0.6, "max": 0.9},
        "overall_performance": {"min": 0.7, "max": 1.0}
    }
    
    # Verify validation details
    assert isinstance(result["validation_details"], Dict)
    assert "timestamp" in result["validation_details"]
    assert "validation_results" in result["validation_details"]
    assert "invalid_rows" in result["validation_details"]
    assert "invalid_columns" in result["validation_details"]
    
    # Verify validation results
    assert isinstance(result["validation_details"]["validation_results"], Dict)
    assert "mining_performance" in result["validation_details"]["validation_results"]
    assert "staking_performance" in result["validation_details"]["validation_results"]
    assert "trading_performance" in result["validation_details"]["validation_results"]
    assert "overall_performance" in result["validation_details"]["validation_results"]
    
    # Verify database entry
    db_record = db_session.query(ValidationRecord).filter_by(
        user_id=test_user.user_id,
        validation_id=result["validation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_validate_risk_data(db_session, test_user, test_risk_data):
    """Test validating risk data"""
    # Create validation config for risk data
    risk_config = {
        "validation_type": "range",
        "columns": None,  # Validate all columns
        "exclude_columns": ["timestamp", "day"],
        "rules": {
            "mining_risk": {"min": 0.0, "max": 0.5},
            "staking_risk": {"min": 0.0, "max": 0.3},
            "trading_risk": {"min": 0.0, "max": 0.6},
            "overall_risk": {"min": 0.0, "max": 0.5}
        }
    }
    
    # Validate risk data
    result = validate_data(
        user_id=test_user.user_id,
        data=test_risk_data,
        validation_config=risk_config,
        db_session=db_session
    )
    
    # Verify validation result
    assert isinstance(result, Dict)
    assert "validation_id" in result
    assert "is_valid" in result
    assert "validation_details" in result
    
    # Verify validation metadata
    assert result["validation_type"] == "range"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["rules"] == {
        "mining_risk": {"min": 0.0, "max": 0.5},
        "staking_risk": {"min": 0.0, "max": 0.3},
        "trading_risk": {"min": 0.0, "max": 0.6},
        "overall_risk": {"min": 0.0, "max": 0.5}
    }
    
    # Verify validation details
    assert isinstance(result["validation_details"], Dict)
    assert "timestamp" in result["validation_details"]
    assert "validation_results" in result["validation_details"]
    assert "invalid_rows" in result["validation_details"]
    assert "invalid_columns" in result["validation_details"]
    
    # Verify validation results
    assert isinstance(result["validation_details"]["validation_results"], Dict)
    assert "mining_risk" in result["validation_details"]["validation_results"]
    assert "staking_risk" in result["validation_details"]["validation_results"]
    assert "trading_risk" in result["validation_details"]["validation_results"]
    assert "overall_risk" in result["validation_details"]["validation_results"]
    
    # Verify database entry
    db_record = db_session.query(ValidationRecord).filter_by(
        user_id=test_user.user_id,
        validation_id=result["validation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_validate_reward_data(db_session, test_user, test_reward_data):
    """Test validating reward data"""
    # Create validation config for reward data
    reward_config = {
        "validation_type": "range",
        "columns": None,  # Validate all columns
        "exclude_columns": ["timestamp", "day"],
        "rules": {
            "mining_reward": {"min": 0.0, "max": 2.0},
            "staking_reward": {"min": 0.0, "max": 2.0},
            "trading_reward": {"min": 0.0, "max": 2.0},
            "overall_reward": {"min": 0.0, "max": 2.0}
        }
    }
    
    # Validate reward data
    result = validate_data(
        user_id=test_user.user_id,
        data=test_reward_data,
        validation_config=reward_config,
        db_session=db_session
    )
    
    # Verify validation result
    assert isinstance(result, Dict)
    assert "validation_id" in result
    assert "is_valid" in result
    assert "validation_details" in result
    
    # Verify validation metadata
    assert result["validation_type"] == "range"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["rules"] == {
        "mining_reward": {"min": 0.0, "max": 2.0},
        "staking_reward": {"min": 0.0, "max": 2.0},
        "trading_reward": {"min": 0.0, "max": 2.0},
        "overall_reward": {"min": 0.0, "max": 2.0}
    }
    
    # Verify validation details
    assert isinstance(result["validation_details"], Dict)
    assert "timestamp" in result["validation_details"]
    assert "validation_results" in result["validation_details"]
    assert "invalid_rows" in result["validation_details"]
    assert "invalid_columns" in result["validation_details"]
    
    # Verify validation results
    assert isinstance(result["validation_details"]["validation_results"], Dict)
    assert "mining_reward" in result["validation_details"]["validation_results"]
    assert "staking_reward" in result["validation_details"]["validation_results"]
    assert "trading_reward" in result["validation_details"]["validation_results"]
    assert "overall_reward" in result["validation_details"]["validation_results"]
    
    # Verify database entry
    db_record = db_session.query(ValidationRecord).filter_by(
        user_id=test_user.user_id,
        validation_id=result["validation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_validate_activity_data(db_session, test_user, test_activity_data):
    """Test validating activity data"""
    # Create validation config for activity data
    activity_config = {
        "validation_type": "range",
        "columns": None,  # Validate all columns
        "exclude_columns": ["timestamp", "day"],
        "rules": {
            "mining_activity": {"min": 0.0, "max": 1.0},
            "staking_activity": {"min": 0.0, "max": 1.0},
            "trading_activity": {"min": 0.0, "max": 1.0},
            "overall_activity": {"min": 0.0, "max": 1.0}
        }
    }
    
    # Validate activity data
    result = validate_data(
        user_id=test_user.user_id,
        data=test_activity_data,
        validation_config=activity_config,
        db_session=db_session
    )
    
    # Verify validation result
    assert isinstance(result, Dict)
    assert "validation_id" in result
    assert "is_valid" in result
    assert "validation_details" in result
    
    # Verify validation metadata
    assert result["validation_type"] == "range"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["rules"] == {
        "mining_activity": {"min": 0.0, "max": 1.0},
        "staking_activity": {"min": 0.0, "max": 1.0},
        "trading_activity": {"min": 0.0, "max": 1.0},
        "overall_activity": {"min": 0.0, "max": 1.0}
    }
    
    # Verify validation details
    assert isinstance(result["validation_details"], Dict)
    assert "timestamp" in result["validation_details"]
    assert "validation_results" in result["validation_details"]
    assert "invalid_rows" in result["validation_details"]
    assert "invalid_columns" in result["validation_details"]
    
    # Verify validation results
    assert isinstance(result["validation_details"]["validation_results"], Dict)
    assert "mining_activity" in result["validation_details"]["validation_results"]
    assert "staking_activity" in result["validation_details"]["validation_results"]
    assert "trading_activity" in result["validation_details"]["validation_results"]
    assert "overall_activity" in result["validation_details"]["validation_results"]
    
    # Verify database entry
    db_record = db_session.query(ValidationRecord).filter_by(
        user_id=test_user.user_id,
        validation_id=result["validation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_validate_reward_data_direct(db_session, test_user, test_reward_data):
    """Test reward data validation directly"""
    result = validate_reward_data(
        user_id=test_user.user_id,
        data=test_reward_data,
        validation_level=ValidationLevel.STRICT,
        db_session=db_session
    )

    # Verify basic result structure
    assert isinstance(result, Dict)
    assert "is_valid" in result
    assert "errors" in result
    assert "warnings" in result
    
    # Verify validation success
    assert result["is_valid"] is True
    assert len(result["errors"]) == 0
    assert len(result["warnings"]) == 0

    # Verify data validation
    assert all(0.0 <= value <= 2.0 for value in test_reward_data["mining_reward"])
    assert all(0.0 <= value <= 2.0 for value in test_reward_data["staking_reward"])
    assert all(0.0 <= value <= 2.0 for value in test_reward_data["trading_reward"])
    assert all(0.0 <= value <= 2.0 for value in test_reward_data["overall_reward"])

    # Verify database entry
    db_result = db_session.query(ValidationResult).filter_by(
        user_id=test_user.user_id,
        validation_type="REWARD_DATA"
    ).first()
    assert db_result is not None
    assert db_result.is_valid is True
    assert db_result.errors == {}
    assert db_result.warnings == {}

def test_sanitize_input(db_session, test_user):
    """Test input sanitization"""
    # Input with potential security issues
    unsafe_input = {
        "username": "user<script>alert('xss')</script>",
        "email": "user@example.com<script>alert('xss')</script>",
        "comment": "<p>Hello</p><script>alert('xss')</script>",
        "query": "SELECT * FROM users; DROP TABLE users;"
    }
    
    # Sanitize input
    sanitized_input = sanitize_input(
        input_data=unsafe_input,
        db_session=db_session
    )
    
    # Verify sanitized input
    assert isinstance(sanitized_input, Dict)
    assert "username" in sanitized_input
    assert "email" in sanitized_input
    assert "comment" in sanitized_input
    assert "query" in sanitized_input
    
    # Verify sanitization
    assert "<script>" not in sanitized_input["username"]
    assert "<script>" not in sanitized_input["email"]
    assert "<script>" not in sanitized_input["comment"]
    assert "DROP TABLE" not in sanitized_input["query"]
    
    # Verify database entry
    db_rule = db_session.query(ValidationRule).filter_by(
        rule_type="SANITIZATION"
    ).first()
    assert db_rule is not None
    assert db_rule.is_active is True

def test_validation_error_handling(db_session, test_user):
    """Test validation error handling"""
    # Invalid user ID
    with pytest.raises(ValidationError) as excinfo:
        validate_data(
            user_id=None,
            data=pd.DataFrame(),
            validation_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(ValidationError) as excinfo:
        validate_data(
            user_id=test_user.user_id,
            data=None,
            validation_config={},
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Invalid validation type
    with pytest.raises(ValidationError) as excinfo:
        validate_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            validation_config={"validation_type": "invalid_type"},
            db_session=db_session
        )
    assert "Invalid validation type" in str(excinfo.value)
    
    # Invalid columns
    with pytest.raises(ValidationError) as excinfo:
        validate_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            validation_config={"validation_type": "range", "columns": ["invalid_column"]},
            db_session=db_session
        )
    assert "Invalid columns" in str(excinfo.value)
    
    # Invalid validation ID
    with pytest.raises(ValidationError) as excinfo:
        get_validation_info(
            user_id=test_user.user_id,
            validation_id="invalid_validation_id",
            db_session=db_session
        )
    assert "Invalid validation ID" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBValidationError) as excinfo:
        validate_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            validation_config={"validation_type": "range"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value)

def test_validate_analytics_data(db_session, test_user, test_analytics_data, test_validation_config):
    """Test analytics data validation"""
    # Validate analytics data
    result = validate_data(
        user_id=test_user.user_id,
        data=test_analytics_data,
        validation_config=test_validation_config,
        db_session=db_session
    )
    
    # Verify validation result
    assert isinstance(result, Dict)
    assert "validation_id" in result
    assert "is_valid" in result
    assert result["is_valid"] is True
    
    # Verify validation metadata
    assert "schema" in result
    assert result["schema"] == test_validation_config["schema"]
    
    # Verify validation details
    assert "validation_details" in result
    assert isinstance(result["validation_details"], Dict)
    assert "timestamp" in result["validation_details"]
    assert "errors" in result["validation_details"]
    assert isinstance(result["validation_details"]["errors"], List)
    assert len(result["validation_details"]["errors"]) == 0
    
    # Verify database entry
    db_record = db_session.query(ValidationRecord).filter_by(
        user_id=test_user.user_id,
        validation_id=result["validation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_verify_schema(db_session, test_user, test_validation_config):
    """Test schema verification"""
    # Verify schema
    result = verify_schema(
        user_id=test_user.user_id,
        schema=test_validation_config["schema"],
        db_session=db_session
    )
    
    # Verify schema verification result
    assert isinstance(result, Dict)
    assert "schema_id" in result
    assert "is_valid" in result
    assert result["is_valid"] is True
    
    # Verify schema metadata
    assert "schema" in result
    assert result["schema"] == test_validation_config["schema"]
    
    # Verify schema details
    assert "schema_details" in result
    assert isinstance(result["schema_details"], Dict)
    assert "timestamp" in result["schema_details"]
    assert "errors" in result["schema_details"]
    assert isinstance(result["schema_details"]["errors"], List)
    assert len(result["schema_details"]["errors"]) == 0
    
    # Verify database entry
    db_record = db_session.query(ValidationRecord).filter_by(
        user_id=test_user.user_id,
        validation_id=result["schema_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_get_validation_info(db_session, test_user, test_performance_data, test_validation_config):
    """Test validation info retrieval"""
    # First, validate performance data
    validation_result = validate_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        validation_config=test_validation_config,
        db_session=db_session
    )
    
    validation_id = validation_result["validation_id"]
    
    # Get validation info
    result = get_validation_info(
        user_id=test_user.user_id,
        validation_id=validation_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "validation_id" in result
    assert result["validation_id"] == validation_id
    
    # Verify validation metadata
    assert result["validation_type"] == "range"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["rules"] == {
        "mining_performance": {"min": 0.7, "max": 1.0},
        "staking_performance": {"min": 0.8, "max": 1.0},
        "trading_performance": {"min": 0.6, "max": 0.9},
        "overall_performance": {"min": 0.7, "max": 1.0}
    }
    
    # Verify validation details
    assert "validation_details" in result
    assert isinstance(result["validation_details"], Dict)
    assert "timestamp" in result["validation_details"]
    assert "validation_results" in result["validation_details"]
    assert "invalid_rows" in result["validation_details"]
    assert "invalid_columns" in result["validation_details"]
    
    # Verify database entry
    db_record = db_session.query(ValidationRecord).filter_by(
        user_id=test_user.user_id,
        validation_id=validation_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_validate_with_regex(db_session, test_user):
    """Test validating with regex"""
    # Create data with text fields
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
    days = dates.date
    
    text_data = pd.DataFrame({
        "timestamp": dates,
        "day": days,
        "email": [f"user{i}@example.com" for i in range(100)],
        "phone": [f"+1-555-{i:04d}" for i in range(100)],
        "username": [f"user{i}" for i in range(100)],
        "password": [f"pass{i}" for i in range(100)]
    })
    
    # Create validation config with regex
    regex_config = {
        "validation_type": "regex",
        "columns": None,  # Validate all columns
        "exclude_columns": ["timestamp", "day"],
        "rules": {
            "email": {"pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"},
            "phone": {"pattern": r"^\+1-\d{3}-\d{4}$"},
            "username": {"pattern": r"^[a-zA-Z0-9_]{3,20}$"},
            "password": {"pattern": r"^[a-zA-Z0-9_]{8,}$"}
        }
    }
    
    # Validate text data
    result = validate_data(
        user_id=test_user.user_id,
        data=text_data,
        validation_config=regex_config,
        db_session=db_session
    )
    
    # Verify validation result
    assert isinstance(result, Dict)
    assert "validation_id" in result
    assert "is_valid" in result
    assert "validation_details" in result
    
    # Verify validation metadata
    assert result["validation_type"] == "regex"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["rules"] == {
        "email": {"pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"},
        "phone": {"pattern": r"^\+1-\d{3}-\d{4}$"},
        "username": {"pattern": r"^[a-zA-Z0-9_]{3,20}$"},
        "password": {"pattern": r"^[a-zA-Z0-9_]{8,}$"}
    }
    
    # Verify validation details
    assert isinstance(result["validation_details"], Dict)
    assert "timestamp" in result["validation_details"]
    assert "validation_results" in result["validation_details"]
    assert "invalid_rows" in result["validation_details"]
    assert "invalid_columns" in result["validation_details"]
    
    # Verify validation results
    assert isinstance(result["validation_details"]["validation_results"], Dict)
    assert "email" in result["validation_details"]["validation_results"]
    assert "phone" in result["validation_details"]["validation_results"]
    assert "username" in result["validation_details"]["validation_results"]
    assert "password" in result["validation_details"]["validation_results"]
    
    # Verify database entry
    db_record = db_session.query(ValidationRecord).filter_by(
        user_id=test_user.user_id,
        validation_id=result["validation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_validate_with_custom_function(db_session, test_user, test_performance_data):
    """Test validating with custom function"""
    # Create validation config with custom function
    custom_config = {
        "validation_type": "custom",
        "columns": None,  # Validate all columns
        "exclude_columns": ["timestamp", "day"],
        "rules": {
            "mining_performance": {"function": "lambda x: x >= 0.7 and x <= 1.0"},
            "staking_performance": {"function": "lambda x: x >= 0.8 and x <= 1.0"},
            "trading_performance": {"function": "lambda x: x >= 0.6 and x <= 0.9"},
            "overall_performance": {"function": "lambda x: x >= 0.7 and x <= 1.0"}
        }
    }
    
    # Validate performance data
    result = validate_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        validation_config=custom_config,
        db_session=db_session
    )
    
    # Verify validation result
    assert isinstance(result, Dict)
    assert "validation_id" in result
    assert "is_valid" in result
    assert "validation_details" in result
    
    # Verify validation metadata
    assert result["validation_type"] == "custom"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["rules"] == {
        "mining_performance": {"function": "lambda x: x >= 0.7 and x <= 1.0"},
        "staking_performance": {"function": "lambda x: x >= 0.8 and x <= 1.0"},
        "trading_performance": {"function": "lambda x: x >= 0.6 and x <= 0.9"},
        "overall_performance": {"function": "lambda x: x >= 0.7 and x <= 1.0"}
    }
    
    # Verify validation details
    assert isinstance(result["validation_details"], Dict)
    assert "timestamp" in result["validation_details"]
    assert "validation_results" in result["validation_details"]
    assert "invalid_rows" in result["validation_details"]
    assert "invalid_columns" in result["validation_details"]
    
    # Verify validation results
    assert isinstance(result["validation_details"]["validation_results"], Dict)
    assert "mining_performance" in result["validation_details"]["validation_results"]
    assert "staking_performance" in result["validation_details"]["validation_results"]
    assert "trading_performance" in result["validation_details"]["validation_results"]
    assert "overall_performance" in result["validation_details"]["validation_results"]
    
    # Verify database entry
    db_record = db_session.query(ValidationRecord).filter_by(
        user_id=test_user.user_id,
        validation_id=result["validation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_validate_with_specific_columns(db_session, test_user, test_performance_data):
    """Test validating with specific columns"""
    # Create validation config with specific columns
    specific_columns_config = {
        "validation_type": "range",
        "columns": ["mining_performance", "trading_performance"],  # Only validate these columns
        "exclude_columns": ["timestamp", "day"],
        "rules": {
            "mining_performance": {"min": 0.7, "max": 1.0},
            "trading_performance": {"min": 0.6, "max": 0.9}
        }
    }
    
    # Validate performance data
    result = validate_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        validation_config=specific_columns_config,
        db_session=db_session
    )
    
    # Verify validation result
    assert isinstance(result, Dict)
    assert "validation_id" in result
    assert "is_valid" in result
    assert "validation_details" in result
    
    # Verify validation metadata
    assert result["validation_type"] == "range"
    assert result["columns"] == ["mining_performance", "trading_performance"]
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["rules"] == {
        "mining_performance": {"min": 0.7, "max": 1.0},
        "trading_performance": {"min": 0.6, "max": 0.9}
    }
    
    # Verify validation details
    assert isinstance(result["validation_details"], Dict)
    assert "timestamp" in result["validation_details"]
    assert "validation_results" in result["validation_details"]
    assert "invalid_rows" in result["validation_details"]
    assert "invalid_columns" in result["validation_details"]
    
    # Verify validation results
    assert isinstance(result["validation_details"]["validation_results"], Dict)
    assert "mining_performance" in result["validation_details"]["validation_results"]
    assert "trading_performance" in result["validation_details"]["validation_results"]
    assert "staking_performance" not in result["validation_details"]["validation_results"]
    assert "overall_performance" not in result["validation_details"]["validation_results"]
    
    # Verify database entry
    db_record = db_session.query(ValidationRecord).filter_by(
        user_id=test_user.user_id,
        validation_id=result["validation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_validate_with_invalid_data(db_session, test_user):
    """Test validating with invalid data"""
    # Create data with invalid values
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
    days = dates.date
    
    # Create data with some invalid values
    invalid_data = pd.DataFrame({
        "timestamp": dates,
        "day": days,
        "mining_performance": np.random.uniform(0.8, 0.9, 100),
        "staking_performance": np.random.uniform(0.85, 0.95, 100),
        "trading_performance": np.random.uniform(0.7, 0.8, 100),
        "overall_performance": np.random.uniform(0.8, 0.9, 100)
    })
    
    # Add some invalid values
    invalid_data.loc[0, "mining_performance"] = 1.5  # Above max
    invalid_data.loc[1, "mining_performance"] = 0.5  # Below min
    invalid_data.loc[2, "trading_performance"] = 1.0  # Above max
    invalid_data.loc[3, "trading_performance"] = 0.5  # Below min
    
    # Create validation config
    validation_config = {
        "validation_type": "range",
        "columns": None,  # Validate all columns
        "exclude_columns": ["timestamp", "day"],
        "rules": {
            "mining_performance": {"min": 0.7, "max": 1.0},
            "staking_performance": {"min": 0.8, "max": 1.0},
            "trading_performance": {"min": 0.6, "max": 0.9},
            "overall_performance": {"min": 0.7, "max": 1.0}
        }
    }
    
    # Validate invalid data
    result = validate_data(
        user_id=test_user.user_id,
        data=invalid_data,
        validation_config=validation_config,
        db_session=db_session
    )
    
    # Verify validation result
    assert isinstance(result, Dict)
    assert "validation_id" in result
    assert "is_valid" in result
    assert result["is_valid"] is False  # Data should be invalid
    assert "validation_details" in result
    
    # Verify validation metadata
    assert result["validation_type"] == "range"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["rules"] == {
        "mining_performance": {"min": 0.7, "max": 1.0},
        "staking_performance": {"min": 0.8, "max": 1.0},
        "trading_performance": {"min": 0.6, "max": 0.9},
        "overall_performance": {"min": 0.7, "max": 1.0}
    }
    
    # Verify validation details
    assert isinstance(result["validation_details"], Dict)
    assert "timestamp" in result["validation_details"]
    assert "validation_results" in result["validation_details"]
    assert "invalid_rows" in result["validation_details"]
    assert "invalid_columns" in result["validation_details"]
    
    # Verify validation results
    assert isinstance(result["validation_details"]["validation_results"], Dict)
    assert "mining_performance" in result["validation_details"]["validation_results"]
    assert "staking_performance" in result["validation_details"]["validation_results"]
    assert "trading_performance" in result["validation_details"]["validation_results"]
    assert "overall_performance" in result["validation_details"]["validation_results"]
    
    # Verify invalid rows
    assert isinstance(result["validation_details"]["invalid_rows"], List)
    assert len(result["validation_details"]["invalid_rows"]) > 0
    assert 0 in result["validation_details"]["invalid_rows"]
    assert 1 in result["validation_details"]["invalid_rows"]
    assert 2 in result["validation_details"]["invalid_rows"]
    assert 3 in result["validation_details"]["invalid_rows"]
    
    # Verify invalid columns
    assert isinstance(result["validation_details"]["invalid_columns"], List)
    assert "mining_performance" in result["validation_details"]["invalid_columns"]
    assert "trading_performance" in result["validation_details"]["invalid_columns"]
    
    # Verify database entry
    db_record = db_session.query(ValidationRecord).filter_by(
        user_id=test_user.user_id,
        validation_id=result["validation_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None 