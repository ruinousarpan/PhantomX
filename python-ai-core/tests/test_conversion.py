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

from core.conversion import (
    convert_data,
    get_conversion_info,
    ConversionError
)
from database.models import User, ConversionRecord
from database.exceptions import ConversionError as DBConversionError

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
def test_conversion_config():
    """Create test conversion configuration"""
    return {
        "conversion_type": "type",
        "columns": None,  # Convert all columns
        "exclude_columns": ["timestamp", "day"],
        "method": "float",
        "params": {}
    }

def test_convert_performance_data(db_session, test_user, test_performance_data, test_conversion_config):
    """Test converting performance data"""
    # Convert performance data
    result = convert_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        conversion_config=test_conversion_config,
        db_session=db_session
    )
    
    # Verify conversion result
    assert isinstance(result, Dict)
    assert "conversion_id" in result
    assert "converted_data" in result
    assert "conversion_details" in result
    
    # Verify conversion metadata
    assert result["conversion_type"] == "type"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["method"] == "float"
    assert result["params"] == {}
    
    # Verify conversion details
    assert isinstance(result["conversion_details"], Dict)
    assert "timestamp" in result["conversion_details"]
    assert "conversion_results" in result["conversion_details"]
    assert "converted_columns" in result["conversion_details"]
    
    # Verify conversion results
    assert isinstance(result["conversion_details"]["conversion_results"], Dict)
    assert "mining_performance" in result["conversion_details"]["conversion_results"]
    assert "staking_performance" in result["conversion_details"]["conversion_results"]
    assert "trading_performance" in result["conversion_details"]["conversion_results"]
    assert "overall_performance" in result["conversion_details"]["conversion_results"]
    
    # Verify converted data
    assert isinstance(result["converted_data"], pd.DataFrame)
    assert "mining_performance" in result["converted_data"].columns
    assert "staking_performance" in result["converted_data"].columns
    assert "trading_performance" in result["converted_data"].columns
    assert "overall_performance" in result["converted_data"].columns
    assert "timestamp" in result["converted_data"].columns
    assert "day" in result["converted_data"].columns
    
    # Verify data conversion
    assert result["converted_data"]["mining_performance"].dtype == np.float64
    assert result["converted_data"]["staking_performance"].dtype == np.float64
    assert result["converted_data"]["trading_performance"].dtype == np.float64
    assert result["converted_data"]["overall_performance"].dtype == np.float64
    
    # Verify database entry
    db_record = db_session.query(ConversionRecord).filter_by(
        user_id=test_user.user_id,
        conversion_id=result["conversion_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_convert_risk_data(db_session, test_user, test_risk_data):
    """Test converting risk data"""
    # Create conversion config for risk data
    risk_config = {
        "conversion_type": "type",
        "columns": None,  # Convert all columns
        "exclude_columns": ["timestamp", "day"],
        "method": "float",
        "params": {}
    }
    
    # Convert risk data
    result = convert_data(
        user_id=test_user.user_id,
        data=test_risk_data,
        conversion_config=risk_config,
        db_session=db_session
    )
    
    # Verify conversion result
    assert isinstance(result, Dict)
    assert "conversion_id" in result
    assert "converted_data" in result
    assert "conversion_details" in result
    
    # Verify conversion metadata
    assert result["conversion_type"] == "type"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["method"] == "float"
    assert result["params"] == {}
    
    # Verify conversion details
    assert isinstance(result["conversion_details"], Dict)
    assert "timestamp" in result["conversion_details"]
    assert "conversion_results" in result["conversion_details"]
    assert "converted_columns" in result["conversion_details"]
    
    # Verify conversion results
    assert isinstance(result["conversion_details"]["conversion_results"], Dict)
    assert "mining_risk" in result["conversion_details"]["conversion_results"]
    assert "staking_risk" in result["conversion_details"]["conversion_results"]
    assert "trading_risk" in result["conversion_details"]["conversion_results"]
    assert "overall_risk" in result["conversion_details"]["conversion_results"]
    
    # Verify converted data
    assert isinstance(result["converted_data"], pd.DataFrame)
    assert "mining_risk" in result["converted_data"].columns
    assert "staking_risk" in result["converted_data"].columns
    assert "trading_risk" in result["converted_data"].columns
    assert "overall_risk" in result["converted_data"].columns
    assert "timestamp" in result["converted_data"].columns
    assert "day" in result["converted_data"].columns
    
    # Verify data conversion
    assert result["converted_data"]["mining_risk"].dtype == np.float64
    assert result["converted_data"]["staking_risk"].dtype == np.float64
    assert result["converted_data"]["trading_risk"].dtype == np.float64
    assert result["converted_data"]["overall_risk"].dtype == np.float64
    
    # Verify database entry
    db_record = db_session.query(ConversionRecord).filter_by(
        user_id=test_user.user_id,
        conversion_id=result["conversion_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_convert_reward_data(db_session, test_user, test_reward_data):
    """Test converting reward data"""
    # Create conversion config for reward data
    reward_config = {
        "conversion_type": "type",
        "columns": None,  # Convert all columns
        "exclude_columns": ["timestamp", "day"],
        "method": "float",
        "params": {}
    }
    
    # Convert reward data
    result = convert_data(
        user_id=test_user.user_id,
        data=test_reward_data,
        conversion_config=reward_config,
        db_session=db_session
    )
    
    # Verify conversion result
    assert isinstance(result, Dict)
    assert "conversion_id" in result
    assert "converted_data" in result
    assert "conversion_details" in result
    
    # Verify conversion metadata
    assert result["conversion_type"] == "type"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["method"] == "float"
    assert result["params"] == {}
    
    # Verify conversion details
    assert isinstance(result["conversion_details"], Dict)
    assert "timestamp" in result["conversion_details"]
    assert "conversion_results" in result["conversion_details"]
    assert "converted_columns" in result["conversion_details"]
    
    # Verify conversion results
    assert isinstance(result["conversion_details"]["conversion_results"], Dict)
    assert "mining_reward" in result["conversion_details"]["conversion_results"]
    assert "staking_reward" in result["conversion_details"]["conversion_results"]
    assert "trading_reward" in result["conversion_details"]["conversion_results"]
    assert "overall_reward" in result["conversion_details"]["conversion_results"]
    
    # Verify converted data
    assert isinstance(result["converted_data"], pd.DataFrame)
    assert "mining_reward" in result["converted_data"].columns
    assert "staking_reward" in result["converted_data"].columns
    assert "trading_reward" in result["converted_data"].columns
    assert "overall_reward" in result["converted_data"].columns
    assert "timestamp" in result["converted_data"].columns
    assert "day" in result["converted_data"].columns
    
    # Verify data conversion
    assert result["converted_data"]["mining_reward"].dtype == np.float64
    assert result["converted_data"]["staking_reward"].dtype == np.float64
    assert result["converted_data"]["trading_reward"].dtype == np.float64
    assert result["converted_data"]["overall_reward"].dtype == np.float64
    
    # Verify database entry
    db_record = db_session.query(ConversionRecord).filter_by(
        user_id=test_user.user_id,
        conversion_id=result["conversion_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_convert_activity_data(db_session, test_user, test_activity_data):
    """Test converting activity data"""
    # Create conversion config for activity data
    activity_config = {
        "conversion_type": "type",
        "columns": None,  # Convert all columns
        "exclude_columns": ["timestamp", "day"],
        "method": "float",
        "params": {}
    }
    
    # Convert activity data
    result = convert_data(
        user_id=test_user.user_id,
        data=test_activity_data,
        conversion_config=activity_config,
        db_session=db_session
    )
    
    # Verify conversion result
    assert isinstance(result, Dict)
    assert "conversion_id" in result
    assert "converted_data" in result
    assert "conversion_details" in result
    
    # Verify conversion metadata
    assert result["conversion_type"] == "type"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["method"] == "float"
    assert result["params"] == {}
    
    # Verify conversion details
    assert isinstance(result["conversion_details"], Dict)
    assert "timestamp" in result["conversion_details"]
    assert "conversion_results" in result["conversion_details"]
    assert "converted_columns" in result["conversion_details"]
    
    # Verify conversion results
    assert isinstance(result["conversion_details"]["conversion_results"], Dict)
    assert "mining_activity" in result["conversion_details"]["conversion_results"]
    assert "staking_activity" in result["conversion_details"]["conversion_results"]
    assert "trading_activity" in result["conversion_details"]["conversion_results"]
    assert "overall_activity" in result["conversion_details"]["conversion_results"]
    
    # Verify converted data
    assert isinstance(result["converted_data"], pd.DataFrame)
    assert "mining_activity" in result["converted_data"].columns
    assert "staking_activity" in result["converted_data"].columns
    assert "trading_activity" in result["converted_data"].columns
    assert "overall_activity" in result["converted_data"].columns
    assert "timestamp" in result["converted_data"].columns
    assert "day" in result["converted_data"].columns
    
    # Verify data conversion
    assert result["converted_data"]["mining_activity"].dtype == np.float64
    assert result["converted_data"]["staking_activity"].dtype == np.float64
    assert result["converted_data"]["trading_activity"].dtype == np.float64
    assert result["converted_data"]["overall_activity"].dtype == np.float64
    
    # Verify database entry
    db_record = db_session.query(ConversionRecord).filter_by(
        user_id=test_user.user_id,
        conversion_id=result["conversion_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_convert_with_integer(db_session, test_user, test_performance_data):
    """Test converting with integer"""
    # Create conversion config with integer
    integer_config = {
        "conversion_type": "type",
        "columns": None,  # Convert all columns
        "exclude_columns": ["timestamp", "day"],
        "method": "int",
        "params": {}
    }
    
    # Convert performance data
    result = convert_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        conversion_config=integer_config,
        db_session=db_session
    )
    
    # Verify conversion result
    assert isinstance(result, Dict)
    assert "conversion_id" in result
    assert "converted_data" in result
    assert "conversion_details" in result
    
    # Verify conversion metadata
    assert result["conversion_type"] == "type"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["method"] == "int"
    assert result["params"] == {}
    
    # Verify conversion details
    assert isinstance(result["conversion_details"], Dict)
    assert "timestamp" in result["conversion_details"]
    assert "conversion_results" in result["conversion_details"]
    assert "converted_columns" in result["conversion_details"]
    
    # Verify conversion results
    assert isinstance(result["conversion_details"]["conversion_results"], Dict)
    assert "mining_performance" in result["conversion_details"]["conversion_results"]
    assert "staking_performance" in result["conversion_details"]["conversion_results"]
    assert "trading_performance" in result["conversion_details"]["conversion_results"]
    assert "overall_performance" in result["conversion_details"]["conversion_results"]
    
    # Verify converted data
    assert isinstance(result["converted_data"], pd.DataFrame)
    assert "mining_performance" in result["converted_data"].columns
    assert "staking_performance" in result["converted_data"].columns
    assert "trading_performance" in result["converted_data"].columns
    assert "overall_performance" in result["converted_data"].columns
    assert "timestamp" in result["converted_data"].columns
    assert "day" in result["converted_data"].columns
    
    # Verify data conversion
    assert result["converted_data"]["mining_performance"].dtype == np.int64
    assert result["converted_data"]["staking_performance"].dtype == np.int64
    assert result["converted_data"]["trading_performance"].dtype == np.int64
    assert result["converted_data"]["overall_performance"].dtype == np.int64
    
    # Verify database entry
    db_record = db_session.query(ConversionRecord).filter_by(
        user_id=test_user.user_id,
        conversion_id=result["conversion_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_convert_with_string(db_session, test_user, test_performance_data):
    """Test converting with string"""
    # Create conversion config with string
    string_config = {
        "conversion_type": "type",
        "columns": None,  # Convert all columns
        "exclude_columns": ["timestamp", "day"],
        "method": "str",
        "params": {}
    }
    
    # Convert performance data
    result = convert_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        conversion_config=string_config,
        db_session=db_session
    )
    
    # Verify conversion result
    assert isinstance(result, Dict)
    assert "conversion_id" in result
    assert "converted_data" in result
    assert "conversion_details" in result
    
    # Verify conversion metadata
    assert result["conversion_type"] == "type"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["method"] == "str"
    assert result["params"] == {}
    
    # Verify conversion details
    assert isinstance(result["conversion_details"], Dict)
    assert "timestamp" in result["conversion_details"]
    assert "conversion_results" in result["conversion_details"]
    assert "converted_columns" in result["conversion_details"]
    
    # Verify conversion results
    assert isinstance(result["conversion_details"]["conversion_results"], Dict)
    assert "mining_performance" in result["conversion_details"]["conversion_results"]
    assert "staking_performance" in result["conversion_details"]["conversion_results"]
    assert "trading_performance" in result["conversion_details"]["conversion_results"]
    assert "overall_performance" in result["conversion_details"]["conversion_results"]
    
    # Verify converted data
    assert isinstance(result["converted_data"], pd.DataFrame)
    assert "mining_performance" in result["converted_data"].columns
    assert "staking_performance" in result["converted_data"].columns
    assert "trading_performance" in result["converted_data"].columns
    assert "overall_performance" in result["converted_data"].columns
    assert "timestamp" in result["converted_data"].columns
    assert "day" in result["converted_data"].columns
    
    # Verify data conversion
    assert result["converted_data"]["mining_performance"].dtype == object
    assert result["converted_data"]["staking_performance"].dtype == object
    assert result["converted_data"]["trading_performance"].dtype == object
    assert result["converted_data"]["overall_performance"].dtype == object
    
    # Verify database entry
    db_record = db_session.query(ConversionRecord).filter_by(
        user_id=test_user.user_id,
        conversion_id=result["conversion_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_convert_with_decimal(db_session, test_user, test_performance_data):
    """Test converting with decimal"""
    # Create conversion config with decimal
    decimal_config = {
        "conversion_type": "type",
        "columns": None,  # Convert all columns
        "exclude_columns": ["timestamp", "day"],
        "method": "decimal",
        "params": {
            "precision": 2
        }
    }
    
    # Convert performance data
    result = convert_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        conversion_config=decimal_config,
        db_session=db_session
    )
    
    # Verify conversion result
    assert isinstance(result, Dict)
    assert "conversion_id" in result
    assert "converted_data" in result
    assert "conversion_details" in result
    
    # Verify conversion metadata
    assert result["conversion_type"] == "type"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["method"] == "decimal"
    assert result["params"] == {
        "precision": 2
    }
    
    # Verify conversion details
    assert isinstance(result["conversion_details"], Dict)
    assert "timestamp" in result["conversion_details"]
    assert "conversion_results" in result["conversion_details"]
    assert "converted_columns" in result["conversion_details"]
    
    # Verify conversion results
    assert isinstance(result["conversion_details"]["conversion_results"], Dict)
    assert "mining_performance" in result["conversion_details"]["conversion_results"]
    assert "staking_performance" in result["conversion_details"]["conversion_results"]
    assert "trading_performance" in result["conversion_details"]["conversion_results"]
    assert "overall_performance" in result["conversion_details"]["conversion_results"]
    
    # Verify converted data
    assert isinstance(result["converted_data"], pd.DataFrame)
    assert "mining_performance" in result["converted_data"].columns
    assert "staking_performance" in result["converted_data"].columns
    assert "trading_performance" in result["converted_data"].columns
    assert "overall_performance" in result["converted_data"].columns
    assert "timestamp" in result["converted_data"].columns
    assert "day" in result["converted_data"].columns
    
    # Verify data conversion
    assert isinstance(result["converted_data"]["mining_performance"].iloc[0], Decimal)
    assert isinstance(result["converted_data"]["staking_performance"].iloc[0], Decimal)
    assert isinstance(result["converted_data"]["trading_performance"].iloc[0], Decimal)
    assert isinstance(result["converted_data"]["overall_performance"].iloc[0], Decimal)
    
    # Verify database entry
    db_record = db_session.query(ConversionRecord).filter_by(
        user_id=test_user.user_id,
        conversion_id=result["conversion_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_convert_with_specific_columns(db_session, test_user, test_performance_data):
    """Test converting with specific columns"""
    # Create conversion config with specific columns
    specific_columns_config = {
        "conversion_type": "type",
        "columns": ["mining_performance", "trading_performance"],  # Only convert these columns
        "exclude_columns": ["timestamp", "day"],
        "method": "int",
        "params": {}
    }
    
    # Convert performance data
    result = convert_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        conversion_config=specific_columns_config,
        db_session=db_session
    )
    
    # Verify conversion result
    assert isinstance(result, Dict)
    assert "conversion_id" in result
    assert "converted_data" in result
    assert "conversion_details" in result
    
    # Verify conversion metadata
    assert result["conversion_type"] == "type"
    assert result["columns"] == ["mining_performance", "trading_performance"]
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["method"] == "int"
    assert result["params"] == {}
    
    # Verify conversion details
    assert isinstance(result["conversion_details"], Dict)
    assert "timestamp" in result["conversion_details"]
    assert "conversion_results" in result["conversion_details"]
    assert "converted_columns" in result["conversion_details"]
    
    # Verify conversion results
    assert isinstance(result["conversion_details"]["conversion_results"], Dict)
    assert "mining_performance" in result["conversion_details"]["conversion_results"]
    assert "trading_performance" in result["conversion_details"]["conversion_results"]
    assert "staking_performance" not in result["conversion_details"]["conversion_results"]
    assert "overall_performance" not in result["conversion_details"]["conversion_results"]
    
    # Verify converted data
    assert isinstance(result["converted_data"], pd.DataFrame)
    assert "mining_performance" in result["converted_data"].columns
    assert "trading_performance" in result["converted_data"].columns
    assert "staking_performance" in result["converted_data"].columns
    assert "overall_performance" in result["converted_data"].columns
    assert "timestamp" in result["converted_data"].columns
    assert "day" in result["converted_data"].columns
    
    # Verify data conversion
    assert result["converted_data"]["mining_performance"].dtype == np.int64
    assert result["converted_data"]["trading_performance"].dtype == np.int64
    assert result["converted_data"]["staking_performance"].dtype == np.float64
    assert result["converted_data"]["overall_performance"].dtype == np.float64
    
    # Verify database entry
    db_record = db_session.query(ConversionRecord).filter_by(
        user_id=test_user.user_id,
        conversion_id=result["conversion_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_get_conversion_info(db_session, test_user, test_performance_data, test_conversion_config):
    """Test conversion info retrieval"""
    # First, convert performance data
    conversion_result = convert_data(
        user_id=test_user.user_id,
        data=test_performance_data,
        conversion_config=test_conversion_config,
        db_session=db_session
    )
    
    conversion_id = conversion_result["conversion_id"]
    
    # Get conversion info
    result = get_conversion_info(
        user_id=test_user.user_id,
        conversion_id=conversion_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "conversion_id" in result
    assert result["conversion_id"] == conversion_id
    
    # Verify conversion metadata
    assert result["conversion_type"] == "type"
    assert result["columns"] is None
    assert result["exclude_columns"] == ["timestamp", "day"]
    assert result["method"] == "float"
    assert result["params"] == {}
    
    # Verify conversion details
    assert "conversion_details" in result
    assert isinstance(result["conversion_details"], Dict)
    assert "timestamp" in result["conversion_details"]
    assert "conversion_results" in result["conversion_details"]
    assert "converted_columns" in result["conversion_details"]
    
    # Verify database entry
    db_record = db_session.query(ConversionRecord).filter_by(
        user_id=test_user.user_id,
        conversion_id=conversion_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_conversion_error_handling(db_session, test_user):
    """Test conversion error handling"""
    # Invalid user ID
    with pytest.raises(ConversionError) as excinfo:
        convert_data(
            user_id=None,
            data=pd.DataFrame(),
            conversion_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(ConversionError) as excinfo:
        convert_data(
            user_id=test_user.user_id,
            data=None,
            conversion_config={},
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Invalid conversion type
    with pytest.raises(ConversionError) as excinfo:
        convert_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            conversion_config={"conversion_type": "invalid_type"},
            db_session=db_session
        )
    assert "Invalid conversion type" in str(excinfo.value)
    
    # Invalid method
    with pytest.raises(ConversionError) as excinfo:
        convert_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            conversion_config={"conversion_type": "type", "method": "invalid_method"},
            db_session=db_session
        )
    assert "Invalid conversion method" in str(excinfo.value)
    
    # Invalid columns
    with pytest.raises(ConversionError) as excinfo:
        convert_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            conversion_config={"conversion_type": "type", "method": "float", "columns": ["invalid_column"]},
            db_session=db_session
        )
    assert "Invalid columns" in str(excinfo.value)
    
    # Invalid conversion ID
    with pytest.raises(ConversionError) as excinfo:
        get_conversion_info(
            user_id=test_user.user_id,
            conversion_id="invalid_conversion_id",
            db_session=db_session
        )
    assert "Invalid conversion ID" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBConversionError) as excinfo:
        convert_data(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            conversion_config={"conversion_type": "type", "method": "float"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 