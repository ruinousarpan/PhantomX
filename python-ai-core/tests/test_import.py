import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from decimal import Decimal
import json
import csv
import os
import tempfile
import io

from core.import_data import (
    import_performance_data,
    import_risk_data,
    import_reward_data,
    import_activity_data,
    import_analytics_data,
    import_report_data,
    parse_import_file,
    validate_import_data,
    ImportError
)
from database.models import User, ImportResult
from database.exceptions import ImportError as DBImportError

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
def test_performance_csv():
    """Create test performance CSV file"""
    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="H"),
        "mining_performance": np.random.uniform(0.8, 0.9, 100),
        "staking_performance": np.random.uniform(0.85, 0.95, 100),
        "trading_performance": np.random.uniform(0.7, 0.8, 100),
        "overall_performance": np.random.uniform(0.8, 0.9, 100)
    })
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    return temp_file.name

@pytest.fixture
def test_risk_json():
    """Create test risk JSON file"""
    data = []
    timestamps = pd.date_range(start="2023-01-01", periods=100, freq="H")
    
    for i in range(100):
        data.append({
            "timestamp": timestamps[i].isoformat(),
            "mining_risk": float(np.random.uniform(0.2, 0.3)),
            "staking_risk": float(np.random.uniform(0.1, 0.2)),
            "trading_risk": float(np.random.uniform(0.3, 0.4)),
            "overall_risk": float(np.random.uniform(0.2, 0.3))
        })
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    with open(temp_file.name, "w") as f:
        json.dump(data, f)
    temp_file.close()
    
    return temp_file.name

@pytest.fixture
def test_reward_csv():
    """Create test reward CSV file"""
    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="H"),
        "mining_rewards": np.random.uniform(0.4, 0.6, 100),
        "staking_rewards": np.random.uniform(0.1, 0.15, 100),
        "trading_rewards": np.random.uniform(0.05, 0.1, 100),
        "overall_rewards": np.random.uniform(0.2, 0.25, 100)
    })
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    return temp_file.name

@pytest.fixture
def test_activity_json():
    """Create test activity JSON file"""
    data = []
    timestamps = pd.date_range(start="2023-01-01", periods=100, freq="H")
    
    for i in range(100):
        data.append({
            "timestamp": timestamps[i].isoformat(),
            "mining_activity": float(np.random.uniform(0.7, 0.9)),
            "staking_activity": float(np.random.uniform(0.8, 0.95)),
            "trading_activity": float(np.random.uniform(0.6, 0.8)),
            "overall_activity": float(np.random.uniform(0.7, 0.85))
        })
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    with open(temp_file.name, "w") as f:
        json.dump(data, f)
    temp_file.close()
    
    return temp_file.name

@pytest.fixture
def test_analytics_json():
    """Create test analytics JSON file"""
    data = {
        "performance_metrics": {
            "mining": {
                "average": 0.85,
                "median": 0.84,
                "std_dev": 0.02,
                "min": 0.80,
                "max": 0.90
            },
            "staking": {
                "average": 0.90,
                "median": 0.89,
                "std_dev": 0.01,
                "min": 0.85,
                "max": 0.95
            },
            "trading": {
                "average": 0.75,
                "median": 0.74,
                "std_dev": 0.03,
                "min": 0.70,
                "max": 0.80
            }
        },
        "risk_metrics": {
            "mining": {
                "average": 0.25,
                "median": 0.24,
                "std_dev": 0.02,
                "min": 0.20,
                "max": 0.30
            },
            "staking": {
                "average": 0.15,
                "median": 0.14,
                "std_dev": 0.01,
                "min": 0.10,
                "max": 0.20
            },
            "trading": {
                "average": 0.35,
                "median": 0.34,
                "std_dev": 0.03,
                "min": 0.30,
                "max": 0.40
            }
        },
        "reward_metrics": {
            "mining": {
                "average": 0.50,
                "median": 0.49,
                "std_dev": 0.05,
                "min": 0.40,
                "max": 0.60
            },
            "staking": {
                "average": 0.12,
                "median": 0.11,
                "std_dev": 0.01,
                "min": 0.10,
                "max": 0.15
            },
            "trading": {
                "average": 0.07,
                "median": 0.06,
                "std_dev": 0.01,
                "min": 0.05,
                "max": 0.10
            }
        }
    }
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    with open(temp_file.name, "w") as f:
        json.dump(data, f)
    temp_file.close()
    
    return temp_file.name

@pytest.fixture
def test_report_json():
    """Create test report JSON file"""
    data = {
        "report_id": "test_report_123",
        "user_id": "test_user",
        "report_type": "PERFORMANCE_REPORT",
        "generated_at": datetime.utcnow().isoformat(),
        "period": "LAST_30_DAYS",
        "summary": {
            "overall_performance": 0.85,
            "overall_risk": 0.25,
            "overall_rewards": 0.23,
            "recommendations": [
                "Increase mining activity to improve rewards",
                "Reduce trading risk by diversifying portfolio",
                "Consider increasing staking allocation"
            ]
        },
        "details": {
            "mining": {
                "performance": 0.85,
                "risk": 0.25,
                "rewards": 0.50,
                "activity": 0.80
            },
            "staking": {
                "performance": 0.90,
                "risk": 0.15,
                "rewards": 0.12,
                "activity": 0.85
            },
            "trading": {
                "performance": 0.75,
                "risk": 0.35,
                "rewards": 0.07,
                "activity": 0.70
            }
        }
    }
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    with open(temp_file.name, "w") as f:
        json.dump(data, f)
    temp_file.close()
    
    return temp_file.name

@pytest.fixture
def test_invalid_csv():
    """Create test invalid CSV file"""
    # Create a temporary file with invalid data
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    with open(temp_file.name, "w") as f:
        f.write("invalid,data,format\n")
        f.write("1,2,3\n")
        f.write("a,b,c\n")
    temp_file.close()
    
    return temp_file.name

@pytest.fixture
def test_invalid_json():
    """Create test invalid JSON file"""
    # Create a temporary file with invalid JSON
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    with open(temp_file.name, "w") as f:
        f.write("{invalid json content")
    temp_file.close()
    
    return temp_file.name

def test_import_performance_data(db_session, test_user, test_performance_csv):
    """Test performance data import"""
    # Import performance data
    result = import_performance_data(
        user_id=test_user.user_id,
        file_path=test_performance_csv,
        import_format="csv",
        db_session=db_session
    )
    
    # Verify import result
    assert isinstance(result, Dict)
    assert "data" in result
    assert "import_format" in result
    assert "row_count" in result
    
    # Verify import metadata
    assert result["import_format"] == "csv"
    assert result["row_count"] == 100
    
    # Verify imported data
    assert isinstance(result["data"], pd.DataFrame)
    assert len(result["data"]) == 100
    assert "timestamp" in result["data"].columns
    assert "mining_performance" in result["data"].columns
    assert "staking_performance" in result["data"].columns
    assert "trading_performance" in result["data"].columns
    assert "overall_performance" in result["data"].columns
    
    # Verify database entry
    db_result = db_session.query(ImportResult).filter_by(
        user_id=test_user.user_id,
        import_type="PERFORMANCE_DATA"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None
    
    # Clean up
    os.remove(test_performance_csv)

def test_import_risk_data(db_session, test_user, test_risk_json):
    """Test risk data import"""
    # Import risk data
    result = import_risk_data(
        user_id=test_user.user_id,
        file_path=test_risk_json,
        import_format="json",
        db_session=db_session
    )
    
    # Verify import result
    assert isinstance(result, Dict)
    assert "data" in result
    assert "import_format" in result
    assert "row_count" in result
    
    # Verify import metadata
    assert result["import_format"] == "json"
    assert result["row_count"] == 100
    
    # Verify imported data
    assert isinstance(result["data"], pd.DataFrame)
    assert len(result["data"]) == 100
    assert "timestamp" in result["data"].columns
    assert "mining_risk" in result["data"].columns
    assert "staking_risk" in result["data"].columns
    assert "trading_risk" in result["data"].columns
    assert "overall_risk" in result["data"].columns
    
    # Verify database entry
    db_result = db_session.query(ImportResult).filter_by(
        user_id=test_user.user_id,
        import_type="RISK_DATA"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None
    
    # Clean up
    os.remove(test_risk_json)

def test_import_reward_data(db_session, test_user, test_reward_csv):
    """Test reward data import"""
    # Import reward data
    result = import_reward_data(
        user_id=test_user.user_id,
        file_path=test_reward_csv,
        import_format="csv",
        db_session=db_session
    )
    
    # Verify import result
    assert isinstance(result, Dict)
    assert "data" in result
    assert "import_format" in result
    assert "row_count" in result
    
    # Verify import metadata
    assert result["import_format"] == "csv"
    assert result["row_count"] == 100
    
    # Verify imported data
    assert isinstance(result["data"], pd.DataFrame)
    assert len(result["data"]) == 100
    assert "timestamp" in result["data"].columns
    assert "mining_rewards" in result["data"].columns
    assert "staking_rewards" in result["data"].columns
    assert "trading_rewards" in result["data"].columns
    assert "overall_rewards" in result["data"].columns
    
    # Verify database entry
    db_result = db_session.query(ImportResult).filter_by(
        user_id=test_user.user_id,
        import_type="REWARD_DATA"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None
    
    # Clean up
    os.remove(test_reward_csv)

def test_import_activity_data(db_session, test_user, test_activity_json):
    """Test activity data import"""
    # Import activity data
    result = import_activity_data(
        user_id=test_user.user_id,
        file_path=test_activity_json,
        import_format="json",
        db_session=db_session
    )
    
    # Verify import result
    assert isinstance(result, Dict)
    assert "data" in result
    assert "import_format" in result
    assert "row_count" in result
    
    # Verify import metadata
    assert result["import_format"] == "json"
    assert result["row_count"] == 100
    
    # Verify imported data
    assert isinstance(result["data"], pd.DataFrame)
    assert len(result["data"]) == 100
    assert "timestamp" in result["data"].columns
    assert "mining_activity" in result["data"].columns
    assert "staking_activity" in result["data"].columns
    assert "trading_activity" in result["data"].columns
    assert "overall_activity" in result["data"].columns
    
    # Verify database entry
    db_result = db_session.query(ImportResult).filter_by(
        user_id=test_user.user_id,
        import_type="ACTIVITY_DATA"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None
    
    # Clean up
    os.remove(test_activity_json)

def test_import_analytics_data(db_session, test_user, test_analytics_json):
    """Test analytics data import"""
    # Import analytics data
    result = import_analytics_data(
        user_id=test_user.user_id,
        file_path=test_analytics_json,
        import_format="json",
        db_session=db_session
    )
    
    # Verify import result
    assert isinstance(result, Dict)
    assert "data" in result
    assert "import_format" in result
    
    # Verify import metadata
    assert result["import_format"] == "json"
    
    # Verify imported data
    assert isinstance(result["data"], Dict)
    assert "performance_metrics" in result["data"]
    assert "risk_metrics" in result["data"]
    assert "reward_metrics" in result["data"]
    assert "mining" in result["data"]["performance_metrics"]
    assert "staking" in result["data"]["performance_metrics"]
    assert "trading" in result["data"]["performance_metrics"]
    
    # Verify database entry
    db_result = db_session.query(ImportResult).filter_by(
        user_id=test_user.user_id,
        import_type="ANALYTICS_DATA"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None
    
    # Clean up
    os.remove(test_analytics_json)

def test_import_report_data(db_session, test_user, test_report_json):
    """Test report data import"""
    # Import report data
    result = import_report_data(
        user_id=test_user.user_id,
        file_path=test_report_json,
        import_format="json",
        db_session=db_session
    )
    
    # Verify import result
    assert isinstance(result, Dict)
    assert "data" in result
    assert "import_format" in result
    
    # Verify import metadata
    assert result["import_format"] == "json"
    
    # Verify imported data
    assert isinstance(result["data"], Dict)
    assert "report_id" in result["data"]
    assert "user_id" in result["data"]
    assert "report_type" in result["data"]
    assert "generated_at" in result["data"]
    assert "period" in result["data"]
    assert "summary" in result["data"]
    assert "details" in result["data"]
    
    # Verify database entry
    db_result = db_session.query(ImportResult).filter_by(
        user_id=test_user.user_id,
        import_type="REPORT_DATA"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None
    
    # Clean up
    os.remove(test_report_json)

def test_parse_import_file(db_session, test_user, test_performance_csv):
    """Test import file parsing"""
    # Parse import file
    result = parse_import_file(
        user_id=test_user.user_id,
        file_path=test_performance_csv,
        file_format="csv",
        db_session=db_session
    )
    
    # Verify parsing result
    assert isinstance(result, Dict)
    assert "parsed_data" in result
    assert "file_format" in result
    
    # Verify format metadata
    assert result["file_format"] == "csv"
    
    # Verify parsed data
    assert isinstance(result["parsed_data"], pd.DataFrame)
    assert len(result["parsed_data"]) == 100
    assert "timestamp" in result["parsed_data"].columns
    assert "mining_performance" in result["parsed_data"].columns
    assert "staking_performance" in result["parsed_data"].columns
    assert "trading_performance" in result["parsed_data"].columns
    assert "overall_performance" in result["parsed_data"].columns
    
    # Verify database entry
    db_result = db_session.query(ImportResult).filter_by(
        user_id=test_user.user_id,
        import_type="FILE_PARSING"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None
    
    # Clean up
    os.remove(test_performance_csv)

def test_validate_import_data(db_session, test_user, test_performance_csv):
    """Test import data validation"""
    # Parse the file first
    parsed_result = parse_import_file(
        user_id=test_user.user_id,
        file_path=test_performance_csv,
        file_format="csv",
        db_session=db_session
    )
    
    # Validate the parsed data
    result = validate_import_data(
        user_id=test_user.user_id,
        data=parsed_result["parsed_data"],
        data_type="performance",
        db_session=db_session
    )
    
    # Verify validation result
    assert isinstance(result, Dict)
    assert "is_valid" in result
    assert "validation_errors" in result
    assert "data_type" in result
    
    # Verify validation metadata
    assert result["is_valid"] is True
    assert len(result["validation_errors"]) == 0
    assert result["data_type"] == "performance"
    
    # Verify database entry
    db_result = db_session.query(ImportResult).filter_by(
        user_id=test_user.user_id,
        import_type="DATA_VALIDATION"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None
    
    # Clean up
    os.remove(test_performance_csv)

def test_import_error_handling(db_session, test_user, test_invalid_csv, test_invalid_json):
    """Test import error handling"""
    # Invalid user ID
    with pytest.raises(ImportError) as excinfo:
        import_performance_data(
            user_id=None,
            file_path=test_performance_csv,
            import_format="csv",
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid file path
    with pytest.raises(ImportError) as excinfo:
        import_risk_data(
            user_id=test_user.user_id,
            file_path="nonexistent_file.csv",
            import_format="csv",
            db_session=db_session
        )
    assert "File not found" in str(excinfo.value)
    
    # Invalid import format
    with pytest.raises(ImportError) as excinfo:
        import_reward_data(
            user_id=test_user.user_id,
            file_path=test_reward_csv,
            import_format="invalid_format",
            db_session=db_session
        )
    assert "Invalid import format" in str(excinfo.value)
    
    # Invalid CSV format
    with pytest.raises(ImportError) as excinfo:
        import_performance_data(
            user_id=test_user.user_id,
            file_path=test_invalid_csv,
            import_format="csv",
            db_session=db_session
        )
    assert "Invalid CSV format" in str(excinfo.value)
    
    # Invalid JSON format
    with pytest.raises(ImportError) as excinfo:
        import_analytics_data(
            user_id=test_user.user_id,
            file_path=test_invalid_json,
            import_format="json",
            db_session=db_session
        )
    assert "Invalid JSON format" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBImportError) as excinfo:
        import_activity_data(
            user_id=test_user.user_id,
            file_path=test_activity_json,
            import_format="json",
            db_session=None
        )
    assert "Database error" in str(excinfo.value)
    
    # Clean up
    os.remove(test_invalid_csv)
    os.remove(test_invalid_json) 