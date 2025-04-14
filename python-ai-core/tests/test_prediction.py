import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from decimal import Decimal
import json
import os
import tempfile

from core.prediction import (
    predict_performance,
    predict_risk,
    predict_reward,
    predict_activity,
    forecast_trends,
    project_metrics,
    evaluate_prediction,
    PredictionError
)
from database.models import User, PredictionResult
from database.exceptions import PredictionError as DBPredictionError

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
def test_historical_data():
    """Create test historical data"""
    return {
        "performance": pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=365, freq="D"),
            "mining_performance": np.random.uniform(0.8, 0.9, 365),
            "staking_performance": np.random.uniform(0.85, 0.95, 365),
            "trading_performance": np.random.uniform(0.7, 0.8, 365),
            "overall_performance": np.random.uniform(0.8, 0.9, 365)
        }),
        "risk": pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=365, freq="D"),
            "mining_risk": np.random.uniform(0.2, 0.3, 365),
            "staking_risk": np.random.uniform(0.1, 0.2, 365),
            "trading_risk": np.random.uniform(0.3, 0.4, 365),
            "overall_risk": np.random.uniform(0.2, 0.3, 365)
        }),
        "reward": pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=365, freq="D"),
            "mining_rewards": np.random.uniform(0.4, 0.6, 365),
            "staking_rewards": np.random.uniform(0.1, 0.15, 365),
            "trading_rewards": np.random.uniform(0.05, 0.1, 365),
            "overall_rewards": np.random.uniform(0.2, 0.25, 365)
        }),
        "activity": pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=365, freq="D"),
            "mining_activity": np.random.uniform(0.7, 0.9, 365),
            "staking_activity": np.random.uniform(0.8, 0.95, 365),
            "trading_activity": np.random.uniform(0.6, 0.8, 365),
            "overall_activity": np.random.uniform(0.7, 0.85, 365)
        })
    }

def test_predict_performance(db_session, test_user, test_performance_data):
    """Test performance prediction"""
    # Predict performance
    result = predict_performance(
        user_id=test_user.user_id,
        data=test_performance_data,
        prediction_horizon="7D",
        db_session=db_session
    )
    
    # Verify prediction result
    assert isinstance(result, Dict)
    assert "predictions" in result
    assert "prediction_horizon" in result
    
    # Verify prediction metadata
    assert result["prediction_horizon"] == "7D"
    
    # Verify predictions
    assert isinstance(result["predictions"], pd.DataFrame)
    assert len(result["predictions"]) > 0
    assert "timestamp" in result["predictions"].columns
    assert "mining_performance" in result["predictions"].columns
    assert "staking_performance" in result["predictions"].columns
    assert "trading_performance" in result["predictions"].columns
    assert "overall_performance" in result["predictions"].columns
    
    # Verify database entry
    db_result = db_session.query(PredictionResult).filter_by(
        user_id=test_user.user_id,
        prediction_type="PERFORMANCE_PREDICTION"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_predict_risk(db_session, test_user, test_risk_data):
    """Test risk prediction"""
    # Predict risk
    result = predict_risk(
        user_id=test_user.user_id,
        data=test_risk_data,
        prediction_horizon="7D",
        db_session=db_session
    )
    
    # Verify prediction result
    assert isinstance(result, Dict)
    assert "predictions" in result
    assert "prediction_horizon" in result
    
    # Verify prediction metadata
    assert result["prediction_horizon"] == "7D"
    
    # Verify predictions
    assert isinstance(result["predictions"], pd.DataFrame)
    assert len(result["predictions"]) > 0
    assert "timestamp" in result["predictions"].columns
    assert "mining_risk" in result["predictions"].columns
    assert "staking_risk" in result["predictions"].columns
    assert "trading_risk" in result["predictions"].columns
    assert "overall_risk" in result["predictions"].columns
    
    # Verify database entry
    db_result = db_session.query(PredictionResult).filter_by(
        user_id=test_user.user_id,
        prediction_type="RISK_PREDICTION"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_predict_reward(db_session, test_user, test_reward_data):
    """Test reward prediction"""
    # Predict reward
    result = predict_reward(
        user_id=test_user.user_id,
        data=test_reward_data,
        prediction_horizon="7D",
        db_session=db_session
    )
    
    # Verify prediction result
    assert isinstance(result, Dict)
    assert "predictions" in result
    assert "prediction_horizon" in result
    
    # Verify prediction metadata
    assert result["prediction_horizon"] == "7D"
    
    # Verify predictions
    assert isinstance(result["predictions"], pd.DataFrame)
    assert len(result["predictions"]) > 0
    assert "timestamp" in result["predictions"].columns
    assert "mining_rewards" in result["predictions"].columns
    assert "staking_rewards" in result["predictions"].columns
    assert "trading_rewards" in result["predictions"].columns
    assert "overall_rewards" in result["predictions"].columns
    
    # Verify database entry
    db_result = db_session.query(PredictionResult).filter_by(
        user_id=test_user.user_id,
        prediction_type="REWARD_PREDICTION"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_predict_activity(db_session, test_user, test_activity_data):
    """Test activity prediction"""
    # Predict activity
    result = predict_activity(
        user_id=test_user.user_id,
        data=test_activity_data,
        prediction_horizon="7D",
        db_session=db_session
    )
    
    # Verify prediction result
    assert isinstance(result, Dict)
    assert "predictions" in result
    assert "prediction_horizon" in result
    
    # Verify prediction metadata
    assert result["prediction_horizon"] == "7D"
    
    # Verify predictions
    assert isinstance(result["predictions"], pd.DataFrame)
    assert len(result["predictions"]) > 0
    assert "timestamp" in result["predictions"].columns
    assert "mining_activity" in result["predictions"].columns
    assert "staking_activity" in result["predictions"].columns
    assert "trading_activity" in result["predictions"].columns
    assert "overall_activity" in result["predictions"].columns
    
    # Verify database entry
    db_result = db_session.query(PredictionResult).filter_by(
        user_id=test_user.user_id,
        prediction_type="ACTIVITY_PREDICTION"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_forecast_trends(db_session, test_user, test_historical_data):
    """Test trend forecasting"""
    # Forecast trends
    result = forecast_trends(
        user_id=test_user.user_id,
        historical_data=test_historical_data,
        forecast_period="30D",
        db_session=db_session
    )
    
    # Verify forecast result
    assert isinstance(result, Dict)
    assert "forecasts" in result
    assert "forecast_period" in result
    
    # Verify forecast metadata
    assert result["forecast_period"] == "30D"
    
    # Verify forecasts
    assert isinstance(result["forecasts"], Dict)
    assert "performance" in result["forecasts"]
    assert "risk" in result["forecasts"]
    assert "reward" in result["forecasts"]
    assert "activity" in result["forecasts"]
    
    # Verify database entry
    db_result = db_session.query(PredictionResult).filter_by(
        user_id=test_user.user_id,
        prediction_type="TREND_FORECAST"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_project_metrics(db_session, test_user, test_historical_data):
    """Test metrics projection"""
    # Project metrics
    result = project_metrics(
        user_id=test_user.user_id,
        historical_data=test_historical_data,
        projection_period="90D",
        db_session=db_session
    )
    
    # Verify projection result
    assert isinstance(result, Dict)
    assert "projections" in result
    assert "projection_period" in result
    
    # Verify projection metadata
    assert result["projection_period"] == "90D"
    
    # Verify projections
    assert isinstance(result["projections"], Dict)
    assert "performance" in result["projections"]
    assert "risk" in result["projections"]
    assert "reward" in result["projections"]
    assert "activity" in result["projections"]
    
    # Verify database entry
    db_result = db_session.query(PredictionResult).filter_by(
        user_id=test_user.user_id,
        prediction_type="METRICS_PROJECTION"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_evaluate_prediction(db_session, test_user, test_performance_data):
    """Test prediction evaluation"""
    # Evaluate prediction
    result = evaluate_prediction(
        user_id=test_user.user_id,
        actual_data=test_performance_data,
        predicted_data=test_performance_data.copy(),
        evaluation_metrics=["mae", "rmse", "r2"],
        db_session=db_session
    )
    
    # Verify evaluation result
    assert isinstance(result, Dict)
    assert "evaluation" in result
    assert "evaluation_metrics" in result
    
    # Verify evaluation metadata
    assert result["evaluation_metrics"] == ["mae", "rmse", "r2"]
    
    # Verify evaluation results
    assert isinstance(result["evaluation"], Dict)
    assert "mae" in result["evaluation"]
    assert "rmse" in result["evaluation"]
    assert "r2" in result["evaluation"]
    
    # Verify database entry
    db_result = db_session.query(PredictionResult).filter_by(
        user_id=test_user.user_id,
        prediction_type="PREDICTION_EVALUATION"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_prediction_error_handling(db_session, test_user):
    """Test prediction error handling"""
    # Invalid user ID
    with pytest.raises(PredictionError) as excinfo:
        predict_performance(
            user_id=None,
            data=pd.DataFrame(),
            prediction_horizon="7D",
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(PredictionError) as excinfo:
        predict_risk(
            user_id=test_user.user_id,
            data=None,
            prediction_horizon="7D",
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Empty data
    with pytest.raises(PredictionError) as excinfo:
        predict_reward(
            user_id=test_user.user_id,
            data=pd.DataFrame(),
            prediction_horizon="7D",
            db_session=db_session
        )
    assert "Empty data" in str(excinfo.value)
    
    # Invalid prediction horizon
    with pytest.raises(PredictionError) as excinfo:
        predict_activity(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            prediction_horizon="invalid_horizon",
            db_session=db_session
        )
    assert "Invalid prediction horizon" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBPredictionError) as excinfo:
        predict_performance(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            prediction_horizon="7D",
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 