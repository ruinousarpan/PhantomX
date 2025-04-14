import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from decimal import Decimal
import json
import os
import tempfile

from core.optimization import (
    optimize_performance,
    optimize_risk,
    optimize_reward,
    optimize_activity,
    tune_parameters,
    enhance_performance,
    validate_optimization,
    OptimizationError
)
from database.models import User, OptimizationResult
from database.exceptions import OptimizationError as DBOptimizationError

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
def test_parameters():
    """Create test parameters"""
    return {
        "mining": {
            "hashrate": 100,
            "power_consumption": 1000,
            "efficiency": 0.8,
            "difficulty": 0.5
        },
        "staking": {
            "amount": 1000,
            "duration": 30,
            "interest_rate": 0.05,
            "lock_period": 7
        },
        "trading": {
            "volume": 100,
            "frequency": 10,
            "strategy": "momentum",
            "risk_tolerance": 0.3
        }
    }

@pytest.fixture
def test_constraints():
    """Create test constraints"""
    return {
        "mining": {
            "max_power": 2000,
            "min_efficiency": 0.7,
            "max_difficulty": 0.8
        },
        "staking": {
            "min_amount": 100,
            "max_duration": 90,
            "min_interest_rate": 0.03
        },
        "trading": {
            "max_volume": 200,
            "min_frequency": 5,
            "max_risk_tolerance": 0.5
        }
    }

def test_optimize_performance(db_session, test_user, test_performance_data, test_parameters, test_constraints):
    """Test performance optimization"""
    # Optimize performance
    result = optimize_performance(
        user_id=test_user.user_id,
        data=test_performance_data,
        parameters=test_parameters,
        constraints=test_constraints,
        optimization_goal="maximize",
        db_session=db_session
    )
    
    # Verify optimization result
    assert isinstance(result, Dict)
    assert "optimized_parameters" in result
    assert "optimization_goal" in result
    
    # Verify optimization metadata
    assert result["optimization_goal"] == "maximize"
    
    # Verify optimized parameters
    assert isinstance(result["optimized_parameters"], Dict)
    assert "mining" in result["optimized_parameters"]
    assert "staking" in result["optimized_parameters"]
    assert "trading" in result["optimized_parameters"]
    
    # Verify database entry
    db_result = db_session.query(OptimizationResult).filter_by(
        user_id=test_user.user_id,
        optimization_type="PERFORMANCE_OPTIMIZATION"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_optimize_risk(db_session, test_user, test_risk_data, test_parameters, test_constraints):
    """Test risk optimization"""
    # Optimize risk
    result = optimize_risk(
        user_id=test_user.user_id,
        data=test_risk_data,
        parameters=test_parameters,
        constraints=test_constraints,
        optimization_goal="minimize",
        db_session=db_session
    )
    
    # Verify optimization result
    assert isinstance(result, Dict)
    assert "optimized_parameters" in result
    assert "optimization_goal" in result
    
    # Verify optimization metadata
    assert result["optimization_goal"] == "minimize"
    
    # Verify optimized parameters
    assert isinstance(result["optimized_parameters"], Dict)
    assert "mining" in result["optimized_parameters"]
    assert "staking" in result["optimized_parameters"]
    assert "trading" in result["optimized_parameters"]
    
    # Verify database entry
    db_result = db_session.query(OptimizationResult).filter_by(
        user_id=test_user.user_id,
        optimization_type="RISK_OPTIMIZATION"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_optimize_reward(db_session, test_user, test_reward_data, test_parameters, test_constraints):
    """Test reward optimization"""
    # Optimize reward
    result = optimize_reward(
        user_id=test_user.user_id,
        data=test_reward_data,
        parameters=test_parameters,
        constraints=test_constraints,
        optimization_goal="maximize",
        db_session=db_session
    )
    
    # Verify optimization result
    assert isinstance(result, Dict)
    assert "optimized_parameters" in result
    assert "optimization_goal" in result
    
    # Verify optimization metadata
    assert result["optimization_goal"] == "maximize"
    
    # Verify optimized parameters
    assert isinstance(result["optimized_parameters"], Dict)
    assert "mining" in result["optimized_parameters"]
    assert "staking" in result["optimized_parameters"]
    assert "trading" in result["optimized_parameters"]
    
    # Verify database entry
    db_result = db_session.query(OptimizationResult).filter_by(
        user_id=test_user.user_id,
        optimization_type="REWARD_OPTIMIZATION"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_optimize_activity(db_session, test_user, test_activity_data, test_parameters, test_constraints):
    """Test activity optimization"""
    # Optimize activity
    result = optimize_activity(
        user_id=test_user.user_id,
        data=test_activity_data,
        parameters=test_parameters,
        constraints=test_constraints,
        optimization_goal="maximize",
        db_session=db_session
    )
    
    # Verify optimization result
    assert isinstance(result, Dict)
    assert "optimized_parameters" in result
    assert "optimization_goal" in result
    
    # Verify optimization metadata
    assert result["optimization_goal"] == "maximize"
    
    # Verify optimized parameters
    assert isinstance(result["optimized_parameters"], Dict)
    assert "mining" in result["optimized_parameters"]
    assert "staking" in result["optimized_parameters"]
    assert "trading" in result["optimized_parameters"]
    
    # Verify database entry
    db_result = db_session.query(OptimizationResult).filter_by(
        user_id=test_user.user_id,
        optimization_type="ACTIVITY_OPTIMIZATION"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_tune_parameters(db_session, test_user, test_performance_data, test_parameters):
    """Test parameter tuning"""
    # Tune parameters
    result = tune_parameters(
        user_id=test_user.user_id,
        data=test_performance_data,
        parameters=test_parameters,
        tuning_method="grid_search",
        db_session=db_session
    )
    
    # Verify tuning result
    assert isinstance(result, Dict)
    assert "tuned_parameters" in result
    assert "tuning_method" in result
    
    # Verify tuning metadata
    assert result["tuning_method"] == "grid_search"
    
    # Verify tuned parameters
    assert isinstance(result["tuned_parameters"], Dict)
    assert "mining" in result["tuned_parameters"]
    assert "staking" in result["tuned_parameters"]
    assert "trading" in result["tuned_parameters"]
    
    # Verify database entry
    db_result = db_session.query(OptimizationResult).filter_by(
        user_id=test_user.user_id,
        optimization_type="PARAMETER_TUNING"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_enhance_performance(db_session, test_user, test_performance_data, test_parameters):
    """Test performance enhancement"""
    # Enhance performance
    result = enhance_performance(
        user_id=test_user.user_id,
        data=test_performance_data,
        parameters=test_parameters,
        enhancement_strategy="adaptive",
        db_session=db_session
    )
    
    # Verify enhancement result
    assert isinstance(result, Dict)
    assert "enhanced_parameters" in result
    assert "enhancement_strategy" in result
    
    # Verify enhancement metadata
    assert result["enhancement_strategy"] == "adaptive"
    
    # Verify enhanced parameters
    assert isinstance(result["enhanced_parameters"], Dict)
    assert "mining" in result["enhanced_parameters"]
    assert "staking" in result["enhanced_parameters"]
    assert "trading" in result["enhanced_parameters"]
    
    # Verify database entry
    db_result = db_session.query(OptimizationResult).filter_by(
        user_id=test_user.user_id,
        optimization_type="PERFORMANCE_ENHANCEMENT"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_validate_optimization(db_session, test_user, test_performance_data, test_parameters):
    """Test optimization validation"""
    # Validate optimization
    result = validate_optimization(
        user_id=test_user.user_id,
        data=test_performance_data,
        parameters=test_parameters,
        validation_metrics=["accuracy", "precision", "recall"],
        db_session=db_session
    )
    
    # Verify validation result
    assert isinstance(result, Dict)
    assert "validation" in result
    assert "validation_metrics" in result
    
    # Verify validation metadata
    assert result["validation_metrics"] == ["accuracy", "precision", "recall"]
    
    # Verify validation results
    assert isinstance(result["validation"], Dict)
    assert "accuracy" in result["validation"]
    assert "precision" in result["validation"]
    assert "recall" in result["validation"]
    
    # Verify database entry
    db_result = db_session.query(OptimizationResult).filter_by(
        user_id=test_user.user_id,
        optimization_type="OPTIMIZATION_VALIDATION"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_optimization_error_handling(db_session, test_user):
    """Test optimization error handling"""
    # Invalid user ID
    with pytest.raises(OptimizationError) as excinfo:
        optimize_performance(
            user_id=None,
            data=pd.DataFrame(),
            parameters={},
            constraints={},
            optimization_goal="maximize",
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(OptimizationError) as excinfo:
        optimize_risk(
            user_id=test_user.user_id,
            data=None,
            parameters={},
            constraints={},
            optimization_goal="minimize",
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Empty data
    with pytest.raises(OptimizationError) as excinfo:
        optimize_reward(
            user_id=test_user.user_id,
            data=pd.DataFrame(),
            parameters={},
            constraints={},
            optimization_goal="maximize",
            db_session=db_session
        )
    assert "Empty data" in str(excinfo.value)
    
    # Invalid optimization goal
    with pytest.raises(OptimizationError) as excinfo:
        optimize_activity(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            parameters={},
            constraints={},
            optimization_goal="invalid_goal",
            db_session=db_session
        )
    assert "Invalid optimization goal" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBOptimizationError) as excinfo:
        optimize_performance(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            parameters={},
            constraints={},
            optimization_goal="maximize",
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 