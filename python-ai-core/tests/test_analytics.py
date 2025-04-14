import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from decimal import Decimal
import json
import uuid
from unittest.mock import patch, MagicMock

from core.analytics import (
    analyze_mining_data,
    analyze_staking_data,
    analyze_trading_data,
    generate_performance_report,
    generate_risk_report,
    generate_reward_report,
    create_performance_chart,
    create_risk_chart,
    create_reward_chart,
    export_report_data,
    AnalyticsType,
    ChartType,
    ReportFormat,
    perform_statistical_analysis,
    build_predictive_model,
    detect_patterns,
    analyze_trends,
    generate_insights,
    get_analytics_info,
    list_models,
    delete_model,
    AnalyticsError
)
from database.models import User, Activity, Performance, Risk, Reward, AnalyticsRecord, ModelRecord, InsightRecord
from database.exceptions import AnalyticsError as DBAnalyticsError

@pytest.fixture
def test_mining_data() -> pd.DataFrame:
    """Create test mining data"""
    return pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="H"),
        "hash_rate": np.random.normal(95, 5, 100),
        "power_usage": np.random.normal(1450, 50, 100),
        "temperature": np.random.normal(75, 3, 100),
        "uptime": np.random.uniform(0.95, 1.0, 100),
        "efficiency": np.random.uniform(0.8, 0.9, 100),
        "block_rewards": np.random.exponential(0.5, 100),
        "network_difficulty": np.random.normal(45, 2, 100),
        "profitability": np.random.uniform(0.2, 0.3, 100)
    })

@pytest.fixture
def test_staking_data() -> pd.DataFrame:
    """Create test staking data"""
    return pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="H"),
        "validator_uptime": np.random.uniform(0.98, 1.0, 100),
        "missed_blocks": np.random.poisson(2, 100),
        "reward_rate": np.random.uniform(0.1, 0.15, 100),
        "peer_count": np.random.normal(50, 5, 100).astype(int),
        "network_participation": np.random.uniform(0.8, 0.9, 100),
        "slashing_events": np.random.binomial(1, 0.01, 100),
        "stake_amount": np.random.normal(1000, 100, 100),
        "validator_count": np.random.normal(100, 10, 100).astype(int)
    })

@pytest.fixture
def test_trading_data() -> pd.DataFrame:
    """Create test trading data"""
    return pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="H"),
        "position_size": np.random.normal(50000, 5000, 100),
        "leverage_ratio": np.random.uniform(1.5, 3.0, 100),
        "win_rate": np.random.uniform(0.5, 0.7, 100),
        "profit_loss": np.random.normal(500, 100, 100),
        "drawdown": np.random.uniform(0.1, 0.2, 100),
        "sharpe_ratio": np.random.normal(1.5, 0.3, 100),
        "volume": np.random.normal(100000, 10000, 100),
        "execution_quality": np.random.uniform(0.85, 0.95, 100)
    })

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
def test_analytics_config():
    """Create test analytics configuration"""
    return {
        "statistical_analysis": {
            "enabled": True,
            "descriptive_stats": True,
            "correlation_analysis": True,
            "distribution_analysis": True,
            "hypothesis_testing": True,
            "confidence_level": 0.95,
            "sample_size": 100
        },
        "predictive_modeling": {
            "enabled": True,
            "model_types": ["regression", "classification", "time_series"],
            "features": ["trading_performance", "risk_score", "transaction_amount"],
            "target": "status",
            "train_test_split": 0.8,
            "cross_validation": 5,
            "hyperparameter_tuning": True
        },
        "pattern_recognition": {
            "enabled": True,
            "pattern_types": ["seasonal", "cyclical", "trend", "anomaly"],
            "detection_methods": ["statistical", "machine_learning", "deep_learning"],
            "min_pattern_length": 10,
            "max_pattern_length": 100,
            "confidence_threshold": 0.8
        },
        "trend_analysis": {
            "enabled": True,
            "trend_types": ["linear", "exponential", "logarithmic", "polynomial"],
            "detection_methods": ["moving_average", "regression", "kalman_filter"],
            "window_size": 24,
            "min_trend_duration": 12,
            "significance_level": 0.05
        },
        "insight_generation": {
            "enabled": True,
            "insight_types": ["correlation", "causation", "anomaly", "prediction"],
            "visualization_types": ["line", "bar", "scatter", "heatmap"],
            "min_confidence": 0.7,
            "max_insights": 10,
            "format": "json"
        }
    }

def test_analyze_mining_data(test_mining_data: pd.DataFrame):
    """Test mining data analysis"""
    # Analyze data
    analysis_results = analyze_mining_data(test_mining_data)
    
    # Validate results
    assert isinstance(analysis_results, Dict)
    assert "summary_statistics" in analysis_results
    assert "trends" in analysis_results
    assert "correlations" in analysis_results
    assert "anomalies" in analysis_results
    
    # Verify summary statistics
    assert "hash_rate" in analysis_results["summary_statistics"]
    assert "power_usage" in analysis_results["summary_statistics"]
    assert "temperature" in analysis_results["summary_statistics"]
    assert "uptime" in analysis_results["summary_statistics"]
    assert "efficiency" in analysis_results["summary_statistics"]
    assert "block_rewards" in analysis_results["summary_statistics"]
    assert "network_difficulty" in analysis_results["summary_statistics"]
    assert "profitability" in analysis_results["summary_statistics"]
    
    # Verify trends
    assert "hash_rate_trend" in analysis_results["trends"]
    assert "power_usage_trend" in analysis_results["trends"]
    assert "temperature_trend" in analysis_results["trends"]
    assert "uptime_trend" in analysis_results["trends"]
    assert "efficiency_trend" in analysis_results["trends"]
    assert "block_rewards_trend" in analysis_results["trends"]
    assert "network_difficulty_trend" in analysis_results["trends"]
    assert "profitability_trend" in analysis_results["trends"]
    
    # Verify correlations
    assert "hash_rate_power_correlation" in analysis_results["correlations"]
    assert "hash_rate_temperature_correlation" in analysis_results["correlations"]
    assert "power_usage_temperature_correlation" in analysis_results["correlations"]
    assert "hash_rate_efficiency_correlation" in analysis_results["correlations"]
    assert "hash_rate_profitability_correlation" in analysis_results["correlations"]
    
    # Verify anomalies
    assert "hash_rate_anomalies" in analysis_results["anomalies"]
    assert "power_usage_anomalies" in analysis_results["anomalies"]
    assert "temperature_anomalies" in analysis_results["anomalies"]
    assert "uptime_anomalies" in analysis_results["anomalies"]
    assert "efficiency_anomalies" in analysis_results["anomalies"]
    assert "block_rewards_anomalies" in analysis_results["anomalies"]
    assert "network_difficulty_anomalies" in analysis_results["anomalies"]
    assert "profitability_anomalies" in analysis_results["anomalies"]

def test_analyze_staking_data(test_staking_data: pd.DataFrame):
    """Test staking data analysis"""
    # Analyze data
    analysis_results = analyze_staking_data(test_staking_data)
    
    # Validate results
    assert isinstance(analysis_results, Dict)
    assert "summary_statistics" in analysis_results
    assert "trends" in analysis_results
    assert "correlations" in analysis_results
    assert "anomalies" in analysis_results
    
    # Verify summary statistics
    assert "validator_uptime" in analysis_results["summary_statistics"]
    assert "missed_blocks" in analysis_results["summary_statistics"]
    assert "reward_rate" in analysis_results["summary_statistics"]
    assert "peer_count" in analysis_results["summary_statistics"]
    assert "network_participation" in analysis_results["summary_statistics"]
    assert "slashing_events" in analysis_results["summary_statistics"]
    assert "stake_amount" in analysis_results["summary_statistics"]
    assert "validator_count" in analysis_results["summary_statistics"]
    
    # Verify trends
    assert "validator_uptime_trend" in analysis_results["trends"]
    assert "missed_blocks_trend" in analysis_results["trends"]
    assert "reward_rate_trend" in analysis_results["trends"]
    assert "peer_count_trend" in analysis_results["trends"]
    assert "network_participation_trend" in analysis_results["trends"]
    assert "slashing_events_trend" in analysis_results["trends"]
    assert "stake_amount_trend" in analysis_results["trends"]
    assert "validator_count_trend" in analysis_results["trends"]
    
    # Verify correlations
    assert "uptime_reward_correlation" in analysis_results["correlations"]
    assert "uptime_missed_blocks_correlation" in analysis_results["correlations"]
    assert "peer_count_uptime_correlation" in analysis_results["correlations"]
    assert "stake_amount_reward_correlation" in analysis_results["correlations"]
    assert "network_participation_reward_correlation" in analysis_results["correlations"]
    
    # Verify anomalies
    assert "validator_uptime_anomalies" in analysis_results["anomalies"]
    assert "missed_blocks_anomalies" in analysis_results["anomalies"]
    assert "reward_rate_anomalies" in analysis_results["anomalies"]
    assert "peer_count_anomalies" in analysis_results["anomalies"]
    assert "network_participation_anomalies" in analysis_results["anomalies"]
    assert "slashing_events_anomalies" in analysis_results["anomalies"]
    assert "stake_amount_anomalies" in analysis_results["anomalies"]
    assert "validator_count_anomalies" in analysis_results["anomalies"]

def test_analyze_trading_data(test_trading_data: pd.DataFrame):
    """Test trading data analysis"""
    # Analyze data
    analysis_results = analyze_trading_data(test_trading_data)
    
    # Validate results
    assert isinstance(analysis_results, Dict)
    assert "summary_statistics" in analysis_results
    assert "trends" in analysis_results
    assert "correlations" in analysis_results
    assert "anomalies" in analysis_results
    
    # Verify summary statistics
    assert "position_size" in analysis_results["summary_statistics"]
    assert "leverage_ratio" in analysis_results["summary_statistics"]
    assert "win_rate" in analysis_results["summary_statistics"]
    assert "profit_loss" in analysis_results["summary_statistics"]
    assert "drawdown" in analysis_results["summary_statistics"]
    assert "sharpe_ratio" in analysis_results["summary_statistics"]
    assert "volume" in analysis_results["summary_statistics"]
    assert "execution_quality" in analysis_results["summary_statistics"]
    
    # Verify trends
    assert "position_size_trend" in analysis_results["trends"]
    assert "leverage_ratio_trend" in analysis_results["trends"]
    assert "win_rate_trend" in analysis_results["trends"]
    assert "profit_loss_trend" in analysis_results["trends"]
    assert "drawdown_trend" in analysis_results["trends"]
    assert "sharpe_ratio_trend" in analysis_results["trends"]
    assert "volume_trend" in analysis_results["trends"]
    assert "execution_quality_trend" in analysis_results["trends"]
    
    # Verify correlations
    assert "position_size_profit_correlation" in analysis_results["correlations"]
    assert "leverage_ratio_drawdown_correlation" in analysis_results["correlations"]
    assert "win_rate_profit_correlation" in analysis_results["correlations"]
    assert "volume_profit_correlation" in analysis_results["correlations"]
    assert "execution_quality_profit_correlation" in analysis_results["correlations"]
    
    # Verify anomalies
    assert "position_size_anomalies" in analysis_results["anomalies"]
    assert "leverage_ratio_anomalies" in analysis_results["anomalies"]
    assert "win_rate_anomalies" in analysis_results["anomalies"]
    assert "profit_loss_anomalies" in analysis_results["anomalies"]
    assert "drawdown_anomalies" in analysis_results["anomalies"]
    assert "sharpe_ratio_anomalies" in analysis_results["anomalies"]
    assert "volume_anomalies" in analysis_results["anomalies"]
    assert "execution_quality_anomalies" in analysis_results["anomalies"]

def test_generate_performance_report(db_session):
    """Test performance report generation"""
    # Create test user
    user = User(
        user_id="test_user",
        username="testuser",
        email="test@example.com"
    )
    db_session.add(user)
    
    # Create test activities with performance metrics
    activities = []
    performance_metrics = []
    
    # Mining activity
    mining_activity = Activity(
        activity_id="mining_activity",
        user_id="test_user",
        activity_type="mining",
        start_time=datetime.utcnow() - timedelta(hours=1),
        end_time=datetime.utcnow()
    )
    activities.append(mining_activity)
    
    mining_performance = Performance(
        performance_id="mining_performance",
        activity_id="mining_activity",
        performance_score=0.85,
        performance_level="HIGH",
        performance_factors=["HIGH_EFFICIENCY", "HIGH_UPTIME"],
        timestamp=datetime.utcnow(),
        type="MINING"
    )
    performance_metrics.append(mining_performance)
    
    # Staking activity
    staking_activity = Activity(
        activity_id="staking_activity",
        user_id="test_user",
        activity_type="staking",
        start_time=datetime.utcnow() - timedelta(hours=1),
        end_time=datetime.utcnow()
    )
    activities.append(staking_activity)
    
    staking_performance = Performance(
        performance_id="staking_performance",
        activity_id="staking_activity",
        performance_score=0.65,
        performance_level="MEDIUM",
        performance_factors=["HIGH_UPTIME"],
        timestamp=datetime.utcnow(),
        type="STAKING"
    )
    performance_metrics.append(staking_performance)
    
    # Add to database
    db_session.add_all(activities)
    db_session.add_all(performance_metrics)
    db_session.commit()
    
    # Generate report
    report = generate_performance_report(
        user_id="test_user",
        start_time=datetime.utcnow() - timedelta(days=1),
        end_time=datetime.utcnow(),
        report_format=ReportFormat.PDF,
        db_session=db_session
    )
    
    # Verify report
    assert isinstance(report, Dict)
    assert "report_id" in report
    assert "report_type" in report
    assert "report_format" in report
    assert "report_content" in report
    assert "report_metadata" in report
    
    # Verify report content
    assert "summary" in report["report_content"]
    assert "mining_performance" in report["report_content"]
    assert "staking_performance" in report["report_content"]
    assert "trading_performance" in report["report_content"]
    assert "recommendations" in report["report_content"]

def test_generate_risk_report(db_session):
    """Test risk report generation"""
    # Create test user
    user = User(
        user_id="test_user",
        username="testuser",
        email="test@example.com"
    )
    db_session.add(user)
    
    # Create test activities with risk assessments
    activities = []
    risk_assessments = []
    
    # Mining activity
    mining_activity = Activity(
        activity_id="mining_activity",
        user_id="test_user",
        activity_type="mining",
        start_time=datetime.utcnow() - timedelta(hours=1),
        end_time=datetime.utcnow()
    )
    activities.append(mining_activity)
    
    mining_risk = Risk(
        risk_id="mining_risk",
        activity_id="mining_activity",
        risk_score=0.75,
        risk_level="HIGH",
        risk_factors=["HIGH_TEMPERATURE", "HARDWARE_AGING"],
        timestamp=datetime.utcnow(),
        type="MINING"
    )
    risk_assessments.append(mining_risk)
    
    # Staking activity
    staking_activity = Activity(
        activity_id="staking_activity",
        user_id="test_user",
        activity_type="staking",
        start_time=datetime.utcnow() - timedelta(hours=1),
        end_time=datetime.utcnow()
    )
    activities.append(staking_activity)
    
    staking_risk = Risk(
        risk_id="staking_risk",
        activity_id="staking_activity",
        risk_score=0.3,
        risk_level="LOW",
        risk_factors=[],
        timestamp=datetime.utcnow(),
        type="STAKING"
    )
    risk_assessments.append(staking_risk)
    
    # Add to database
    db_session.add_all(activities)
    db_session.add_all(risk_assessments)
    db_session.commit()
    
    # Generate report
    report = generate_risk_report(
        user_id="test_user",
        start_time=datetime.utcnow() - timedelta(days=1),
        end_time=datetime.utcnow(),
        report_format=ReportFormat.PDF,
        db_session=db_session
    )
    
    # Verify report
    assert isinstance(report, Dict)
    assert "report_id" in report
    assert "report_type" in report
    assert "report_format" in report
    assert "report_content" in report
    assert "report_metadata" in report
    
    # Verify report content
    assert "summary" in report["report_content"]
    assert "mining_risk" in report["report_content"]
    assert "staking_risk" in report["report_content"]
    assert "trading_risk" in report["report_content"]
    assert "mitigation_steps" in report["report_content"]

def test_generate_reward_report(db_session):
    """Test reward report generation"""
    # Create test user
    user = User(
        user_id="test_user",
        username="testuser",
        email="test@example.com"
    )
    db_session.add(user)
    
    # Create test activities with rewards
    activities = []
    rewards = []
    
    # Mining activity
    mining_activity = Activity(
        activity_id="mining_activity",
        user_id="test_user",
        activity_type="mining",
        start_time=datetime.utcnow() - timedelta(hours=1),
        end_time=datetime.utcnow()
    )
    activities.append(mining_activity)
    
    mining_reward = Reward(
        reward_id="mining_reward",
        activity_id="mining_activity",
        reward_amount=0.5,
        reward_type="BLOCK_REWARD",
        timestamp=datetime.utcnow(),
        status="CONFIRMED"
    )
    rewards.append(mining_reward)
    
    # Staking activity
    staking_activity = Activity(
        activity_id="staking_activity",
        user_id="test_user",
        activity_type="staking",
        start_time=datetime.utcnow() - timedelta(hours=1),
        end_time=datetime.utcnow()
    )
    activities.append(staking_activity)
    
    staking_reward = Reward(
        reward_id="staking_reward",
        activity_id="staking_activity",
        reward_amount=0.1,
        reward_type="STAKING_REWARD",
        timestamp=datetime.utcnow(),
        status="CONFIRMED"
    )
    rewards.append(staking_reward)
    
    # Add to database
    db_session.add_all(activities)
    db_session.add_all(rewards)
    db_session.commit()
    
    # Generate report
    report = generate_reward_report(
        user_id="test_user",
        start_time=datetime.utcnow() - timedelta(days=1),
        end_time=datetime.utcnow(),
        report_format=ReportFormat.PDF,
        db_session=db_session
    )
    
    # Verify report
    assert isinstance(report, Dict)
    assert "report_id" in report
    assert "report_type" in report
    assert "report_format" in report
    assert "report_content" in report
    assert "report_metadata" in report
    
    # Verify report content
    assert "summary" in report["report_content"]
    assert "mining_rewards" in report["report_content"]
    assert "staking_rewards" in report["report_content"]
    assert "trading_rewards" in report["report_content"]
    assert "total_rewards" in report["report_content"]

def test_create_performance_chart(db_session):
    """Test performance chart creation"""
    # Create test performance metrics
    performance_metrics = [
        Performance(
            performance_id=f"performance_{i}",
            activity_id=f"activity_{i}",
            performance_score=0.2 * (i + 1),  # 0.2, 0.4, 0.6, 0.8, 1.0
            performance_level="LOW" if i < 2 else "MEDIUM" if i < 4 else "HIGH",
            performance_factors=["HIGH_WIN_RATE"] if i > 2 else [],
            timestamp=datetime.utcnow() - timedelta(days=i),
            type="TRADING"
        )
        for i in range(5)
    ]
    
    db_session.add_all(performance_metrics)
    db_session.commit()
    
    # Create chart
    chart = create_performance_chart(
        start_time=datetime.utcnow() - timedelta(days=5),
        end_time=datetime.utcnow(),
        chart_type=ChartType.LINE,
        db_session=db_session
    )
    
    # Verify chart
    assert isinstance(chart, Dict)
    assert "chart_id" in chart
    assert "chart_type" in chart
    assert "chart_data" in chart
    assert "chart_metadata" in chart
    
    # Verify chart data
    assert "labels" in chart["chart_data"]
    assert "datasets" in chart["chart_data"]
    assert len(chart["chart_data"]["datasets"]) > 0

def test_create_risk_chart(db_session):
    """Test risk chart creation"""
    # Create test risk assessments
    risk_assessments = [
        Risk(
            risk_id=f"risk_{i}",
            activity_id=f"activity_{i}",
            risk_score=0.2 * (i + 1),  # 0.2, 0.4, 0.6, 0.8, 1.0
            risk_level="LOW" if i < 2 else "MEDIUM" if i < 4 else "HIGH",
            risk_factors=["HIGH_LEVERAGE"] if i > 2 else [],
            timestamp=datetime.utcnow() - timedelta(days=i),
            type="TRADING"
        )
        for i in range(5)
    ]
    
    db_session.add_all(risk_assessments)
    db_session.commit()
    
    # Create chart
    chart = create_risk_chart(
        start_time=datetime.utcnow() - timedelta(days=5),
        end_time=datetime.utcnow(),
        chart_type=ChartType.LINE,
        db_session=db_session
    )
    
    # Verify chart
    assert isinstance(chart, Dict)
    assert "chart_id" in chart
    assert "chart_type" in chart
    assert "chart_data" in chart
    assert "chart_metadata" in chart
    
    # Verify chart data
    assert "labels" in chart["chart_data"]
    assert "datasets" in chart["chart_data"]
    assert len(chart["chart_data"]["datasets"]) > 0

def test_create_reward_chart(db_session):
    """Test reward chart creation"""
    # Create test rewards
    rewards = [
        Reward(
            reward_id=f"reward_{i}",
            activity_id=f"activity_{i}",
            reward_amount=0.1 * (i + 1),  # 0.1, 0.2, 0.3, 0.4, 0.5
            reward_type="BLOCK_REWARD" if i % 2 == 0 else "STAKING_REWARD",
            timestamp=datetime.utcnow() - timedelta(days=i),
            status="CONFIRMED"
        )
        for i in range(5)
    ]
    
    db_session.add_all(rewards)
    db_session.commit()
    
    # Create chart
    chart = create_reward_chart(
        start_time=datetime.utcnow() - timedelta(days=5),
        end_time=datetime.utcnow(),
        chart_type=ChartType.BAR,
        db_session=db_session
    )
    
    # Verify chart
    assert isinstance(chart, Dict)
    assert "chart_id" in chart
    assert "chart_type" in chart
    assert "chart_data" in chart
    assert "chart_metadata" in chart
    
    # Verify chart data
    assert "labels" in chart["chart_data"]
    assert "datasets" in chart["chart_data"]
    assert len(chart["chart_data"]["datasets"]) > 0

def test_export_report_data(db_session):
    """Test report data export"""
    # Create test performance metrics
    performance_metrics = [
        Performance(
            performance_id=f"performance_{i}",
            activity_id=f"activity_{i}",
            performance_score=0.2 * (i + 1),  # 0.2, 0.4, 0.6, 0.8, 1.0
            performance_level="LOW" if i < 2 else "MEDIUM" if i < 4 else "HIGH",
            performance_factors=["HIGH_WIN_RATE"] if i > 2 else [],
            timestamp=datetime.utcnow() - timedelta(days=i),
            type="TRADING"
        )
        for i in range(5)
    ]
    
    db_session.add_all(performance_metrics)
    db_session.commit()
    
    # Export data
    export_data = export_report_data(
        start_time=datetime.utcnow() - timedelta(days=5),
        end_time=datetime.utcnow(),
        analytics_type=AnalyticsType.PERFORMANCE,
        db_session=db_session
    )
    
    # Verify export data
    assert isinstance(export_data, Dict)
    assert "data" in export_data
    assert "metadata" in export_data
    
    # Verify data
    assert isinstance(export_data["data"], pd.DataFrame)
    assert len(export_data["data"]) > 0
    assert "performance_id" in export_data["data"].columns
    assert "activity_id" in export_data["data"].columns
    assert "performance_score" in export_data["data"].columns
    assert "performance_level" in export_data["data"].columns
    assert "performance_factors" in export_data["data"].columns
    assert "timestamp" in export_data["data"].columns
    assert "type" in export_data["data"].columns

def test_perform_statistical_analysis(db_session, test_user, test_data, test_analytics_config):
    """Test statistical analysis"""
    # Perform statistical analysis
    result = perform_statistical_analysis(
        data=test_data,
        analysis_config=test_analytics_config["statistical_analysis"],
        user_id=test_user.user_id,
        message="Perform statistical analysis",
        db_session=db_session
    )
    
    # Verify analysis result
    assert isinstance(result, Dict)
    assert "analysis_id" in result
    assert "timestamp" in result
    assert "statistics" in result
    
    # Verify statistics
    statistics = result["statistics"]
    assert "descriptive" in statistics
    assert "correlation" in statistics
    assert "distribution" in statistics
    assert "hypothesis" in statistics
    
    # Verify descriptive statistics
    descriptive = statistics["descriptive"]
    assert isinstance(descriptive, Dict)
    for column in test_data.select_dtypes(include=[np.number]).columns:
        assert column in descriptive
        assert "mean" in descriptive[column]
        assert "median" in descriptive[column]
        assert "std" in descriptive[column]
        assert "min" in descriptive[column]
        assert "max" in descriptive[column]
    
    # Verify analytics record
    analytics_record = db_session.query(AnalyticsRecord).filter_by(
        analysis_id=result["analysis_id"]
    ).first()
    assert analytics_record is not None
    assert analytics_record.status == "ANALYZED"
    assert analytics_record.error is None

def test_build_predictive_model(db_session, test_user, test_data, test_analytics_config):
    """Test predictive modeling"""
    # Build predictive model
    result = build_predictive_model(
        data=test_data,
        model_config=test_analytics_config["predictive_modeling"],
        user_id=test_user.user_id,
        message="Build predictive model",
        db_session=db_session
    )
    
    # Verify model result
    assert isinstance(result, Dict)
    assert "model_id" in result
    assert "timestamp" in result
    assert "model" in result
    
    # Verify model details
    model = result["model"]
    assert "type" in model
    assert "features" in model
    assert "target" in model
    assert "performance" in model
    assert "parameters" in model
    
    # Verify model performance
    performance = model["performance"]
    assert "accuracy" in performance
    assert "precision" in performance
    assert "recall" in performance
    assert "f1_score" in performance
    
    # Verify model record
    model_record = db_session.query(ModelRecord).filter_by(
        model_id=result["model_id"]
    ).first()
    assert model_record is not None
    assert model_record.status == "BUILT"
    assert model_record.error is None

def test_detect_patterns(db_session, test_user, test_data, test_analytics_config):
    """Test pattern detection"""
    # Detect patterns
    result = detect_patterns(
        data=test_data,
        pattern_config=test_analytics_config["pattern_recognition"],
        user_id=test_user.user_id,
        message="Detect patterns",
        db_session=db_session
    )
    
    # Verify pattern detection result
    assert isinstance(result, Dict)
    assert "pattern_id" in result
    assert "timestamp" in result
    assert "patterns" in result
    
    # Verify patterns
    patterns = result["patterns"]
    assert isinstance(patterns, List)
    assert len(patterns) > 0
    
    # Verify pattern details
    for pattern in patterns:
        assert "type" in pattern
        assert "start_time" in pattern
        assert "end_time" in pattern
        assert "confidence" in pattern
        assert "description" in pattern
        assert pattern["type"] in test_analytics_config["pattern_recognition"]["pattern_types"]
        assert pattern["confidence"] >= test_analytics_config["pattern_recognition"]["confidence_threshold"]
    
    # Verify analytics record
    analytics_record = db_session.query(AnalyticsRecord).filter_by(
        pattern_id=result["pattern_id"]
    ).first()
    assert analytics_record is not None
    assert analytics_record.status == "DETECTED"
    assert analytics_record.error is None

def test_analyze_trends(db_session, test_user, test_data, test_analytics_config):
    """Test trend analysis"""
    # Analyze trends
    result = analyze_trends(
        data=test_data,
        trend_config=test_analytics_config["trend_analysis"],
        user_id=test_user.user_id,
        message="Analyze trends",
        db_session=db_session
    )
    
    # Verify trend analysis result
    assert isinstance(result, Dict)
    assert "trend_id" in result
    assert "timestamp" in result
    assert "trends" in result
    
    # Verify trends
    trends = result["trends"]
    assert isinstance(trends, List)
    assert len(trends) > 0
    
    # Verify trend details
    for trend in trends:
        assert "type" in trend
        assert "start_time" in trend
        assert "end_time" in trend
        assert "direction" in trend
        assert "magnitude" in trend
        assert "significance" in trend
        assert trend["type"] in test_analytics_config["trend_analysis"]["trend_types"]
        assert trend["significance"] <= test_analytics_config["trend_analysis"]["significance_level"]
    
    # Verify analytics record
    analytics_record = db_session.query(AnalyticsRecord).filter_by(
        trend_id=result["trend_id"]
    ).first()
    assert analytics_record is not None
    assert analytics_record.status == "ANALYZED"
    assert analytics_record.error is None

def test_generate_insights(db_session, test_user, test_data, test_analytics_config):
    """Test insight generation"""
    # Generate insights
    result = generate_insights(
        data=test_data,
        insight_config=test_analytics_config["insight_generation"],
        user_id=test_user.user_id,
        message="Generate insights",
        db_session=db_session
    )
    
    # Verify insight generation result
    assert isinstance(result, Dict)
    assert "insight_id" in result
    assert "timestamp" in result
    assert "insights" in result
    
    # Verify insights
    insights = result["insights"]
    assert isinstance(insights, List)
    assert len(insights) <= test_analytics_config["insight_generation"]["max_insights"]
    
    # Verify insight details
    for insight in insights:
        assert "type" in insight
        assert "description" in insight
        assert "confidence" in insight
        assert "visualization" in insight
        assert insight["type"] in test_analytics_config["insight_generation"]["insight_types"]
        assert insight["confidence"] >= test_analytics_config["insight_generation"]["min_confidence"]
    
    # Verify insight record
    insight_record = db_session.query(InsightRecord).filter_by(
        insight_id=result["insight_id"]
    ).first()
    assert insight_record is not None
    assert insight_record.status == "GENERATED"
    assert insight_record.error is None

def test_get_analytics_info(db_session, test_user, test_data, test_analytics_config):
    """Test analytics information retrieval"""
    # Perform statistical analysis
    analysis = perform_statistical_analysis(
        data=test_data,
        analysis_config=test_analytics_config["statistical_analysis"],
        user_id=test_user.user_id,
        message="Test analysis",
        db_session=db_session
    )
    
    # Get analytics info
    result = get_analytics_info(
        analysis_id=analysis["analysis_id"],
        db_session=db_session
    )
    
    # Verify analytics info result
    assert isinstance(result, Dict)
    assert "analysis_id" in result
    assert "timestamp" in result
    assert "info" in result
    
    # Verify info content
    info = result["info"]
    assert "statistics" in info
    assert "models" in info
    assert "patterns" in info
    assert "trends" in info
    assert "insights" in info
    
    # Verify statistics details
    statistics = info["statistics"]
    assert "descriptive" in statistics
    assert "correlation" in statistics
    assert "distribution" in statistics
    assert "hypothesis" in statistics
    
    # Verify analytics record
    analytics_record = db_session.query(AnalyticsRecord).filter_by(
        analysis_id=result["analysis_id"]
    ).first()
    assert analytics_record is not None
    assert analytics_record.status == "RETRIEVED"
    assert analytics_record.error is None

def test_list_models(db_session, test_user, test_data, test_analytics_config):
    """Test model listing"""
    # Build multiple models
    for i in range(5):
        build_predictive_model(
            data=test_data,
            model_config=test_analytics_config["predictive_modeling"],
            user_id=test_user.user_id,
            message=f"Model {i+1}",
            db_session=db_session
        )
    
    # List models
    result = list_models(
        data_id="test_data",
        db_session=db_session
    )
    
    # Verify model listing result
    assert isinstance(result, Dict)
    assert "timestamp" in result
    assert "models" in result
    
    # Verify models list
    models = result["models"]
    assert isinstance(models, List)
    assert len(models) == 5
    
    # Verify model details
    for model in models:
        assert "model_id" in model
        assert "timestamp" in model
        assert "type" in model
        assert "performance" in model
        assert "user_id" in model

def test_delete_model(db_session, test_user, test_data, test_analytics_config):
    """Test model deletion"""
    # Build model
    model = build_predictive_model(
        data=test_data,
        model_config=test_analytics_config["predictive_modeling"],
        user_id=test_user.user_id,
        message="Test model",
        db_session=db_session
    )
    
    # Delete model
    result = delete_model(
        model_id=model["model_id"],
        user_id=test_user.user_id,
        message="Delete test model",
        db_session=db_session
    )
    
    # Verify deletion result
    assert isinstance(result, Dict)
    assert "deletion_id" in result
    assert "timestamp" in result
    assert "status" in result
    
    # Verify status
    assert result["status"] == "DELETED"
    
    # Verify model record
    model_record = db_session.query(ModelRecord).filter_by(
        model_id=model["model_id"]
    ).first()
    assert model_record is not None
    assert model_record.status == "DELETED"
    assert model_record.error is None

def test_analytics_error_handling(db_session, test_user):
    """Test analytics error handling"""
    # Invalid statistical analysis configuration
    with pytest.raises(AnalyticsError) as excinfo:
        perform_statistical_analysis(
            data=pd.DataFrame(),
            analysis_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid statistical analysis configuration" in str(excinfo.value)
    
    # Invalid predictive modeling configuration
    with pytest.raises(AnalyticsError) as excinfo:
        build_predictive_model(
            data=pd.DataFrame(),
            model_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid predictive modeling configuration" in str(excinfo.value)
    
    # Invalid pattern recognition configuration
    with pytest.raises(AnalyticsError) as excinfo:
        detect_patterns(
            data=pd.DataFrame(),
            pattern_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid pattern recognition configuration" in str(excinfo.value)
    
    # Invalid trend analysis configuration
    with pytest.raises(AnalyticsError) as excinfo:
        analyze_trends(
            data=pd.DataFrame(),
            trend_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid trend analysis configuration" in str(excinfo.value)
    
    # Invalid insight generation configuration
    with pytest.raises(AnalyticsError) as excinfo:
        generate_insights(
            data=pd.DataFrame(),
            insight_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid insight generation configuration" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBAnalyticsError) as excinfo:
        perform_statistical_analysis(
            data=pd.DataFrame(),
            analysis_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 