import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, Any, List
from decimal import Decimal
import io
import base64
import json
import os
import tempfile
import seaborn as sns
import uuid
from unittest.mock import patch, MagicMock

from core.visualization import (
    create_performance_chart,
    create_risk_chart,
    create_reward_chart,
    create_activity_chart,
    create_correlation_heatmap,
    create_trend_chart,
    create_distribution_chart,
    create_anomaly_chart,
    format_chart,
    export_chart,
    VisualizationError,
    generate_performance_chart,
    generate_risk_chart,
    generate_reward_chart,
    generate_activity_chart,
    generate_analytics_chart,
    customize_chart,
    generate_visualization,
    get_visualization_info,
    create_chart,
    create_dashboard,
    create_interactive_visualization,
    create_custom_plot,
    export_visualization,
    list_visualizations,
    delete_visualization
)
from database.models import User, VisualizationResult, Chart, ChartExport, VisualizationRecord, DashboardRecord, ChartRecord
from database.exceptions import VisualizationError as DBVisualizationError

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
def test_correlation_data():
    """Create test correlation data"""
    return pd.DataFrame({
        "mining_performance": np.random.uniform(0.8, 0.9, 100),
        "staking_performance": np.random.uniform(0.85, 0.95, 100),
        "trading_performance": np.random.uniform(0.7, 0.8, 100),
        "mining_risk": np.random.uniform(0.2, 0.3, 100),
        "staking_risk": np.random.uniform(0.1, 0.2, 100),
        "trading_risk": np.random.uniform(0.3, 0.4, 100),
        "mining_rewards": np.random.uniform(0.4, 0.6, 100),
        "staking_rewards": np.random.uniform(0.1, 0.15, 100),
        "trading_rewards": np.random.uniform(0.05, 0.1, 100)
    })

@pytest.fixture
def test_trend_data():
    """Create test trend data"""
    return pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="H"),
        "value": np.linspace(0, 1, 100) + np.random.normal(0, 0.05, 100)
    })

@pytest.fixture
def test_distribution_data():
    """Create test distribution data"""
    return pd.DataFrame({
        "mining_performance": np.random.normal(0.85, 0.05, 1000),
        "staking_performance": np.random.normal(0.9, 0.03, 1000),
        "trading_performance": np.random.normal(0.75, 0.07, 1000)
    })

@pytest.fixture
def test_anomaly_data():
    """Create test anomaly data"""
    data = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="H"),
        "value": np.random.normal(0.5, 0.1, 100)
    })
    # Add some anomalies
    data.loc[20:25, "value"] = np.random.normal(1.5, 0.1, 6)
    data.loc[50:55, "value"] = np.random.normal(-0.5, 0.1, 6)
    return data

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
def test_chart_style():
    """Create test chart style"""
    return {
        "type": "line",
        "theme": "dark",
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        "font": {
            "family": "Arial",
            "size": 12,
            "color": "#FFFFFF"
        },
        "background": "#121212",
        "grid": {
            "show": True,
            "color": "#333333",
            "style": "dashed"
        },
        "legend": {
            "show": True,
            "position": "top",
            "orientation": "horizontal"
        },
        "title": {
            "show": True,
            "text": "Test Chart",
            "font_size": 16
        },
        "axes": {
            "x_label": "Time",
            "y_label": "Value",
            "show_ticks": True
        }
    }

@pytest.fixture
def test_export_config():
    """Create test export configuration"""
    return {
        "format": "png",
        "resolution": "high",
        "width": 1200,
        "height": 800,
        "dpi": 300,
        "transparent": False,
        "metadata": {
            "title": "Test Chart",
            "author": "Test User",
            "description": "Test chart export",
            "date": datetime.utcnow().isoformat()
        }
    }

@pytest.fixture
def test_visualization_config():
    """Create test visualization configuration"""
    return {
        "chart_generation": {
            "enabled": True,
            "chart_types": ["line", "bar", "scatter", "pie", "heatmap"],
            "color_schemes": ["default", "dark", "light", "custom"],
            "interactive": True,
            "responsive": True,
            "animation": True,
            "tooltips": True,
            "legends": True
        },
        "dashboard_creation": {
            "enabled": True,
            "layout_types": ["grid", "flex", "free"],
            "max_widgets": 12,
            "refresh_rate": 60,
            "auto_save": True,
            "sharing_enabled": True,
            "export_formats": ["png", "pdf", "html"]
        },
        "interactive_features": {
            "enabled": True,
            "zoom": True,
            "pan": True,
            "filter": True,
            "drill_down": True,
            "cross_filter": True,
            "annotations": True,
            "custom_controls": True
        },
        "custom_plots": {
            "enabled": True,
            "plot_types": ["candlestick", "box", "violin", "histogram", "kde"],
            "custom_styles": True,
            "advanced_options": True,
            "templates": ["default", "minimal", "dark", "light"]
        },
        "export_options": {
            "enabled": True,
            "formats": ["png", "pdf", "svg", "html", "json"],
            "resolution": "high",
            "include_metadata": True,
            "compression": True,
            "watermark": False
        }
    }

def test_create_performance_chart(db_session, test_user, test_performance_data):
    """Test performance chart creation"""
    # Create performance chart
    result = create_performance_chart(
        user_id=test_user.user_id,
        data=test_performance_data,
        chart_type="line",
        title="Performance Metrics",
        x_label="Time",
        y_label="Performance Score",
        db_session=db_session
    )
    
    # Verify chart creation result
    assert isinstance(result, Dict)
    assert "chart_data" in result
    assert "chart_type" in result
    assert "title" in result
    assert "x_label" in result
    assert "y_label" in result
    
    # Verify chart data
    assert isinstance(result["chart_data"], str)
    assert result["chart_data"].startswith("data:image/png;base64,")
    
    # Verify chart metadata
    assert result["chart_type"] == "line"
    assert result["title"] == "Performance Metrics"
    assert result["x_label"] == "Time"
    assert result["y_label"] == "Performance Score"
    
    # Verify database entry
    db_result = db_session.query(VisualizationResult).filter_by(
        user_id=test_user.user_id,
        visualization_type="PERFORMANCE_CHART"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_create_risk_chart(db_session, test_user, test_risk_data):
    """Test risk chart creation"""
    # Create risk chart
    result = create_risk_chart(
        user_id=test_user.user_id,
        data=test_risk_data,
        chart_type="area",
        title="Risk Metrics",
        x_label="Time",
        y_label="Risk Score",
        db_session=db_session
    )
    
    # Verify chart creation result
    assert isinstance(result, Dict)
    assert "chart_data" in result
    assert "chart_type" in result
    assert "title" in result
    assert "x_label" in result
    assert "y_label" in result
    
    # Verify chart data
    assert isinstance(result["chart_data"], str)
    assert result["chart_data"].startswith("data:image/png;base64,")
    
    # Verify chart metadata
    assert result["chart_type"] == "area"
    assert result["title"] == "Risk Metrics"
    assert result["x_label"] == "Time"
    assert result["y_label"] == "Risk Score"
    
    # Verify database entry
    db_result = db_session.query(VisualizationResult).filter_by(
        user_id=test_user.user_id,
        visualization_type="RISK_CHART"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_create_reward_chart(db_session, test_user, test_reward_data):
    """Test reward chart creation"""
    # Create reward chart
    result = create_reward_chart(
        user_id=test_user.user_id,
        data=test_reward_data,
        chart_type="bar",
        title="Reward Metrics",
        x_label="Time",
        y_label="Reward Amount",
        db_session=db_session
    )
    
    # Verify chart creation result
    assert isinstance(result, Dict)
    assert "chart_data" in result
    assert "chart_type" in result
    assert "title" in result
    assert "x_label" in result
    assert "y_label" in result
    
    # Verify chart data
    assert isinstance(result["chart_data"], str)
    assert result["chart_data"].startswith("data:image/png;base64,")
    
    # Verify chart metadata
    assert result["chart_type"] == "bar"
    assert result["title"] == "Reward Metrics"
    assert result["x_label"] == "Time"
    assert result["y_label"] == "Reward Amount"
    
    # Verify database entry
    db_result = db_session.query(VisualizationResult).filter_by(
        user_id=test_user.user_id,
        visualization_type="REWARD_CHART"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_create_activity_chart(db_session, test_user, test_activity_data):
    """Test activity chart creation"""
    # Create activity chart
    result = create_activity_chart(
        user_id=test_user.user_id,
        data=test_activity_data,
        chart_type="line",
        title="Activity Metrics",
        x_label="Time",
        y_label="Activity Level",
        db_session=db_session
    )
    
    # Verify chart creation result
    assert isinstance(result, Dict)
    assert "chart_data" in result
    assert "chart_type" in result
    assert "title" in result
    assert "x_label" in result
    assert "y_label" in result
    
    # Verify chart data
    assert isinstance(result["chart_data"], str)
    assert result["chart_data"].startswith("data:image/png;base64,")
    
    # Verify chart metadata
    assert result["chart_type"] == "line"
    assert result["title"] == "Activity Metrics"
    assert result["x_label"] == "Time"
    assert result["y_label"] == "Activity Level"
    
    # Verify database entry
    db_result = db_session.query(VisualizationResult).filter_by(
        user_id=test_user.user_id,
        visualization_type="ACTIVITY_CHART"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_create_correlation_heatmap(db_session, test_user, test_correlation_data):
    """Test correlation heatmap creation"""
    # Create correlation heatmap
    result = create_correlation_heatmap(
        user_id=test_user.user_id,
        data=test_correlation_data,
        title="Correlation Heatmap",
        db_session=db_session
    )
    
    # Verify heatmap creation result
    assert isinstance(result, Dict)
    assert "chart_data" in result
    assert "chart_type" in result
    assert "title" in result
    
    # Verify chart data
    assert isinstance(result["chart_data"], str)
    assert result["chart_data"].startswith("data:image/png;base64,")
    
    # Verify chart metadata
    assert result["chart_type"] == "heatmap"
    assert result["title"] == "Correlation Heatmap"
    
    # Verify database entry
    db_result = db_session.query(VisualizationResult).filter_by(
        user_id=test_user.user_id,
        visualization_type="CORRELATION_HEATMAP"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_create_trend_chart(db_session, test_user, test_trend_data):
    """Test trend chart creation"""
    # Create trend chart
    result = create_trend_chart(
        user_id=test_user.user_id,
        data=test_trend_data,
        value_column="value",
        chart_type="line",
        title="Trend Analysis",
        x_label="Time",
        y_label="Value",
        db_session=db_session
    )
    
    # Verify trend chart creation result
    assert isinstance(result, Dict)
    assert "chart_data" in result
    assert "chart_type" in result
    assert "title" in result
    assert "x_label" in result
    assert "y_label" in result
    
    # Verify chart data
    assert isinstance(result["chart_data"], str)
    assert result["chart_data"].startswith("data:image/png;base64,")
    
    # Verify chart metadata
    assert result["chart_type"] == "line"
    assert result["title"] == "Trend Analysis"
    assert result["x_label"] == "Time"
    assert result["y_label"] == "Value"
    
    # Verify database entry
    db_result = db_session.query(VisualizationResult).filter_by(
        user_id=test_user.user_id,
        visualization_type="TREND_CHART"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_create_distribution_chart(db_session, test_user, test_distribution_data):
    """Test distribution chart creation"""
    # Create distribution chart
    result = create_distribution_chart(
        user_id=test_user.user_id,
        data=test_distribution_data,
        chart_type="histogram",
        title="Performance Distribution",
        x_label="Performance Score",
        y_label="Frequency",
        db_session=db_session
    )
    
    # Verify distribution chart creation result
    assert isinstance(result, Dict)
    assert "chart_data" in result
    assert "chart_type" in result
    assert "title" in result
    assert "x_label" in result
    assert "y_label" in result
    
    # Verify chart data
    assert isinstance(result["chart_data"], str)
    assert result["chart_data"].startswith("data:image/png;base64,")
    
    # Verify chart metadata
    assert result["chart_type"] == "histogram"
    assert result["title"] == "Performance Distribution"
    assert result["x_label"] == "Performance Score"
    assert result["y_label"] == "Frequency"
    
    # Verify database entry
    db_result = db_session.query(VisualizationResult).filter_by(
        user_id=test_user.user_id,
        visualization_type="DISTRIBUTION_CHART"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_create_anomaly_chart(db_session, test_user, test_anomaly_data):
    """Test anomaly chart creation"""
    # Create anomaly chart
    result = create_anomaly_chart(
        user_id=test_user.user_id,
        data=test_anomaly_data,
        value_column="value",
        anomaly_indices=[20, 21, 22, 23, 24, 25, 50, 51, 52, 53, 54, 55],
        chart_type="scatter",
        title="Anomaly Detection",
        x_label="Time",
        y_label="Value",
        db_session=db_session
    )
    
    # Verify anomaly chart creation result
    assert isinstance(result, Dict)
    assert "chart_data" in result
    assert "chart_type" in result
    assert "title" in result
    assert "x_label" in result
    assert "y_label" in result
    
    # Verify chart data
    assert isinstance(result["chart_data"], str)
    assert result["chart_data"].startswith("data:image/png;base64,")
    
    # Verify chart metadata
    assert result["chart_type"] == "scatter"
    assert result["title"] == "Anomaly Detection"
    assert result["x_label"] == "Time"
    assert result["y_label"] == "Value"
    
    # Verify database entry
    db_result = db_session.query(VisualizationResult).filter_by(
        user_id=test_user.user_id,
        visualization_type="ANOMALY_CHART"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None

def test_format_chart(db_session, test_user, test_performance_data):
    """Test chart formatting"""
    # Create a basic chart
    plt.figure(figsize=(10, 6))
    plt.plot(test_performance_data["timestamp"], test_performance_data["mining_performance"])
    plt.title("Test Chart")
    plt.xlabel("Time")
    plt.ylabel("Value")
    
    # Format the chart
    result = format_chart(
        user_id=test_user.user_id,
        chart=plt.gcf(),
        format_type="png",
        dpi=300,
        db_session=db_session
    )
    
    # Verify formatting result
    assert isinstance(result, Dict)
    assert "chart_data" in result
    assert "format_type" in result
    assert "dpi" in result
    
    # Verify chart data
    assert isinstance(result["chart_data"], str)
    assert result["chart_data"].startswith("data:image/png;base64,")
    
    # Verify format metadata
    assert result["format_type"] == "png"
    assert result["dpi"] == 300
    
    # Verify database entry
    db_result = db_session.query(VisualizationResult).filter_by(
        user_id=test_user.user_id,
        visualization_type="CHART_FORMATTING"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None
    
    # Clean up
    plt.close()

def test_export_chart(db_session, test_user, test_performance_data):
    """Test chart export"""
    # Create a basic chart
    plt.figure(figsize=(10, 6))
    plt.plot(test_performance_data["timestamp"], test_performance_data["mining_performance"])
    plt.title("Test Chart")
    plt.xlabel("Time")
    plt.ylabel("Value")
    
    # Export the chart
    result = export_chart(
        user_id=test_user.user_id,
        chart=plt.gcf(),
        export_format="png",
        filename="test_chart",
        db_session=db_session
    )
    
    # Verify export result
    assert isinstance(result, Dict)
    assert "file_path" in result
    assert "export_format" in result
    assert "filename" in result
    
    # Verify export metadata
    assert result["export_format"] == "png"
    assert result["filename"] == "test_chart"
    assert result["file_path"].endswith(".png")
    
    # Verify database entry
    db_result = db_session.query(VisualizationResult).filter_by(
        user_id=test_user.user_id,
        visualization_type="CHART_EXPORT"
    ).first()
    assert db_result is not None
    assert db_result.is_success is True
    assert db_result.error is None
    
    # Clean up
    plt.close()

def test_visualization_error_handling():
    """Test visualization error handling"""
    # Invalid user ID
    with pytest.raises(VisualizationError) as excinfo:
        create_performance_chart(
            user_id=None,
            data=pd.DataFrame(),
            chart_type="line",
            title="Test Chart",
            x_label="Time",
            y_label="Value",
            db_session=None
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data type
    with pytest.raises(VisualizationError) as excinfo:
        create_risk_chart(
            user_id="test_user",
            data="not a DataFrame",
            chart_type="area",
            title="Test Chart",
            x_label="Time",
            y_label="Value",
            db_session=None
        )
    assert "Invalid data type" in str(excinfo.value)
    
    # Empty data
    with pytest.raises(VisualizationError) as excinfo:
        create_reward_chart(
            user_id="test_user",
            data=pd.DataFrame(),
            chart_type="bar",
            title="Test Chart",
            x_label="Time",
            y_label="Value",
            db_session=None
        )
    assert "Empty data" in str(excinfo.value)
    
    # Invalid chart type
    with pytest.raises(VisualizationError) as excinfo:
        create_activity_chart(
            user_id="test_user",
            data=pd.DataFrame({"column1": [1, 2, 3]}),
            chart_type="invalid_type",
            title="Test Chart",
            x_label="Time",
            y_label="Value",
            db_session=None
        )
    assert "Invalid chart type" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBVisualizationError) as excinfo:
        create_correlation_heatmap(
            user_id="test_user",
            data=pd.DataFrame(),
            title="Test Chart",
            db_session=None
        )
    assert "Database error" in str(excinfo.value)

def test_generate_performance_chart(db_session, test_user, test_performance_data, test_chart_style):
    """Test performance chart generation"""
    # Generate performance chart
    result = generate_performance_chart(
        user_id=test_user.user_id,
        data=test_performance_data,
        chart_style=test_chart_style,
        db_session=db_session
    )
    
    # Verify chart result
    assert isinstance(result, Dict)
    assert "chart_id" in result
    assert "chart_type" in result
    
    # Verify chart metadata
    assert result["chart_type"] == "PERFORMANCE_CHART"
    assert result["chart_style"]["type"] == "line"
    
    # Verify chart content
    assert "chart_data" in result
    assert isinstance(result["chart_data"], bytes)
    assert len(result["chart_data"]) > 0
    
    # Verify database entry
    db_chart = db_session.query(Chart).filter_by(
        user_id=test_user.user_id,
        chart_type="PERFORMANCE_CHART"
    ).first()
    assert db_chart is not None
    assert db_chart.is_success is True
    assert db_chart.error is None

def test_generate_risk_chart(db_session, test_user, test_risk_data, test_chart_style):
    """Test risk chart generation"""
    # Generate risk chart
    result = generate_risk_chart(
        user_id=test_user.user_id,
        data=test_risk_data,
        chart_style=test_chart_style,
        db_session=db_session
    )
    
    # Verify chart result
    assert isinstance(result, Dict)
    assert "chart_id" in result
    assert "chart_type" in result
    
    # Verify chart metadata
    assert result["chart_type"] == "RISK_CHART"
    assert result["chart_style"]["type"] == "line"
    
    # Verify chart content
    assert "chart_data" in result
    assert isinstance(result["chart_data"], bytes)
    assert len(result["chart_data"]) > 0
    
    # Verify database entry
    db_chart = db_session.query(Chart).filter_by(
        user_id=test_user.user_id,
        chart_type="RISK_CHART"
    ).first()
    assert db_chart is not None
    assert db_chart.is_success is True
    assert db_chart.error is None

def test_generate_reward_chart(db_session, test_user, test_reward_data, test_chart_style):
    """Test reward chart generation"""
    # Generate reward chart
    result = generate_reward_chart(
        user_id=test_user.user_id,
        data=test_reward_data,
        chart_style=test_chart_style,
        db_session=db_session
    )
    
    # Verify chart result
    assert isinstance(result, Dict)
    assert "chart_id" in result
    assert "chart_type" in result
    
    # Verify chart metadata
    assert result["chart_type"] == "REWARD_CHART"
    assert result["chart_style"]["type"] == "line"
    
    # Verify chart content
    assert "chart_data" in result
    assert isinstance(result["chart_data"], bytes)
    assert len(result["chart_data"]) > 0
    
    # Verify database entry
    db_chart = db_session.query(Chart).filter_by(
        user_id=test_user.user_id,
        chart_type="REWARD_CHART"
    ).first()
    assert db_chart is not None
    assert db_chart.is_success is True
    assert db_chart.error is None

def test_generate_activity_chart(db_session, test_user, test_activity_data, test_chart_style):
    """Test activity chart generation"""
    # Generate activity chart
    result = generate_activity_chart(
        user_id=test_user.user_id,
        data=test_activity_data,
        chart_style=test_chart_style,
        db_session=db_session
    )
    
    # Verify chart result
    assert isinstance(result, Dict)
    assert "chart_id" in result
    assert "chart_type" in result
    
    # Verify chart metadata
    assert result["chart_type"] == "ACTIVITY_CHART"
    assert result["chart_style"]["type"] == "line"
    
    # Verify chart content
    assert "chart_data" in result
    assert isinstance(result["chart_data"], bytes)
    assert len(result["chart_data"]) > 0
    
    # Verify database entry
    db_chart = db_session.query(Chart).filter_by(
        user_id=test_user.user_id,
        chart_type="ACTIVITY_CHART"
    ).first()
    assert db_chart is not None
    assert db_chart.is_success is True
    assert db_chart.error is None

def test_generate_analytics_chart(db_session, test_user, test_analytics_data, test_chart_style):
    """Test analytics chart generation"""
    # Generate analytics chart
    result = generate_analytics_chart(
        user_id=test_user.user_id,
        data=test_analytics_data,
        chart_style=test_chart_style,
        db_session=db_session
    )
    
    # Verify chart result
    assert isinstance(result, Dict)
    assert "chart_id" in result
    assert "chart_type" in result
    
    # Verify chart metadata
    assert result["chart_type"] == "ANALYTICS_CHART"
    assert result["chart_style"]["type"] == "line"
    
    # Verify chart content
    assert "chart_data" in result
    assert isinstance(result["chart_data"], bytes)
    assert len(result["chart_data"]) > 0
    
    # Verify database entry
    db_chart = db_session.query(Chart).filter_by(
        user_id=test_user.user_id,
        chart_type="ANALYTICS_CHART"
    ).first()
    assert db_chart is not None
    assert db_chart.is_success is True
    assert db_chart.error is None

def test_customize_chart(db_session, test_user, test_performance_data, test_chart_style):
    """Test chart customization"""
    # Customize chart
    result = customize_chart(
        user_id=test_user.user_id,
        data=test_performance_data,
        chart_style=test_chart_style,
        db_session=db_session
    )
    
    # Verify customization result
    assert isinstance(result, Dict)
    assert "chart_id" in result
    assert "chart_type" in result
    
    # Verify chart metadata
    assert result["chart_type"] == "CUSTOMIZED_CHART"
    assert result["chart_style"]["type"] == "line"
    assert result["chart_style"]["theme"] == "dark"
    
    # Verify chart content
    assert "chart_data" in result
    assert isinstance(result["chart_data"], bytes)
    assert len(result["chart_data"]) > 0
    
    # Verify database entry
    db_chart = db_session.query(Chart).filter_by(
        user_id=test_user.user_id,
        chart_type="CUSTOMIZED_CHART"
    ).first()
    assert db_chart is not None
    assert db_chart.is_success is True
    assert db_chart.error is None

def test_export_chart(db_session, test_user, test_performance_data, test_chart_style, test_export_config):
    """Test chart export"""
    # Export chart
    result = export_chart(
        user_id=test_user.user_id,
        data=test_performance_data,
        chart_style=test_chart_style,
        export_config=test_export_config,
        db_session=db_session
    )
    
    # Verify export result
    assert isinstance(result, Dict)
    assert "export_id" in result
    assert "export_status" in result
    
    # Verify export metadata
    assert result["export_format"] == "png"
    assert result["export_status"] in ["SUCCESS", "FAILED"]
    
    # Verify export details
    assert "export_details" in result
    assert isinstance(result["export_details"], Dict)
    assert "file_path" in result["export_details"]
    assert "file_size" in result["export_details"]
    assert "timestamp" in result["export_details"]
    
    # Verify database entry
    db_export = db_session.query(ChartExport).filter_by(
        user_id=test_user.user_id,
        export_format="PNG"
    ).first()
    assert db_export is not None
    assert db_export.is_success is True
    assert db_export.error is None

def test_visualization_error_handling(db_session, test_user):
    """Test visualization error handling"""
    # Invalid user ID
    with pytest.raises(VisualizationError) as excinfo:
        generate_performance_chart(
            user_id=None,
            data=pd.DataFrame(),
            chart_style={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(VisualizationError) as excinfo:
        generate_risk_chart(
            user_id=test_user.user_id,
            data=None,
            chart_style={},
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Empty data
    with pytest.raises(VisualizationError) as excinfo:
        generate_reward_chart(
            user_id=test_user.user_id,
            data=pd.DataFrame(),
            chart_style={},
            db_session=db_session
        )
    assert "Empty data" in str(excinfo.value)
    
    # Invalid chart type
    with pytest.raises(VisualizationError) as excinfo:
        generate_activity_chart(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            chart_style={"type": "invalid_type"},
            db_session=db_session
        )
    assert "Invalid chart type" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBVisualizationError) as excinfo:
        generate_performance_chart(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            chart_style={"type": "line"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value)

def test_generate_performance_visualization(db_session, test_user, test_performance_data, test_visualization_config):
    """Test generating performance visualization"""
    # Generate performance visualization
    result = generate_visualization(
        user_id=test_user.user_id,
        data=test_performance_data,
        visualization_config=test_visualization_config,
        db_session=db_session
    )
    
    # Verify visualization result
    assert isinstance(result, Dict)
    assert "visualization_id" in result
    assert "visualization_data" in result
    assert "visualization_details" in result
    
    # Verify visualization metadata
    assert result["visualization_type"] == "performance"
    assert result["chart_type"] == "line"
    assert "style" in result
    assert "interactive" in result
    assert "export" in result
    
    # Verify data columns
    assert "data_columns" in result
    assert all(col in result["data_columns"] for col in test_visualization_config["data_columns"])
    
    # Verify style configuration
    style = result["style"]
    assert style["theme"] == "dark"
    assert len(style["colors"]) == 4
    assert style["font_size"] == 12
    assert style["title"] == "Performance Metrics Over Time"
    assert style["x_label"] == "Time"
    assert style["y_label"] == "Performance"
    
    # Verify interactive features
    interactive = result["interactive"]
    assert interactive["zoom"] is True
    assert interactive["pan"] is True
    assert interactive["tooltip"] is True
    assert interactive["legend"] is True
    
    # Verify export configuration
    export = result["export"]
    assert export["format"] == "png"
    assert export["resolution"] == "high"
    assert export["width"] == 1200
    assert export["height"] == 800
    
    # Verify visualization data
    assert "image_data" in result["visualization_data"]
    assert "metadata" in result["visualization_data"]
    
    # Verify database entry
    db_record = db_session.query(VisualizationRecord).filter_by(
        user_id=test_user.user_id,
        visualization_id=result["visualization_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_generate_risk_visualization(db_session, test_user, test_risk_data):
    """Test generating risk visualization"""
    # Create visualization config for risk data
    risk_config = {
        "visualization_type": "risk",
        "chart_type": "heatmap",
        "data_columns": ["mining_risk", "staking_risk", "trading_risk", "overall_risk"],
        "time_column": "timestamp",
        "style": {
            "theme": "light",
            "colors": ["#fee2e2", "#fecaca", "#fca5a5", "#f87171", "#ef4444"],
            "font_size": 14,
            "title": "Risk Metrics Heatmap",
            "x_label": "Time",
            "y_label": "Risk Level"
        },
        "interactive": {
            "zoom": True,
            "pan": True,
            "tooltip": True,
            "legend": True
        },
        "export": {
            "format": "svg",
            "resolution": "high",
            "width": 1000,
            "height": 600
        }
    }
    
    # Generate risk visualization
    result = generate_visualization(
        user_id=test_user.user_id,
        data=test_risk_data,
        visualization_config=risk_config,
        db_session=db_session
    )
    
    # Verify visualization result
    assert isinstance(result, Dict)
    assert "visualization_id" in result
    assert "visualization_data" in result
    assert "visualization_details" in result
    
    # Verify visualization metadata
    assert result["visualization_type"] == "risk"
    assert result["chart_type"] == "heatmap"
    assert "style" in result
    assert "interactive" in result
    assert "export" in result
    
    # Verify data columns
    assert "data_columns" in result
    assert all(col in result["data_columns"] for col in risk_config["data_columns"])
    
    # Verify style configuration
    style = result["style"]
    assert style["theme"] == "light"
    assert len(style["colors"]) == 5
    assert style["font_size"] == 14
    assert style["title"] == "Risk Metrics Heatmap"
    assert style["x_label"] == "Time"
    assert style["y_label"] == "Risk Level"
    
    # Verify interactive features
    interactive = result["interactive"]
    assert interactive["zoom"] is True
    assert interactive["pan"] is True
    assert interactive["tooltip"] is True
    assert interactive["legend"] is True
    
    # Verify export configuration
    export = result["export"]
    assert export["format"] == "svg"
    assert export["resolution"] == "high"
    assert export["width"] == 1000
    assert export["height"] == 600
    
    # Verify visualization data
    assert "image_data" in result["visualization_data"]
    assert "metadata" in result["visualization_data"]
    
    # Verify database entry
    db_record = db_session.query(VisualizationRecord).filter_by(
        user_id=test_user.user_id,
        visualization_id=result["visualization_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_generate_reward_visualization(db_session, test_user, test_reward_data):
    """Test generating reward visualization"""
    # Create visualization config for reward data
    reward_config = {
        "visualization_type": "reward",
        "chart_type": "bar",
        "data_columns": ["mining_reward", "staking_reward", "trading_reward", "overall_reward"],
        "time_column": "timestamp",
        "style": {
            "theme": "dark",
            "colors": ["#4ade80", "#22c55e", "#16a34a", "#15803d"],
            "font_size": 12,
            "title": "Reward Distribution",
            "x_label": "Time",
            "y_label": "Reward Amount"
        },
        "interactive": {
            "zoom": True,
            "pan": True,
            "tooltip": True,
            "legend": True
        },
        "export": {
            "format": "pdf",
            "resolution": "high",
            "width": 1200,
            "height": 800
        }
    }
    
    # Generate reward visualization
    result = generate_visualization(
        user_id=test_user.user_id,
        data=test_reward_data,
        visualization_config=reward_config,
        db_session=db_session
    )
    
    # Verify visualization result
    assert isinstance(result, Dict)
    assert "visualization_id" in result
    assert "visualization_data" in result
    assert "visualization_details" in result
    
    # Verify visualization metadata
    assert result["visualization_type"] == "reward"
    assert result["chart_type"] == "bar"
    assert "style" in result
    assert "interactive" in result
    assert "export" in result
    
    # Verify data columns
    assert "data_columns" in result
    assert all(col in result["data_columns"] for col in reward_config["data_columns"])
    
    # Verify style configuration
    style = result["style"]
    assert style["theme"] == "dark"
    assert len(style["colors"]) == 4
    assert style["font_size"] == 12
    assert style["title"] == "Reward Distribution"
    assert style["x_label"] == "Time"
    assert style["y_label"] == "Reward Amount"
    
    # Verify interactive features
    interactive = result["interactive"]
    assert interactive["zoom"] is True
    assert interactive["pan"] is True
    assert interactive["tooltip"] is True
    assert interactive["legend"] is True
    
    # Verify export configuration
    export = result["export"]
    assert export["format"] == "pdf"
    assert export["resolution"] == "high"
    assert export["width"] == 1200
    assert export["height"] == 800
    
    # Verify visualization data
    assert "image_data" in result["visualization_data"]
    assert "metadata" in result["visualization_data"]
    
    # Verify database entry
    db_record = db_session.query(VisualizationRecord).filter_by(
        user_id=test_user.user_id,
        visualization_id=result["visualization_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_generate_activity_visualization(db_session, test_user, test_activity_data):
    """Test generating activity visualization"""
    # Create visualization config for activity data
    activity_config = {
        "visualization_type": "activity",
        "chart_type": "area",
        "data_columns": ["mining_activity", "staking_activity", "trading_activity", "overall_activity"],
        "time_column": "timestamp",
        "style": {
            "theme": "light",
            "colors": ["#93c5fd", "#60a5fa", "#3b82f6", "#2563eb"],
            "font_size": 14,
            "title": "Activity Levels Over Time",
            "x_label": "Time",
            "y_label": "Activity Level"
        },
        "interactive": {
            "zoom": True,
            "pan": True,
            "tooltip": True,
            "legend": True
        },
        "export": {
            "format": "png",
            "resolution": "high",
            "width": 1000,
            "height": 600
        }
    }
    
    # Generate activity visualization
    result = generate_visualization(
        user_id=test_user.user_id,
        data=test_activity_data,
        visualization_config=activity_config,
        db_session=db_session
    )
    
    # Verify visualization result
    assert isinstance(result, Dict)
    assert "visualization_id" in result
    assert "visualization_data" in result
    assert "visualization_details" in result
    
    # Verify visualization metadata
    assert result["visualization_type"] == "activity"
    assert result["chart_type"] == "area"
    assert "style" in result
    assert "interactive" in result
    assert "export" in result
    
    # Verify data columns
    assert "data_columns" in result
    assert all(col in result["data_columns"] for col in activity_config["data_columns"])
    
    # Verify style configuration
    style = result["style"]
    assert style["theme"] == "light"
    assert len(style["colors"]) == 4
    assert style["font_size"] == 14
    assert style["title"] == "Activity Levels Over Time"
    assert style["x_label"] == "Time"
    assert style["y_label"] == "Activity Level"
    
    # Verify interactive features
    interactive = result["interactive"]
    assert interactive["zoom"] is True
    assert interactive["pan"] is True
    assert interactive["tooltip"] is True
    assert interactive["legend"] is True
    
    # Verify export configuration
    export = result["export"]
    assert export["format"] == "png"
    assert export["resolution"] == "high"
    assert export["width"] == 1000
    assert export["height"] == 600
    
    # Verify visualization data
    assert "image_data" in result["visualization_data"]
    assert "metadata" in result["visualization_data"]
    
    # Verify database entry
    db_record = db_session.query(VisualizationRecord).filter_by(
        user_id=test_user.user_id,
        visualization_id=result["visualization_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_generate_custom_visualization(db_session, test_user, test_performance_data):
    """Test generating custom visualization"""
    # Create visualization config with custom settings
    custom_config = {
        "visualization_type": "custom",
        "chart_type": "scatter",
        "data_columns": ["mining_performance", "staking_performance"],
        "time_column": "timestamp",
        "style": {
            "theme": "custom",
            "colors": ["#a855f7", "#8b5cf6"],
            "font_size": 16,
            "title": "Custom Performance Analysis",
            "x_label": "Mining Performance",
            "y_label": "Staking Performance"
        },
        "interactive": {
            "zoom": True,
            "pan": True,
            "tooltip": True,
            "legend": True,
            "custom_features": {
                "regression_line": True,
                "confidence_interval": True
            }
        },
        "export": {
            "format": "svg",
            "resolution": "ultra",
            "width": 1500,
            "height": 1000
        }
    }
    
    # Generate custom visualization
    result = generate_visualization(
        user_id=test_user.user_id,
        data=test_performance_data,
        visualization_config=custom_config,
        db_session=db_session
    )
    
    # Verify visualization result
    assert isinstance(result, Dict)
    assert "visualization_id" in result
    assert "visualization_data" in result
    assert "visualization_details" in result
    
    # Verify visualization metadata
    assert result["visualization_type"] == "custom"
    assert result["chart_type"] == "scatter"
    assert "style" in result
    assert "interactive" in result
    assert "export" in result
    
    # Verify data columns
    assert "data_columns" in result
    assert all(col in result["data_columns"] for col in custom_config["data_columns"])
    
    # Verify style configuration
    style = result["style"]
    assert style["theme"] == "custom"
    assert len(style["colors"]) == 2
    assert style["font_size"] == 16
    assert style["title"] == "Custom Performance Analysis"
    assert style["x_label"] == "Mining Performance"
    assert style["y_label"] == "Staking Performance"
    
    # Verify interactive features
    interactive = result["interactive"]
    assert interactive["zoom"] is True
    assert interactive["pan"] is True
    assert interactive["tooltip"] is True
    assert interactive["legend"] is True
    assert interactive["custom_features"]["regression_line"] is True
    assert interactive["custom_features"]["confidence_interval"] is True
    
    # Verify export configuration
    export = result["export"]
    assert export["format"] == "svg"
    assert export["resolution"] == "ultra"
    assert export["width"] == 1500
    assert export["height"] == 1000
    
    # Verify visualization data
    assert "image_data" in result["visualization_data"]
    assert "metadata" in result["visualization_data"]
    
    # Verify database entry
    db_record = db_session.query(VisualizationRecord).filter_by(
        user_id=test_user.user_id,
        visualization_id=result["visualization_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_get_visualization_info(db_session, test_user, test_performance_data, test_visualization_config):
    """Test visualization info retrieval"""
    # First, generate performance visualization
    visualization_result = generate_visualization(
        user_id=test_user.user_id,
        data=test_performance_data,
        visualization_config=test_visualization_config,
        db_session=db_session
    )
    
    visualization_id = visualization_result["visualization_id"]
    
    # Get visualization info
    result = get_visualization_info(
        user_id=test_user.user_id,
        visualization_id=visualization_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "visualization_id" in result
    assert result["visualization_id"] == visualization_id
    
    # Verify visualization metadata
    assert result["visualization_type"] == "performance"
    assert result["chart_type"] == "line"
    assert "style" in result
    assert "interactive" in result
    assert "export" in result
    
    # Verify visualization details
    assert "visualization_details" in result
    assert isinstance(result["visualization_details"], Dict)
    assert "timestamp" in result["visualization_details"]
    assert "generation_time" in result["visualization_details"]
    assert "file_size" in result["visualization_details"]
    
    # Verify database entry
    db_record = db_session.query(VisualizationRecord).filter_by(
        user_id=test_user.user_id,
        visualization_id=visualization_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_visualization_error_handling(db_session, test_user):
    """Test visualization error handling"""
    # Invalid user ID
    with pytest.raises(VisualizationError) as excinfo:
        generate_visualization(
            user_id=None,
            data=pd.DataFrame(),
            visualization_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(VisualizationError) as excinfo:
        generate_visualization(
            user_id=test_user.user_id,
            data=None,
            visualization_config={},
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Invalid visualization type
    with pytest.raises(VisualizationError) as excinfo:
        generate_visualization(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            visualization_config={"visualization_type": "invalid_type"},
            db_session=db_session
        )
    assert "Invalid visualization type" in str(excinfo.value)
    
    # Invalid chart type
    with pytest.raises(VisualizationError) as excinfo:
        generate_visualization(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            visualization_config={"visualization_type": "performance", "chart_type": "invalid_chart"},
            db_session=db_session
        )
    assert "Invalid chart type" in str(excinfo.value)
    
    # Invalid data columns
    with pytest.raises(VisualizationError) as excinfo:
        generate_visualization(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            visualization_config={"visualization_type": "performance", "data_columns": ["invalid_column"]},
            db_session=db_session
        )
    assert "Invalid data columns" in str(excinfo.value)
    
    # Invalid visualization ID
    with pytest.raises(VisualizationError) as excinfo:
        get_visualization_info(
            user_id=test_user.user_id,
            visualization_id="invalid_visualization_id",
            db_session=db_session
        )
    assert "Invalid visualization ID" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBVisualizationError) as excinfo:
        generate_visualization(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            visualization_config={"visualization_type": "performance"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value)

def test_create_chart(db_session, test_user, test_data, test_visualization_config):
    """Test chart creation"""
    # Create chart
    result = create_chart(
        data=test_data,
        chart_config=test_visualization_config["chart_generation"],
        user_id=test_user.user_id,
        message="Create performance chart",
        db_session=db_session
    )
    
    # Verify chart result
    assert isinstance(result, Dict)
    assert "chart_id" in result
    assert "timestamp" in result
    assert "chart" in result
    
    # Verify chart details
    chart = result["chart"]
    assert "type" in chart
    assert "data" in chart
    assert "options" in chart
    assert "metadata" in chart
    
    # Verify chart data
    data = chart["data"]
    assert "labels" in data
    assert "datasets" in data
    assert len(data["datasets"]) > 0
    
    # Verify chart options
    options = chart["options"]
    assert "responsive" in options
    assert "interactive" in options
    assert "animation" in options
    assert "tooltips" in options
    assert "legend" in options
    
    # Verify chart record
    chart_record = db_session.query(ChartRecord).filter_by(
        chart_id=result["chart_id"]
    ).first()
    assert chart_record is not None
    assert chart_record.status == "CREATED"
    assert chart_record.error is None

def test_create_dashboard(db_session, test_user, test_data, test_visualization_config):
    """Test dashboard creation"""
    # Create multiple charts first
    charts = []
    for i in range(3):
        chart = create_chart(
            data=test_data,
            chart_config=test_visualization_config["chart_generation"],
            user_id=test_user.user_id,
            message=f"Chart {i+1}",
            db_session=db_session
        )
        charts.append(chart["chart_id"])
    
    # Create dashboard
    result = create_dashboard(
        chart_ids=charts,
        dashboard_config=test_visualization_config["dashboard_creation"],
        user_id=test_user.user_id,
        message="Create performance dashboard",
        db_session=db_session
    )
    
    # Verify dashboard result
    assert isinstance(result, Dict)
    assert "dashboard_id" in result
    assert "timestamp" in result
    assert "dashboard" in result
    
    # Verify dashboard details
    dashboard = result["dashboard"]
    assert "layout" in dashboard
    assert "charts" in dashboard
    assert "options" in dashboard
    assert "metadata" in dashboard
    
    # Verify dashboard layout
    layout = dashboard["layout"]
    assert "type" in layout
    assert "widgets" in layout
    assert len(layout["widgets"]) == len(charts)
    
    # Verify dashboard options
    options = dashboard["options"]
    assert "refresh_rate" in options
    assert "auto_save" in options
    assert "sharing_enabled" in options
    assert "export_formats" in options
    
    # Verify dashboard record
    dashboard_record = db_session.query(DashboardRecord).filter_by(
        dashboard_id=result["dashboard_id"]
    ).first()
    assert dashboard_record is not None
    assert dashboard_record.status == "CREATED"
    assert dashboard_record.error is None

def test_create_interactive_visualization(db_session, test_user, test_data, test_visualization_config):
    """Test interactive visualization creation"""
    # Create interactive visualization
    result = create_interactive_visualization(
        data=test_data,
        interactive_config=test_visualization_config["interactive_features"],
        user_id=test_user.user_id,
        message="Create interactive visualization",
        db_session=db_session
    )
    
    # Verify visualization result
    assert isinstance(result, Dict)
    assert "visualization_id" in result
    assert "timestamp" in result
    assert "visualization" in result
    
    # Verify visualization details
    visualization = result["visualization"]
    assert "type" in visualization
    assert "data" in visualization
    assert "interactive_features" in visualization
    assert "metadata" in visualization
    
    # Verify interactive features
    features = visualization["interactive_features"]
    assert "zoom" in features
    assert "pan" in features
    assert "filter" in features
    assert "drill_down" in features
    assert "cross_filter" in features
    assert "annotations" in features
    assert "custom_controls" in features
    
    # Verify visualization record
    visualization_record = db_session.query(VisualizationRecord).filter_by(
        visualization_id=result["visualization_id"]
    ).first()
    assert visualization_record is not None
    assert visualization_record.status == "CREATED"
    assert visualization_record.error is None

def test_create_custom_plot(db_session, test_user, test_data, test_visualization_config):
    """Test custom plot creation"""
    # Create custom plot
    result = create_custom_plot(
        data=test_data,
        plot_config=test_visualization_config["custom_plots"],
        user_id=test_user.user_id,
        message="Create custom plot",
        db_session=db_session
    )
    
    # Verify plot result
    assert isinstance(result, Dict)
    assert "plot_id" in result
    assert "timestamp" in result
    assert "plot" in result
    
    # Verify plot details
    plot = result["plot"]
    assert "type" in plot
    assert "data" in plot
    assert "style" in plot
    assert "metadata" in plot
    
    # Verify plot style
    style = plot["style"]
    assert "template" in style
    assert "colors" in style
    assert "fonts" in style
    assert "layout" in style
    
    # Verify plot record
    visualization_record = db_session.query(VisualizationRecord).filter_by(
        plot_id=result["plot_id"]
    ).first()
    assert visualization_record is not None
    assert visualization_record.status == "CREATED"
    assert visualization_record.error is None

def test_export_visualization(db_session, test_user, test_data, test_visualization_config):
    """Test visualization export"""
    # Create chart first
    chart = create_chart(
        data=test_data,
        chart_config=test_visualization_config["chart_generation"],
        user_id=test_user.user_id,
        message="Test chart",
        db_session=db_session
    )
    
    # Export visualization
    result = export_visualization(
        visualization_id=chart["chart_id"],
        export_config=test_visualization_config["export_options"],
        user_id=test_user.user_id,
        message="Export visualization",
        db_session=db_session
    )
    
    # Verify export result
    assert isinstance(result, Dict)
    assert "export_id" in result
    assert "timestamp" in result
    assert "export" in result
    
    # Verify export details
    export = result["export"]
    assert "format" in export
    assert "url" in export
    assert "metadata" in export
    assert "size" in export
    
    # Verify export format
    assert export["format"] in test_visualization_config["export_options"]["formats"]
    
    # Verify visualization record
    visualization_record = db_session.query(VisualizationRecord).filter_by(
        visualization_id=chart["chart_id"]
    ).first()
    assert visualization_record is not None
    assert visualization_record.status == "EXPORTED"
    assert visualization_record.error is None

def test_list_visualizations(db_session, test_user, test_data, test_visualization_config):
    """Test visualization listing"""
    # Create multiple visualizations
    for i in range(5):
        create_chart(
            data=test_data,
            chart_config=test_visualization_config["chart_generation"],
            user_id=test_user.user_id,
            message=f"Chart {i+1}",
            db_session=db_session
        )
    
    # List visualizations
    result = list_visualizations(
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify listing result
    assert isinstance(result, Dict)
    assert "timestamp" in result
    assert "visualizations" in result
    
    # Verify visualizations list
    visualizations = result["visualizations"]
    assert isinstance(visualizations, List)
    assert len(visualizations) == 5
    
    # Verify visualization details
    for visualization in visualizations:
        assert "visualization_id" in visualization
        assert "timestamp" in visualization
        assert "type" in visualization

def test_delete_visualization(db_session, test_user, test_data, test_visualization_config):
    """Test visualization deletion"""
    # Create chart
    chart = create_chart(
        data=test_data,
        chart_config=test_visualization_config["chart_generation"],
        user_id=test_user.user_id,
        message="Test chart",
        db_session=db_session
    )
    
    # Delete visualization
    result = delete_visualization(
        visualization_id=chart["chart_id"],
        user_id=test_user.user_id,
        message="Delete test visualization",
        db_session=db_session
    )
    
    # Verify deletion result
    assert isinstance(result, Dict)
    assert "deletion_id" in result
    assert "timestamp" in result
    assert "status" in result
    
    # Verify status
    assert result["status"] == "DELETED"
    
    # Verify visualization record
    visualization_record = db_session.query(VisualizationRecord).filter_by(
        visualization_id=chart["chart_id"]
    ).first()
    assert visualization_record is not None
    assert visualization_record.status == "DELETED"
    assert visualization_record.error is None

def test_visualization_error_handling(db_session, test_user):
    """Test visualization error handling"""
    # Invalid chart configuration
    with pytest.raises(VisualizationError) as excinfo:
        create_chart(
            data=pd.DataFrame(),
            chart_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid chart configuration" in str(excinfo.value)
    
    # Invalid dashboard configuration
    with pytest.raises(VisualizationError) as excinfo:
        create_dashboard(
            chart_ids=[],
            dashboard_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid dashboard configuration" in str(excinfo.value)
    
    # Invalid interactive configuration
    with pytest.raises(VisualizationError) as excinfo:
        create_interactive_visualization(
            data=pd.DataFrame(),
            interactive_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid interactive configuration" in str(excinfo.value)
    
    # Invalid custom plot configuration
    with pytest.raises(VisualizationError) as excinfo:
        create_custom_plot(
            data=pd.DataFrame(),
            plot_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid custom plot configuration" in str(excinfo.value)
    
    # Invalid export configuration
    with pytest.raises(VisualizationError) as excinfo:
        export_visualization(
            visualization_id="invalid_id",
            export_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=db_session
        )
    assert "Invalid export configuration" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBVisualizationError) as excinfo:
        create_chart(
            data=pd.DataFrame(),
            chart_config={},
            user_id=test_user.user_id,
            message="Test",
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 