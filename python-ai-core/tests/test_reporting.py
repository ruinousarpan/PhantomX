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
import matplotlib.pyplot as plt
import seaborn as sns
from unittest.mock import patch, MagicMock

from core.reporting import (
    generate_report,
    create_report_template,
    aggregate_report_data,
    schedule_report,
    distribute_report,
    get_report_info,
    list_reports,
    delete_report,
    ReportingError
)
from database.models import User, ReportRecord, ReportTemplate, ReportSchedule, ReportDistribution
from database.exceptions import ReportingError as DBReportingError

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
def test_report_config():
    """Create test report configuration"""
    return {
        "report_type": "performance",
        "sections": {
            "summary": {
                "title": "Performance Summary",
                "metrics": ["mean", "std", "min", "max"],
                "visualizations": ["line_plot", "box_plot"]
            },
            "analysis": {
                "title": "Performance Analysis",
                "metrics": ["trend", "seasonality", "correlation"],
                "visualizations": ["heatmap", "scatter_plot"]
            },
            "insights": {
                "title": "Key Insights",
                "metrics": ["top_performers", "risk_adjusted_returns"],
                "visualizations": ["bar_plot", "radar_plot"]
            }
        },
        "format": {
            "output_type": "pdf",
            "template": "default",
            "style": {
                "theme": "light",
                "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
                "font_size": 12
            }
        }
    }

@pytest.fixture
def test_report_data():
    """Create test report data"""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
    return pd.DataFrame({
        "timestamp": dates,
        "revenue": np.random.uniform(1000, 5000, 100),
        "expenses": np.random.uniform(500, 2500, 100),
        "profit": np.random.uniform(500, 2500, 100),
        "customers": np.random.randint(100, 500, 100),
        "transactions": np.random.randint(500, 1500, 100),
        "satisfaction": np.random.uniform(4.0, 5.0, 100)
    })

@pytest.fixture
def test_reporting_config():
    """Create test reporting configuration"""
    return {
        "report_generation": {
            "enabled": True,
            "report_types": [
                "financial",
                "operational",
                "analytical",
                "compliance",
                "executive"
            ],
            "formats": [
                "pdf",
                "excel",
                "html",
                "json",
                "csv"
            ],
            "components": [
                "summary",
                "details",
                "charts",
                "tables",
                "appendix"
            ],
            "styling": {
                "theme": "corporate",
                "colors": ["#1f77b4", "#ff7f0e", "#2ca02c"],
                "fonts": ["Arial", "Helvetica"],
                "logo_path": "/assets/logo.png"
            }
        },
        "report_templates": {
            "enabled": True,
            "template_types": [
                "standard",
                "custom",
                "regulatory"
            ],
            "sections": [
                "header",
                "content",
                "footer"
            ],
            "variables": [
                "company_name",
                "report_period",
                "generated_date",
                "author"
            ],
            "placeholders": {
                "title": "${title}",
                "subtitle": "${subtitle}",
                "date_range": "${date_range}",
                "metrics": "${metrics}"
            }
        },
        "data_aggregation": {
            "enabled": True,
            "methods": [
                "sum",
                "average",
                "min",
                "max",
                "count",
                "distinct"
            ],
            "dimensions": [
                "time",
                "category",
                "location",
                "department"
            ],
            "time_periods": [
                "hourly",
                "daily",
                "weekly",
                "monthly",
                "quarterly",
                "yearly"
            ],
            "calculations": {
                "growth_rate": "percentage",
                "moving_average": "window",
                "year_over_year": "comparison"
            }
        },
        "report_scheduling": {
            "enabled": True,
            "frequencies": [
                "hourly",
                "daily",
                "weekly",
                "monthly",
                "quarterly"
            ],
            "timing": {
                "timezone": "UTC",
                "business_days": True,
                "holidays": True
            },
            "retry_policy": {
                "max_retries": 3,
                "retry_delay": 300,
                "timeout": 3600
            },
            "notifications": {
                "on_success": True,
                "on_failure": True,
                "on_delay": True
            }
        },
        "report_distribution": {
            "enabled": True,
            "channels": [
                "email",
                "sftp",
                "api",
                "dashboard",
                "storage"
            ],
            "formats": {
                "email": ["pdf", "excel"],
                "sftp": ["csv", "json"],
                "api": ["json"],
                "dashboard": ["html"],
                "storage": ["all"]
            },
            "security": {
                "encryption": True,
                "password_protection": True,
                "expiry": "7d",
                "access_tracking": True
            }
        }
    }

def test_generate_performance_report(db_session, test_user, test_performance_data, test_report_config):
    """Test generating performance report"""
    # Generate performance report
    result = generate_report(
        user_id=test_user.user_id,
        data=test_performance_data,
        report_config=test_report_config,
        db_session=db_session
    )
    
    # Verify report result
    assert isinstance(result, Dict)
    assert "report_id" in result
    assert "report_content" in result
    assert "report_details" in result
    
    # Verify report metadata
    assert result["report_type"] == "performance"
    assert "sections" in result
    assert "format" in result
    
    # Verify report sections
    sections = result["sections"]
    assert "summary" in sections
    assert "analysis" in sections
    assert "insights" in sections
    
    # Verify summary section
    summary = sections["summary"]
    assert "title" in summary
    assert "metrics" in summary
    assert "visualizations" in summary
    assert isinstance(summary["metrics"], Dict)
    assert isinstance(summary["visualizations"], List)
    
    # Verify analysis section
    analysis = sections["analysis"]
    assert "title" in analysis
    assert "metrics" in analysis
    assert "visualizations" in analysis
    assert isinstance(analysis["metrics"], Dict)
    assert isinstance(analysis["visualizations"], List)
    
    # Verify insights section
    insights = sections["insights"]
    assert "title" in insights
    assert "metrics" in insights
    assert "visualizations" in insights
    assert isinstance(insights["metrics"], Dict)
    assert isinstance(insights["visualizations"], List)
    
    # Verify report format
    format_info = result["format"]
    assert format_info["output_type"] == "pdf"
    assert format_info["template"] == "default"
    assert "style" in format_info
    
    # Verify database entry
    db_record = db_session.query(ReportRecord).filter_by(
        user_id=test_user.user_id,
        report_id=result["report_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_generate_risk_report(db_session, test_user, test_risk_data):
    """Test generating risk report"""
    # Create report config for risk data
    risk_config = {
        "report_type": "risk",
        "sections": {
            "summary": {
                "title": "Risk Summary",
                "metrics": ["var", "volatility", "tail_risk"],
                "visualizations": ["line_plot", "histogram"]
            },
            "analysis": {
                "title": "Risk Analysis",
                "metrics": ["risk_decomposition", "risk_contribution"],
                "visualizations": ["heatmap", "treemap"]
            },
            "alerts": {
                "title": "Risk Alerts",
                "metrics": ["threshold_breaches", "trend_changes"],
                "visualizations": ["gauge_plot", "alert_timeline"]
            }
        },
        "format": {
            "output_type": "html",
            "template": "risk",
            "style": {
                "theme": "dark",
                "colors": ["#d73027", "#fc8d59", "#fee090", "#91bfdb"],
                "font_size": 14
            }
        }
    }
    
    # Generate risk report
    result = generate_report(
        user_id=test_user.user_id,
        data=test_risk_data,
        report_config=risk_config,
        db_session=db_session
    )
    
    # Verify report result
    assert isinstance(result, Dict)
    assert "report_id" in result
    assert "report_content" in result
    assert "report_details" in result
    
    # Verify report metadata
    assert result["report_type"] == "risk"
    assert "sections" in result
    assert "format" in result
    
    # Verify report sections
    sections = result["sections"]
    assert "summary" in sections
    assert "analysis" in sections
    assert "alerts" in sections
    
    # Verify summary section
    summary = sections["summary"]
    assert "title" in summary
    assert "metrics" in summary
    assert "visualizations" in summary
    assert isinstance(summary["metrics"], Dict)
    assert isinstance(summary["visualizations"], List)
    
    # Verify analysis section
    analysis = sections["analysis"]
    assert "title" in analysis
    assert "metrics" in analysis
    assert "visualizations" in analysis
    assert isinstance(analysis["metrics"], Dict)
    assert isinstance(analysis["visualizations"], List)
    
    # Verify alerts section
    alerts = sections["alerts"]
    assert "title" in alerts
    assert "metrics" in alerts
    assert "visualizations" in alerts
    assert isinstance(alerts["metrics"], Dict)
    assert isinstance(alerts["visualizations"], List)
    
    # Verify report format
    format_info = result["format"]
    assert format_info["output_type"] == "html"
    assert format_info["template"] == "risk"
    assert "style" in format_info
    
    # Verify database entry
    db_record = db_session.query(ReportRecord).filter_by(
        user_id=test_user.user_id,
        report_id=result["report_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_generate_reward_report(db_session, test_user, test_reward_data):
    """Test generating reward report"""
    # Create report config for reward data
    reward_config = {
        "report_type": "reward",
        "sections": {
            "summary": {
                "title": "Reward Summary",
                "metrics": ["total_rewards", "reward_rate", "reward_distribution"],
                "visualizations": ["line_plot", "pie_chart"]
            },
            "analysis": {
                "title": "Reward Analysis",
                "metrics": ["reward_attribution", "reward_efficiency"],
                "visualizations": ["waterfall_plot", "scatter_plot"]
            },
            "forecasts": {
                "title": "Reward Forecasts",
                "metrics": ["projected_rewards", "reward_scenarios"],
                "visualizations": ["forecast_plot", "scenario_plot"]
            }
        },
        "format": {
            "output_type": "excel",
            "template": "reward",
            "style": {
                "theme": "light",
                "colors": ["#4daf4a", "#377eb8", "#ff7f00", "#984ea3"],
                "font_size": 11
            }
        }
    }
    
    # Generate reward report
    result = generate_report(
        user_id=test_user.user_id,
        data=test_reward_data,
        report_config=reward_config,
        db_session=db_session
    )
    
    # Verify report result
    assert isinstance(result, Dict)
    assert "report_id" in result
    assert "report_content" in result
    assert "report_details" in result
    
    # Verify report metadata
    assert result["report_type"] == "reward"
    assert "sections" in result
    assert "format" in result
    
    # Verify report sections
    sections = result["sections"]
    assert "summary" in sections
    assert "analysis" in sections
    assert "forecasts" in sections
    
    # Verify summary section
    summary = sections["summary"]
    assert "title" in summary
    assert "metrics" in summary
    assert "visualizations" in summary
    assert isinstance(summary["metrics"], Dict)
    assert isinstance(summary["visualizations"], List)
    
    # Verify analysis section
    analysis = sections["analysis"]
    assert "title" in analysis
    assert "metrics" in analysis
    assert "visualizations" in analysis
    assert isinstance(analysis["metrics"], Dict)
    assert isinstance(analysis["visualizations"], List)
    
    # Verify forecasts section
    forecasts = sections["forecasts"]
    assert "title" in forecasts
    assert "metrics" in forecasts
    assert "visualizations" in forecasts
    assert isinstance(forecasts["metrics"], Dict)
    assert isinstance(forecasts["visualizations"], List)
    
    # Verify report format
    format_info = result["format"]
    assert format_info["output_type"] == "excel"
    assert format_info["template"] == "reward"
    assert "style" in format_info
    
    # Verify database entry
    db_record = db_session.query(ReportRecord).filter_by(
        user_id=test_user.user_id,
        report_id=result["report_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_generate_activity_report(db_session, test_user, test_activity_data):
    """Test generating activity report"""
    # Create report config for activity data
    activity_config = {
        "report_type": "activity",
        "sections": {
            "summary": {
                "title": "Activity Summary",
                "metrics": ["active_hours", "peak_times", "activity_patterns"],
                "visualizations": ["line_plot", "heatmap"]
            },
            "analysis": {
                "title": "Activity Analysis",
                "metrics": ["engagement_metrics", "consistency_metrics"],
                "visualizations": ["bar_plot", "radar_plot"]
            },
            "comparisons": {
                "title": "Activity Comparisons",
                "metrics": ["cross_activity", "temporal_patterns"],
                "visualizations": ["comparison_plot", "calendar_plot"]
            }
        },
        "format": {
            "output_type": "json",
            "template": "activity",
            "style": {
                "theme": "light",
                "colors": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"],
                "font_size": 12
            }
        }
    }
    
    # Generate activity report
    result = generate_report(
        user_id=test_user.user_id,
        data=test_activity_data,
        report_config=activity_config,
        db_session=db_session
    )
    
    # Verify report result
    assert isinstance(result, Dict)
    assert "report_id" in result
    assert "report_content" in result
    assert "report_details" in result
    
    # Verify report metadata
    assert result["report_type"] == "activity"
    assert "sections" in result
    assert "format" in result
    
    # Verify report sections
    sections = result["sections"]
    assert "summary" in sections
    assert "analysis" in sections
    assert "comparisons" in sections
    
    # Verify summary section
    summary = sections["summary"]
    assert "title" in summary
    assert "metrics" in summary
    assert "visualizations" in summary
    assert isinstance(summary["metrics"], Dict)
    assert isinstance(summary["visualizations"], List)
    
    # Verify analysis section
    analysis = sections["analysis"]
    assert "title" in analysis
    assert "metrics" in analysis
    assert "visualizations" in analysis
    assert isinstance(analysis["metrics"], Dict)
    assert isinstance(analysis["visualizations"], List)
    
    # Verify comparisons section
    comparisons = sections["comparisons"]
    assert "title" in comparisons
    assert "metrics" in comparisons
    assert "visualizations" in comparisons
    assert isinstance(comparisons["metrics"], Dict)
    assert isinstance(comparisons["visualizations"], List)
    
    # Verify report format
    format_info = result["format"]
    assert format_info["output_type"] == "json"
    assert format_info["template"] == "activity"
    assert "style" in format_info
    
    # Verify database entry
    db_record = db_session.query(ReportRecord).filter_by(
        user_id=test_user.user_id,
        report_id=result["report_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_generate_custom_report(db_session, test_user, test_performance_data):
    """Test generating custom report"""
    # Create custom report config
    custom_config = {
        "report_type": "custom",
        "sections": {
            "custom_metrics": {
                "title": "Custom Metrics",
                "metrics": {
                    "weighted_average": "lambda x: np.average(x, weights=range(len(x)))",
                    "rolling_zscore": "lambda x: (x - x.rolling(window=24).mean()) / x.rolling(window=24).std()",
                    "momentum": "lambda x: x.diff(periods=24) / x.shift(periods=24)"
                },
                "visualizations": ["custom_plot_1", "custom_plot_2"]
            }
        },
        "format": {
            "output_type": "custom",
            "template": "custom",
            "style": {
                "theme": "custom",
                "colors": ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c"],
                "font_size": 13
            }
        }
    }
    
    # Generate custom report
    result = generate_report(
        user_id=test_user.user_id,
        data=test_performance_data,
        report_config=custom_config,
        db_session=db_session
    )
    
    # Verify report result
    assert isinstance(result, Dict)
    assert "report_id" in result
    assert "report_content" in result
    assert "report_details" in result
    
    # Verify report metadata
    assert result["report_type"] == "custom"
    assert "sections" in result
    assert "format" in result
    
    # Verify custom metrics section
    custom_metrics = result["sections"]["custom_metrics"]
    assert "title" in custom_metrics
    assert "metrics" in custom_metrics
    assert "visualizations" in custom_metrics
    assert isinstance(custom_metrics["metrics"], Dict)
    assert isinstance(custom_metrics["visualizations"], List)
    
    # Verify custom format
    format_info = result["format"]
    assert format_info["output_type"] == "custom"
    assert format_info["template"] == "custom"
    assert "style" in format_info
    
    # Verify database entry
    db_record = db_session.query(ReportRecord).filter_by(
        user_id=test_user.user_id,
        report_id=result["report_id"]
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_get_report_info(db_session, test_user, test_performance_data, test_report_config):
    """Test report info retrieval"""
    # First, generate performance report
    report_result = generate_report(
        user_id=test_user.user_id,
        data=test_performance_data,
        report_config=test_report_config,
        db_session=db_session
    )
    
    report_id = report_result["report_id"]
    
    # Get report info
    result = get_report_info(
        user_id=test_user.user_id,
        report_id=report_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "report_id" in result
    assert result["report_id"] == report_id
    
    # Verify report metadata
    assert result["report_type"] == "performance"
    assert "sections" in result
    assert "format" in result
    
    # Verify report details
    assert "report_details" in result
    assert isinstance(result["report_details"], Dict)
    assert "timestamp" in result["report_details"]
    assert "generation_time" in result["report_details"]
    assert "file_size" in result["report_details"]
    
    # Verify database entry
    db_record = db_session.query(ReportRecord).filter_by(
        user_id=test_user.user_id,
        report_id=report_id
    ).first()
    assert db_record is not None
    assert db_record.is_active is True
    assert db_record.error is None

def test_reporting_error_handling(db_session, test_user):
    """Test reporting error handling"""
    # Invalid user ID
    with pytest.raises(ReportingError) as excinfo:
        generate_report(
            user_id=None,
            data=pd.DataFrame(),
            report_config={},
            db_session=db_session
        )
    assert "Invalid user ID" in str(excinfo.value)
    
    # Invalid data
    with pytest.raises(ReportingError) as excinfo:
        generate_report(
            user_id=test_user.user_id,
            data=None,
            report_config={},
            db_session=db_session
        )
    assert "Invalid data" in str(excinfo.value)
    
    # Invalid report type
    with pytest.raises(ReportingError) as excinfo:
        generate_report(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            report_config={"report_type": "invalid_type"},
            db_session=db_session
        )
    assert "Invalid report type" in str(excinfo.value)
    
    # Invalid sections
    with pytest.raises(ReportingError) as excinfo:
        generate_report(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            report_config={"report_type": "performance", "sections": {"invalid_section": {}}},
            db_session=db_session
        )
    assert "Invalid sections" in str(excinfo.value)
    
    # Invalid format
    with pytest.raises(ReportingError) as excinfo:
        generate_report(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            report_config={"report_type": "performance", "format": {"output_type": "invalid_type"}},
            db_session=db_session
        )
    assert "Invalid format" in str(excinfo.value)
    
    # Invalid report ID
    with pytest.raises(ReportingError) as excinfo:
        get_report_info(
            user_id=test_user.user_id,
            report_id="invalid_report_id",
            db_session=db_session
        )
    assert "Invalid report ID" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBReportingError) as excinfo:
        generate_report(
            user_id=test_user.user_id,
            data=pd.DataFrame({"col": [1, 2, 3]}),
            report_config={"report_type": "performance"},
            db_session=None
        )
    assert "Database error" in str(excinfo.value)

def test_generate_report(db_session, test_user, test_report_data, test_reporting_config):
    """Test report generation"""
    # Generate report
    result = generate_report(
        data=test_report_data,
        config=test_reporting_config["report_generation"],
        report_type="financial",
        format="pdf",
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify report result
    assert isinstance(result, Dict)
    assert "report_id" in result
    assert "timestamp" in result
    assert "report" in result
    
    # Verify report details
    report = result["report"]
    assert "type" in report
    assert "format" in report
    assert "content" in report
    assert "metadata" in report
    
    # Verify report components
    content = report["content"]
    for component in test_reporting_config["report_generation"]["components"]:
        assert component in content
    
    # Verify report record
    report_record = db_session.query(ReportRecord).filter_by(
        report_id=result["report_id"]
    ).first()
    assert report_record is not None
    assert report_record.status == "GENERATED"
    assert report_record.error is None

def test_create_report_template(db_session, test_user, test_reporting_config):
    """Test report template creation"""
    # Create template
    result = create_report_template(
        template_type="standard",
        config=test_reporting_config["report_templates"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify template result
    assert isinstance(result, Dict)
    assert "template_id" in result
    assert "timestamp" in result
    assert "template" in result
    
    # Verify template details
    template = result["template"]
    assert "type" in template
    assert "sections" in template
    assert "variables" in template
    assert "placeholders" in template
    
    # Verify template sections
    sections = template["sections"]
    for section in test_reporting_config["report_templates"]["sections"]:
        assert section in sections
    
    # Verify template record
    template_record = db_session.query(ReportTemplate).filter_by(
        template_id=result["template_id"]
    ).first()
    assert template_record is not None
    assert template_record.status == "ACTIVE"
    assert template_record.error is None

def test_aggregate_report_data(db_session, test_user, test_report_data, test_reporting_config):
    """Test report data aggregation"""
    # Aggregate data
    result = aggregate_report_data(
        data=test_report_data,
        config=test_reporting_config["data_aggregation"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify aggregation result
    assert isinstance(result, Dict)
    assert "aggregation_id" in result
    assert "timestamp" in result
    assert "data" in result
    
    # Verify aggregated data
    aggregated_data = result["data"]
    assert isinstance(aggregated_data, pd.DataFrame)
    assert len(aggregated_data) > 0
    
    # Verify aggregation methods
    for method in test_reporting_config["data_aggregation"]["methods"]:
        assert f"{method}_metrics" in result
    
    # Verify report record
    report_record = db_session.query(ReportRecord).filter_by(
        aggregation_id=result["aggregation_id"]
    ).first()
    assert report_record is not None
    assert report_record.status == "AGGREGATED"
    assert report_record.error is None

def test_schedule_report(db_session, test_user, test_reporting_config):
    """Test report scheduling"""
    # Schedule report
    result = schedule_report(
        schedule_config=test_reporting_config["report_scheduling"],
        frequency="daily",
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify schedule result
    assert isinstance(result, Dict)
    assert "schedule_id" in result
    assert "timestamp" in result
    assert "schedule" in result
    
    # Verify schedule details
    schedule = result["schedule"]
    assert "frequency" in schedule
    assert "timing" in schedule
    assert "retry_policy" in schedule
    assert "notifications" in schedule
    
    # Verify schedule record
    schedule_record = db_session.query(ReportSchedule).filter_by(
        schedule_id=result["schedule_id"]
    ).first()
    assert schedule_record is not None
    assert schedule_record.status == "SCHEDULED"
    assert schedule_record.error is None

def test_distribute_report(db_session, test_user, test_report_data, test_reporting_config):
    """Test report distribution"""
    # First generate a report
    report = generate_report(
        data=test_report_data,
        config=test_reporting_config["report_generation"],
        report_type="financial",
        format="pdf",
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Distribute report
    result = distribute_report(
        report_id=report["report_id"],
        config=test_reporting_config["report_distribution"],
        channels=["email", "dashboard"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify distribution result
    assert isinstance(result, Dict)
    assert "distribution_id" in result
    assert "timestamp" in result
    assert "distribution" in result
    
    # Verify distribution details
    distribution = result["distribution"]
    assert "channels" in distribution
    assert "status" in distribution
    assert "security" in distribution
    
    # Verify distribution channels
    channels = distribution["channels"]
    for channel in ["email", "dashboard"]:
        assert channel in channels
        assert "status" in channels[channel]
        assert "timestamp" in channels[channel]
    
    # Verify distribution record
    distribution_record = db_session.query(ReportDistribution).filter_by(
        distribution_id=result["distribution_id"]
    ).first()
    assert distribution_record is not None
    assert distribution_record.status == "DISTRIBUTED"
    assert distribution_record.error is None

def test_get_report_info(db_session, test_user, test_report_data, test_reporting_config):
    """Test report information retrieval"""
    # First generate a report
    report = generate_report(
        data=test_report_data,
        config=test_reporting_config["report_generation"],
        report_type="financial",
        format="pdf",
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Get report info
    result = get_report_info(
        report_id=report["report_id"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "report_id" in result
    assert "timestamp" in result
    assert "info" in result
    
    # Verify report info
    info = result["info"]
    assert "type" in info
    assert "format" in info
    assert "status" in info
    assert "metadata" in info
    
    # Verify report record
    report_record = db_session.query(ReportRecord).filter_by(
        report_id=result["report_id"]
    ).first()
    assert report_record is not None
    assert report_record.status == "RETRIEVED"
    assert report_record.error is None

def test_list_reports(db_session, test_user, test_report_data, test_reporting_config):
    """Test report listing"""
    # Generate multiple reports
    for _ in range(5):
        generate_report(
            data=test_report_data,
            config=test_reporting_config["report_generation"],
            report_type="financial",
            format="pdf",
            user_id=test_user.user_id,
            db_session=db_session
        )
    
    # List reports
    result = list_reports(
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify listing result
    assert isinstance(result, Dict)
    assert "timestamp" in result
    assert "reports" in result
    
    # Verify reports list
    reports = result["reports"]
    assert isinstance(reports, List)
    assert len(reports) == 5
    
    # Verify report details
    for report in reports:
        assert "report_id" in report
        assert "timestamp" in report
        assert "type" in report
        assert "status" in report

def test_delete_report(db_session, test_user, test_report_data, test_reporting_config):
    """Test report deletion"""
    # First generate a report
    report = generate_report(
        data=test_report_data,
        config=test_reporting_config["report_generation"],
        report_type="financial",
        format="pdf",
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Delete report
    result = delete_report(
        report_id=report["report_id"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify deletion result
    assert isinstance(result, Dict)
    assert "deletion_id" in result
    assert "timestamp" in result
    assert "status" in result
    
    # Verify status
    assert result["status"] == "DELETED"
    
    # Verify report record
    report_record = db_session.query(ReportRecord).filter_by(
        report_id=report["report_id"]
    ).first()
    assert report_record is not None
    assert report_record.status == "DELETED"
    assert report_record.error is None

def test_reporting_error_handling(db_session, test_user):
    """Test reporting error handling"""
    # Invalid report configuration
    with pytest.raises(ReportingError) as excinfo:
        generate_report(
            data=pd.DataFrame(),
            config={},
            report_type="invalid_type",
            format="invalid_format",
            user_id=test_user.user_id,
            db_session=db_session
        )
    assert "Invalid report configuration" in str(excinfo.value)
    
    # Invalid template configuration
    with pytest.raises(ReportingError) as excinfo:
        create_report_template(
            template_type="invalid_type",
            config={},
            user_id=test_user.user_id,
            db_session=db_session
        )
    assert "Invalid template configuration" in str(excinfo.value)
    
    # Invalid aggregation configuration
    with pytest.raises(ReportingError) as excinfo:
        aggregate_report_data(
            data=pd.DataFrame(),
            config={},
            user_id=test_user.user_id,
            db_session=db_session
        )
    assert "Invalid aggregation configuration" in str(excinfo.value)
    
    # Invalid schedule configuration
    with pytest.raises(ReportingError) as excinfo:
        schedule_report(
            schedule_config={},
            frequency="invalid_frequency",
            user_id=test_user.user_id,
            db_session=db_session
        )
    assert "Invalid schedule configuration" in str(excinfo.value)
    
    # Invalid distribution configuration
    with pytest.raises(ReportingError) as excinfo:
        distribute_report(
            report_id="invalid_id",
            config={},
            channels=["invalid_channel"],
            user_id=test_user.user_id,
            db_session=db_session
        )
    assert "Invalid distribution configuration" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBReportingError) as excinfo:
        generate_report(
            data=pd.DataFrame(),
            config={},
            report_type="financial",
            format="pdf",
            user_id=test_user.user_id,
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 