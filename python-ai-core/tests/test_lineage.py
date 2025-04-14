import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import uuid
from unittest.mock import patch, MagicMock

from core.lineage import (
    track_lineage,
    map_data_flow,
    analyze_impact,
    manage_dependencies,
    maintain_audit_trail,
    get_lineage_info,
    get_upstream_dependencies,
    get_downstream_dependencies,
    LineageError
)
from database.models import User, LineageRecord, DependencyRecord, AuditRecord
from database.exceptions import LineageError as DBLineageError

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
def test_lineage_config():
    """Create test lineage configuration"""
    return {
        "tracking": {
            "enabled": True,
            "track_sources": True,
            "track_transformations": True,
            "track_destinations": True,
            "track_metadata": True,
            "track_operations": True
        },
        "data_flow": {
            "enabled": True,
            "track_relationships": True,
            "track_dependencies": True,
            "track_versions": True,
            "track_changes": True
        },
        "impact_analysis": {
            "enabled": True,
            "analyze_upstream": True,
            "analyze_downstream": True,
            "analyze_dependencies": True,
            "analyze_metadata": True
        },
        "dependencies": {
            "enabled": True,
            "track_direct": True,
            "track_indirect": True,
            "track_versions": True,
            "track_metadata": True
        },
        "audit": {
            "enabled": True,
            "track_operations": True,
            "track_changes": True,
            "track_users": True,
            "track_timestamps": True,
            "retention_period": "1_year"
        }
    }

def test_track_lineage(db_session, test_user, test_data, test_lineage_config):
    """Test lineage tracking"""
    # Track lineage
    result = track_lineage(
        data=test_data,
        operation="data_transformation",
        details={
            "source": "raw_data",
            "transformation": "normalization",
            "target": "normalized_data"
        },
        config=test_lineage_config["tracking"],
        db_session=db_session
    )
    
    # Verify lineage result
    assert isinstance(result, Dict)
    assert "lineage_id" in result
    assert "timestamp" in result
    assert "operation" in result
    assert "details" in result
    
    # Verify lineage details
    details = result["details"]
    assert "source" in details
    assert "transformation" in details
    assert "target" in details
    assert "metadata" in details
    
    # Verify lineage record
    lineage_record = db_session.query(LineageRecord).filter_by(
        lineage_id=result["lineage_id"]
    ).first()
    assert lineage_record is not None
    assert lineage_record.status == "RECORDED"
    assert lineage_record.error is None

def test_map_data_flow(db_session, test_user, test_data, test_lineage_config):
    """Test data flow mapping"""
    # Map data flow
    result = map_data_flow(
        data=test_data,
        flow_config=test_lineage_config["data_flow"],
        db_session=db_session
    )
    
    # Verify flow mapping result
    assert isinstance(result, Dict)
    assert "flow_id" in result
    assert "timestamp" in result
    assert "flow_map" in result
    
    # Verify flow map
    flow_map = result["flow_map"]
    assert "nodes" in flow_map
    assert "edges" in flow_map
    assert "metadata" in flow_map
    
    # Verify nodes
    nodes = flow_map["nodes"]
    assert isinstance(nodes, List)
    for node in nodes:
        assert "id" in node
        assert "type" in node
        assert "metadata" in node
    
    # Verify edges
    edges = flow_map["edges"]
    assert isinstance(edges, List)
    for edge in edges:
        assert "source" in edge
        assert "target" in edge
        assert "type" in edge
        assert "metadata" in edge
    
    # Verify lineage record
    lineage_record = db_session.query(LineageRecord).filter_by(
        flow_id=result["flow_id"]
    ).first()
    assert lineage_record is not None
    assert lineage_record.status == "MAPPED"
    assert lineage_record.error is None

def test_analyze_impact(db_session, test_user, test_data, test_lineage_config):
    """Test impact analysis"""
    # Analyze impact
    result = analyze_impact(
        data=test_data,
        impact_config=test_lineage_config["impact_analysis"],
        db_session=db_session
    )
    
    # Verify impact analysis result
    assert isinstance(result, Dict)
    assert "impact_id" in result
    assert "timestamp" in result
    assert "analysis" in result
    
    # Verify analysis
    analysis = result["analysis"]
    assert "upstream" in analysis
    assert "downstream" in analysis
    assert "dependencies" in analysis
    assert "metadata" in analysis
    
    # Verify upstream analysis
    upstream = analysis["upstream"]
    assert isinstance(upstream, Dict)
    assert "direct_sources" in upstream
    assert "indirect_sources" in upstream
    assert "impact_score" in upstream
    
    # Verify downstream analysis
    downstream = analysis["downstream"]
    assert isinstance(downstream, Dict)
    assert "direct_targets" in downstream
    assert "indirect_targets" in downstream
    assert "impact_score" in downstream
    
    # Verify lineage record
    lineage_record = db_session.query(LineageRecord).filter_by(
        impact_id=result["impact_id"]
    ).first()
    assert lineage_record is not None
    assert lineage_record.status == "ANALYZED"
    assert lineage_record.error is None

def test_manage_dependencies(db_session, test_user, test_data, test_lineage_config):
    """Test dependency management"""
    # Manage dependencies
    result = manage_dependencies(
        data=test_data,
        dependency_config=test_lineage_config["dependencies"],
        db_session=db_session
    )
    
    # Verify dependency management result
    assert isinstance(result, Dict)
    assert "dependency_id" in result
    assert "timestamp" in result
    assert "dependencies" in result
    
    # Verify dependencies
    dependencies = result["dependencies"]
    assert "direct" in dependencies
    assert "indirect" in dependencies
    assert "versions" in dependencies
    assert "metadata" in dependencies
    
    # Verify direct dependencies
    direct = dependencies["direct"]
    assert isinstance(direct, List)
    for dep in direct:
        assert "source" in dep
        assert "target" in dep
        assert "type" in dep
        assert "metadata" in dep
    
    # Verify indirect dependencies
    indirect = dependencies["indirect"]
    assert isinstance(indirect, List)
    for dep in indirect:
        assert "source" in dep
        assert "target" in dep
        assert "path" in dep
        assert "metadata" in dep
    
    # Verify dependency record
    dependency_record = db_session.query(DependencyRecord).filter_by(
        dependency_id=result["dependency_id"]
    ).first()
    assert dependency_record is not None
    assert dependency_record.status == "MANAGED"
    assert dependency_record.error is None

def test_maintain_audit_trail(db_session, test_user, test_data, test_lineage_config):
    """Test audit trail maintenance"""
    # Maintain audit trail
    result = maintain_audit_trail(
        data=test_data,
        audit_config=test_lineage_config["audit"],
        operation="data_transformation",
        user_id=test_user.user_id,
        details={
            "transformation": "normalization",
            "parameters": {"method": "minmax"}
        },
        db_session=db_session
    )
    
    # Verify audit trail result
    assert isinstance(result, Dict)
    assert "audit_id" in result
    assert "timestamp" in result
    assert "audit_trail" in result
    
    # Verify audit trail
    audit_trail = result["audit_trail"]
    assert "operation" in audit_trail
    assert "user_id" in audit_trail
    assert "details" in audit_trail
    assert "metadata" in audit_trail
    
    # Verify audit record
    audit_record = db_session.query(AuditRecord).filter_by(
        audit_id=result["audit_id"]
    ).first()
    assert audit_record is not None
    assert audit_record.status == "RECORDED"
    assert audit_record.error is None

def test_get_lineage_info(db_session, test_user, test_data, test_lineage_config):
    """Test lineage information retrieval"""
    # Get lineage info
    result = get_lineage_info(
        data_id="test_data",
        config=test_lineage_config,
        db_session=db_session
    )
    
    # Verify lineage info result
    assert isinstance(result, Dict)
    assert "lineage_id" in result
    assert "timestamp" in result
    assert "info" in result
    
    # Verify info content
    info = result["info"]
    assert "sources" in info
    assert "transformations" in info
    assert "destinations" in info
    assert "metadata" in info
    
    # Verify sources
    sources = info["sources"]
    assert isinstance(sources, List)
    for source in sources:
        assert "id" in source
        assert "type" in source
        assert "metadata" in source
    
    # Verify transformations
    transformations = info["transformations"]
    assert isinstance(transformations, List)
    for transform in transformations:
        assert "id" in transform
        assert "type" in transform
        assert "metadata" in transform
    
    # Verify lineage record
    lineage_record = db_session.query(LineageRecord).filter_by(
        lineage_id=result["lineage_id"]
    ).first()
    assert lineage_record is not None
    assert lineage_record.status == "RETRIEVED"
    assert lineage_record.error is None

def test_get_upstream_dependencies(db_session, test_user, test_data, test_lineage_config):
    """Test upstream dependency retrieval"""
    # Get upstream dependencies
    result = get_upstream_dependencies(
        data_id="test_data",
        config=test_lineage_config["dependencies"],
        db_session=db_session
    )
    
    # Verify upstream dependencies result
    assert isinstance(result, Dict)
    assert "dependency_id" in result
    assert "timestamp" in result
    assert "dependencies" in result
    
    # Verify dependencies
    dependencies = result["dependencies"]
    assert "direct" in dependencies
    assert "indirect" in dependencies
    assert "metadata" in dependencies
    
    # Verify direct dependencies
    direct = dependencies["direct"]
    assert isinstance(direct, List)
    for dep in direct:
        assert "id" in dep
        assert "type" in dep
        assert "metadata" in dep
    
    # Verify indirect dependencies
    indirect = dependencies["indirect"]
    assert isinstance(indirect, List)
    for dep in indirect:
        assert "id" in dep
        assert "type" in dep
        assert "path" in dep
        assert "metadata" in dep
    
    # Verify dependency record
    dependency_record = db_session.query(DependencyRecord).filter_by(
        dependency_id=result["dependency_id"]
    ).first()
    assert dependency_record is not None
    assert dependency_record.status == "RETRIEVED"
    assert dependency_record.error is None

def test_get_downstream_dependencies(db_session, test_user, test_data, test_lineage_config):
    """Test downstream dependency retrieval"""
    # Get downstream dependencies
    result = get_downstream_dependencies(
        data_id="test_data",
        config=test_lineage_config["dependencies"],
        db_session=db_session
    )
    
    # Verify downstream dependencies result
    assert isinstance(result, Dict)
    assert "dependency_id" in result
    assert "timestamp" in result
    assert "dependencies" in result
    
    # Verify dependencies
    dependencies = result["dependencies"]
    assert "direct" in dependencies
    assert "indirect" in dependencies
    assert "metadata" in dependencies
    
    # Verify direct dependencies
    direct = dependencies["direct"]
    assert isinstance(direct, List)
    for dep in direct:
        assert "id" in dep
        assert "type" in dep
        assert "metadata" in dep
    
    # Verify indirect dependencies
    indirect = dependencies["indirect"]
    assert isinstance(indirect, List)
    for dep in indirect:
        assert "id" in dep
        assert "type" in dep
        assert "path" in dep
        assert "metadata" in dep
    
    # Verify dependency record
    dependency_record = db_session.query(DependencyRecord).filter_by(
        dependency_id=result["dependency_id"]
    ).first()
    assert dependency_record is not None
    assert dependency_record.status == "RETRIEVED"
    assert dependency_record.error is None

def test_lineage_error_handling(db_session, test_user):
    """Test lineage error handling"""
    # Invalid tracking configuration
    with pytest.raises(LineageError) as excinfo:
        track_lineage(
            data=pd.DataFrame(),
            operation="invalid",
            details={},
            config={},
            db_session=db_session
        )
    assert "Invalid tracking configuration" in str(excinfo.value)
    
    # Invalid flow configuration
    with pytest.raises(LineageError) as excinfo:
        map_data_flow(
            data=pd.DataFrame(),
            flow_config={},
            db_session=db_session
        )
    assert "Invalid flow configuration" in str(excinfo.value)
    
    # Invalid impact configuration
    with pytest.raises(LineageError) as excinfo:
        analyze_impact(
            data=pd.DataFrame(),
            impact_config={},
            db_session=db_session
        )
    assert "Invalid impact configuration" in str(excinfo.value)
    
    # Invalid dependency configuration
    with pytest.raises(LineageError) as excinfo:
        manage_dependencies(
            data=pd.DataFrame(),
            dependency_config={},
            db_session=db_session
        )
    assert "Invalid dependency configuration" in str(excinfo.value)
    
    # Invalid audit configuration
    with pytest.raises(LineageError) as excinfo:
        maintain_audit_trail(
            data=pd.DataFrame(),
            audit_config={},
            operation="invalid",
            user_id="invalid",
            details={},
            db_session=db_session
        )
    assert "Invalid audit configuration" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBLineageError) as excinfo:
        track_lineage(
            data=pd.DataFrame(),
            operation="test",
            details={},
            config={},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 