import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import uuid
import hashlib
from unittest.mock import patch, MagicMock

from core.compliance import (
    validate_privacy_requirements,
    anonymize_data,
    mask_sensitive_data,
    verify_gdpr_compliance,
    verify_ccpa_compliance,
    verify_hipaa_compliance,
    track_data_lineage,
    manage_data_retention,
    manage_user_consent,
    verify_consent,
    revoke_consent,
    generate_compliance_report,
    track_violations,
    ComplianceError
)
from database.models import User, ComplianceRecord, ConsentRecord, ViolationRecord
from database.exceptions import ComplianceError as DBComplianceError

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
        last_login=datetime.utcnow(),
        country="US",
        data_region="us-west"
    )
    db_session.add(user)
    db_session.commit()
    return user

@pytest.fixture
def test_sensitive_data():
    """Create test sensitive data"""
    return pd.DataFrame({
        "user_id": [f"user_{i}" for i in range(100)],
        "email": [f"user_{i}@example.com" for i in range(100)],
        "ssn": [f"123-45-{str(i).zfill(4)}" for i in range(100)],
        "credit_card": [f"4111-1111-1111-{str(i).zfill(4)}" for i in range(100)],
        "address": [f"123 Main St, City {i}, State" for i in range(100)],
        "phone": [f"(555) 123-{str(i).zfill(4)}" for i in range(100)],
        "medical_condition": [f"Condition {i}" for i in range(100)],
        "account_balance": np.random.uniform(1000, 10000, 100),
        "trading_performance": np.random.uniform(0.8, 0.9, 100),
        "risk_score": np.random.uniform(0.1, 0.5, 100)
    })

@pytest.fixture
def test_compliance_config():
    """Create test compliance configuration"""
    return {
        "privacy_requirements": {
            "pii_fields": [
                "email",
                "ssn",
                "credit_card",
                "address",
                "phone"
            ],
            "sensitive_fields": [
                "medical_condition",
                "account_balance"
            ],
            "anonymization": {
                "method": "k_anonymity",
                "k_value": 5,
                "sensitive_attributes": ["medical_condition"],
                "quasi_identifiers": ["age", "gender", "zipcode"]
            },
            "masking": {
                "email": "partial",
                "ssn": "full",
                "credit_card": "last4",
                "phone": "partial"
            }
        },
        "regulatory_compliance": {
            "gdpr": {
                "enabled": True,
                "requirements": [
                    "data_minimization",
                    "purpose_limitation",
                    "storage_limitation",
                    "accuracy",
                    "integrity_confidentiality"
                ],
                "data_subject_rights": [
                    "access",
                    "rectification",
                    "erasure",
                    "portability",
                    "object"
                ]
            },
            "ccpa": {
                "enabled": True,
                "requirements": [
                    "notice",
                    "access",
                    "deletion",
                    "opt_out_sale",
                    "non_discrimination"
                ]
            },
            "hipaa": {
                "enabled": True,
                "requirements": [
                    "privacy_rule",
                    "security_rule",
                    "enforcement_rule",
                    "breach_notification"
                ],
                "phi_fields": [
                    "medical_condition"
                ]
            }
        },
        "data_governance": {
            "classification": {
                "public": ["trading_performance", "risk_score"],
                "internal": ["account_balance"],
                "confidential": ["email", "phone", "address"],
                "restricted": ["ssn", "credit_card", "medical_condition"]
            },
            "retention": {
                "public": "7_years",
                "internal": "5_years",
                "confidential": "3_years",
                "restricted": "1_year"
            },
            "lineage": {
                "track_sources": True,
                "track_transformations": True,
                "track_usage": True,
                "track_exports": True
            }
        },
        "consent_management": {
            "purposes": [
                "account_management",
                "risk_assessment",
                "performance_analysis",
                "marketing"
            ],
            "granularity": "field_level",
            "expiration": "1_year",
            "revocation": {
                "allowed": True,
                "grace_period_days": 30
            },
            "record_keeping": {
                "store_history": True,
                "audit_trail": True
            }
        },
        "compliance_reporting": {
            "audit_frequency": "monthly",
            "violation_tracking": {
                "enabled": True,
                "severity_levels": ["low", "medium", "high", "critical"],
                "auto_remediation": True
            },
            "reports": [
                "privacy_impact_assessment",
                "data_protection_impact_assessment",
                "consent_audit",
                "violation_summary",
                "remediation_status"
            ]
        }
    }

def test_validate_privacy_requirements(db_session, test_user, test_sensitive_data, test_compliance_config):
    """Test privacy requirements validation"""
    # Validate privacy requirements
    result = validate_privacy_requirements(
        data=test_sensitive_data,
        privacy_config=test_compliance_config["privacy_requirements"],
        db_session=db_session
    )
    
    # Verify validation result
    assert isinstance(result, Dict)
    assert "valid" in result
    assert result["valid"] is True
    assert "validation_id" in result
    assert "details" in result
    
    # Verify validation details
    details = result["details"]
    assert "pii_fields_found" in details
    assert all(field in details["pii_fields_found"] for field in test_compliance_config["privacy_requirements"]["pii_fields"])
    assert "sensitive_fields_found" in details
    assert all(field in details["sensitive_fields_found"] for field in test_compliance_config["privacy_requirements"]["sensitive_fields"])
    
    # Verify compliance record
    compliance_record = db_session.query(ComplianceRecord).filter_by(
        validation_id=result["validation_id"]
    ).first()
    assert compliance_record is not None
    assert compliance_record.status == "COMPLIANT"
    assert compliance_record.error is None

def test_anonymize_data(db_session, test_user, test_sensitive_data, test_compliance_config):
    """Test data anonymization"""
    # Anonymize data
    result = anonymize_data(
        data=test_sensitive_data,
        anonymization_config=test_compliance_config["privacy_requirements"]["anonymization"],
        db_session=db_session
    )
    
    # Verify anonymization result
    assert isinstance(result, Dict)
    assert "anonymized_data" in result
    assert "anonymization_id" in result
    assert "metadata" in result
    
    # Verify anonymized data
    anonymized_data = result["anonymized_data"]
    assert isinstance(anonymized_data, pd.DataFrame)
    assert len(anonymized_data) == len(test_sensitive_data)
    
    # Verify k-anonymity
    k = test_compliance_config["privacy_requirements"]["anonymization"]["k_value"]
    for _, group in anonymized_data.groupby(test_compliance_config["privacy_requirements"]["anonymization"]["quasi_identifiers"]):
        assert len(group) >= k
    
    # Verify sensitive attributes are preserved
    sensitive_attrs = test_compliance_config["privacy_requirements"]["anonymization"]["sensitive_attributes"]
    for attr in sensitive_attrs:
        assert attr in anonymized_data.columns
    
    # Verify compliance record
    compliance_record = db_session.query(ComplianceRecord).filter_by(
        anonymization_id=result["anonymization_id"]
    ).first()
    assert compliance_record is not None
    assert compliance_record.status == "COMPLETED"
    assert compliance_record.error is None

def test_mask_sensitive_data(db_session, test_user, test_sensitive_data, test_compliance_config):
    """Test sensitive data masking"""
    # Mask sensitive data
    result = mask_sensitive_data(
        data=test_sensitive_data,
        masking_config=test_compliance_config["privacy_requirements"]["masking"],
        db_session=db_session
    )
    
    # Verify masking result
    assert isinstance(result, Dict)
    assert "masked_data" in result
    assert "masking_id" in result
    assert "metadata" in result
    
    # Verify masked data
    masked_data = result["masked_data"]
    assert isinstance(masked_data, pd.DataFrame)
    assert len(masked_data) == len(test_sensitive_data)
    
    # Verify masking rules
    # Email masking (partial)
    assert all(email.split("@")[0].endswith("***") for email in masked_data["email"])
    
    # SSN masking (full)
    assert all(ssn == "***-**-****" for ssn in masked_data["ssn"])
    
    # Credit card masking (last4)
    assert all(cc.startswith("****-****-****-") and len(cc.split("-")[-1]) == 4 
              for cc in masked_data["credit_card"])
    
    # Phone masking (partial)
    assert all(phone.startswith("(***) ***-") and len(phone.split("-")[-1]) == 4 
              for phone in masked_data["phone"])
    
    # Verify compliance record
    compliance_record = db_session.query(ComplianceRecord).filter_by(
        masking_id=result["masking_id"]
    ).first()
    assert compliance_record is not None
    assert compliance_record.status == "COMPLETED"
    assert compliance_record.error is None

def test_verify_gdpr_compliance(db_session, test_user, test_sensitive_data, test_compliance_config):
    """Test GDPR compliance verification"""
    # Verify GDPR compliance
    result = verify_gdpr_compliance(
        data=test_sensitive_data,
        gdpr_config=test_compliance_config["regulatory_compliance"]["gdpr"],
        db_session=db_session
    )
    
    # Verify compliance result
    assert isinstance(result, Dict)
    assert "compliant" in result
    assert result["compliant"] is True
    assert "verification_id" in result
    assert "details" in result
    
    # Verify compliance details
    details = result["details"]
    assert "requirements_met" in details
    assert all(req in details["requirements_met"] 
              for req in test_compliance_config["regulatory_compliance"]["gdpr"]["requirements"])
    assert "data_subject_rights_supported" in details
    assert all(right in details["data_subject_rights_supported"] 
              for right in test_compliance_config["regulatory_compliance"]["gdpr"]["data_subject_rights"])
    
    # Verify compliance record
    compliance_record = db_session.query(ComplianceRecord).filter_by(
        verification_id=result["verification_id"]
    ).first()
    assert compliance_record is not None
    assert compliance_record.status == "COMPLIANT"
    assert compliance_record.error is None

def test_track_data_lineage(db_session, test_user, test_sensitive_data, test_compliance_config):
    """Test data lineage tracking"""
    # Track data lineage
    result = track_data_lineage(
        data=test_sensitive_data,
        lineage_config=test_compliance_config["data_governance"]["lineage"],
        operation="data_transformation",
        details={
            "source": "raw_data",
            "transformation": "anonymization",
            "target": "anonymized_data"
        },
        db_session=db_session
    )
    
    # Verify lineage result
    assert isinstance(result, Dict)
    assert "lineage_id" in result
    assert "timestamp" in result
    assert "operation" in result
    assert "details" in result
    
    # Verify lineage details
    assert result["operation"] == "data_transformation"
    assert "source" in result["details"]
    assert "transformation" in result["details"]
    assert "target" in result["details"]
    
    # Verify compliance record
    compliance_record = db_session.query(ComplianceRecord).filter_by(
        lineage_id=result["lineage_id"]
    ).first()
    assert compliance_record is not None
    assert compliance_record.status == "RECORDED"
    assert compliance_record.error is None

def test_manage_user_consent(db_session, test_user, test_compliance_config):
    """Test user consent management"""
    # Grant consent
    consent_data = {
        "user_id": test_user.user_id,
        "purposes": ["account_management", "risk_assessment"],
        "fields": ["email", "trading_performance", "risk_score"],
        "expiration": datetime.utcnow() + timedelta(days=365)
    }
    
    result = manage_user_consent(
        consent_data=consent_data,
        consent_config=test_compliance_config["consent_management"],
        operation="grant",
        db_session=db_session
    )
    
    # Verify consent result
    assert isinstance(result, Dict)
    assert "consent_id" in result
    assert "status" in result
    assert result["status"] == "GRANTED"
    assert "details" in result
    
    # Verify consent record
    consent_record = db_session.query(ConsentRecord).filter_by(
        consent_id=result["consent_id"]
    ).first()
    assert consent_record is not None
    assert consent_record.status == "ACTIVE"
    assert consent_record.error is None
    
    # Verify consent
    verify_result = verify_consent(
        user_id=test_user.user_id,
        purpose="account_management",
        fields=["email", "trading_performance"],
        db_session=db_session
    )
    
    assert isinstance(verify_result, Dict)
    assert "verified" in verify_result
    assert verify_result["verified"] is True
    
    # Revoke consent
    revoke_result = revoke_consent(
        consent_id=result["consent_id"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    assert isinstance(revoke_result, Dict)
    assert "status" in revoke_result
    assert revoke_result["status"] == "REVOKED"
    
    # Verify revoked consent
    verify_result = verify_consent(
        user_id=test_user.user_id,
        purpose="account_management",
        fields=["email", "trading_performance"],
        db_session=db_session
    )
    
    assert isinstance(verify_result, Dict)
    assert "verified" in verify_result
    assert verify_result["verified"] is False

def test_generate_compliance_report(db_session, test_user, test_compliance_config):
    """Test compliance report generation"""
    # Generate compliance report
    result = generate_compliance_report(
        report_type="privacy_impact_assessment",
        start_time=datetime.utcnow() - timedelta(days=30),
        end_time=datetime.utcnow(),
        config=test_compliance_config["compliance_reporting"],
        db_session=db_session
    )
    
    # Verify report result
    assert isinstance(result, Dict)
    assert "report_id" in result
    assert "report_type" in result
    assert "timestamp" in result
    assert "content" in result
    
    # Verify report content
    content = result["content"]
    assert "summary" in content
    assert "findings" in content
    assert "recommendations" in content
    assert "metrics" in content
    
    # Verify compliance record
    compliance_record = db_session.query(ComplianceRecord).filter_by(
        report_id=result["report_id"]
    ).first()
    assert compliance_record is not None
    assert compliance_record.status == "GENERATED"
    assert compliance_record.error is None

def test_track_violations(db_session, test_user, test_compliance_config):
    """Test compliance violation tracking"""
    # Create violation
    violation_data = {
        "type": "unauthorized_access",
        "severity": "high",
        "description": "Unauthorized access attempt to restricted data",
        "affected_data": ["ssn", "credit_card"],
        "timestamp": datetime.utcnow(),
        "source_ip": "192.168.1.100",
        "user_id": test_user.user_id
    }
    
    result = track_violations(
        violation_data=violation_data,
        tracking_config=test_compliance_config["compliance_reporting"]["violation_tracking"],
        db_session=db_session
    )
    
    # Verify violation result
    assert isinstance(result, Dict)
    assert "violation_id" in result
    assert "status" in result
    assert "timestamp" in result
    assert "details" in result
    
    # Verify violation record
    violation_record = db_session.query(ViolationRecord).filter_by(
        violation_id=result["violation_id"]
    ).first()
    assert violation_record is not None
    assert violation_record.status == "RECORDED"
    assert violation_record.severity == "high"
    assert violation_record.error is None

def test_compliance_error_handling(db_session, test_user):
    """Test compliance error handling"""
    # Invalid privacy requirements
    with pytest.raises(ComplianceError) as excinfo:
        validate_privacy_requirements(
            data=pd.DataFrame(),
            privacy_config={},
            db_session=db_session
        )
    assert "Invalid privacy configuration" in str(excinfo.value)
    
    # Invalid anonymization
    with pytest.raises(ComplianceError) as excinfo:
        anonymize_data(
            data=pd.DataFrame(),
            anonymization_config={},
            db_session=db_session
        )
    assert "Invalid anonymization configuration" in str(excinfo.value)
    
    # Invalid masking
    with pytest.raises(ComplianceError) as excinfo:
        mask_sensitive_data(
            data=pd.DataFrame(),
            masking_config={},
            db_session=db_session
        )
    assert "Invalid masking configuration" in str(excinfo.value)
    
    # Invalid consent
    with pytest.raises(ComplianceError) as excinfo:
        manage_user_consent(
            consent_data={},
            consent_config={},
            operation="invalid",
            db_session=db_session
        )
    assert "Invalid consent operation" in str(excinfo.value)
    
    # Invalid report type
    with pytest.raises(ComplianceError) as excinfo:
        generate_compliance_report(
            report_type="invalid",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            config={},
            db_session=db_session
        )
    assert "Invalid report type" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBComplianceError) as excinfo:
        validate_privacy_requirements(
            data=pd.DataFrame(),
            privacy_config={},
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 