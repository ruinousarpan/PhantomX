import pytest
import json
import jwt
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

from core.api import (
    create_endpoint,
    handle_request,
    authenticate_request,
    authorize_request,
    rate_limit_request,
    validate_request,
    process_response,
    version_endpoint,
    document_endpoint,
    get_api_info,
    list_endpoints,
    delete_endpoint,
    APIError
)
from database.models import User, APIEndpoint, APIRequest, APIResponse, RateLimit
from database.exceptions import APIError as DBAPIError

@pytest.fixture
def test_user(db_session):
    """Create test user"""
    user = User(
        user_id="test_user",
        username="testuser",
        email="test@example.com",
        first_name="Test",
        last_name="User",
        role="API_USER",
        status="ACTIVE",
        email_verified=True,
        created_at=datetime.utcnow(),
        last_login=datetime.utcnow()
    )
    db_session.add(user)
    db_session.commit()
    return user

@pytest.fixture
def test_api_config():
    """Create test API configuration"""
    return {
        "endpoint_config": {
            "enabled": True,
            "methods": ["GET", "POST", "PUT", "DELETE"],
            "paths": [
                "/api/v1/data",
                "/api/v1/analytics",
                "/api/v1/reports"
            ],
            "versions": ["v1", "v2"],
            "formats": ["json", "xml"],
            "compression": ["gzip", "deflate"],
            "timeout": 30,
            "max_payload": 10485760  # 10MB
        },
        "auth_config": {
            "enabled": True,
            "methods": [
                "jwt",
                "api_key",
                "oauth2"
            ],
            "token_expiry": "1h",
            "refresh_enabled": True,
            "scope_required": True,
            "roles": [
                "reader",
                "writer",
                "admin"
            ]
        },
        "rate_limit_config": {
            "enabled": True,
            "strategies": [
                "fixed_window",
                "sliding_window",
                "token_bucket"
            ],
            "limits": {
                "requests_per_second": 10,
                "requests_per_minute": 100,
                "requests_per_hour": 1000
            },
            "headers": {
                "remaining": "X-RateLimit-Remaining",
                "reset": "X-RateLimit-Reset",
                "total": "X-RateLimit-Total"
            }
        },
        "validation_config": {
            "enabled": True,
            "schemas": {
                "request": "openapi",
                "response": "json_schema"
            },
            "validations": [
                "type",
                "format",
                "required",
                "enum",
                "range"
            ],
            "sanitization": True,
            "strict_mode": True
        },
        "response_config": {
            "enabled": True,
            "formats": [
                "json",
                "xml",
                "csv"
            ],
            "compression": [
                "gzip",
                "deflate"
            ],
            "pagination": {
                "enabled": True,
                "max_page_size": 100,
                "default_page_size": 20
            },
            "caching": {
                "enabled": True,
                "ttl": 300,
                "strategy": "lru"
            }
        },
        "versioning_config": {
            "enabled": True,
            "strategies": [
                "url",
                "header",
                "param"
            ],
            "versions": {
                "current": "v1",
                "supported": ["v1", "v2"],
                "deprecated": []
            },
            "headers": {
                "version": "X-API-Version",
                "deprecated": "X-API-Deprecated"
            }
        },
        "documentation_config": {
            "enabled": True,
            "formats": [
                "openapi",
                "swagger",
                "raml"
            ],
            "output": [
                "json",
                "yaml",
                "html"
            ],
            "features": [
                "examples",
                "schemas",
                "security",
                "responses"
            ],
            "hosting": {
                "enabled": True,
                "path": "/api/docs",
                "ui": "swagger_ui"
            }
        }
    }

@pytest.fixture
def test_endpoint_data():
    """Create test endpoint data"""
    return {
        "path": "/api/v1/data",
        "method": "GET",
        "handler": "get_data",
        "auth_required": True,
        "rate_limit": {
            "requests_per_minute": 60,
            "burst_size": 10
        },
        "params": {
            "required": ["dataset_id"],
            "optional": ["fields", "filter", "sort"]
        },
        "responses": {
            "200": {"description": "Success"},
            "400": {"description": "Bad Request"},
            "401": {"description": "Unauthorized"},
            "429": {"description": "Too Many Requests"}
        }
    }

def test_create_endpoint(db_session, test_user, test_api_config, test_endpoint_data):
    """Test endpoint creation"""
    # Create endpoint
    result = create_endpoint(
        endpoint_data=test_endpoint_data,
        config=test_api_config["endpoint_config"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify endpoint result
    assert isinstance(result, Dict)
    assert "endpoint_id" in result
    assert "timestamp" in result
    assert "endpoint" in result
    
    # Verify endpoint details
    endpoint = result["endpoint"]
    assert endpoint["path"] == test_endpoint_data["path"]
    assert endpoint["method"] == test_endpoint_data["method"]
    assert endpoint["auth_required"] == test_endpoint_data["auth_required"]
    
    # Verify endpoint record
    endpoint_record = db_session.query(APIEndpoint).filter_by(
        endpoint_id=result["endpoint_id"]
    ).first()
    assert endpoint_record is not None
    assert endpoint_record.status == "ACTIVE"
    assert endpoint_record.error is None

def test_handle_request(db_session, test_user, test_api_config, test_endpoint_data):
    """Test request handling"""
    # Create mock request
    request_data = {
        "method": "GET",
        "path": "/api/v1/data",
        "params": {"dataset_id": "test_dataset"},
        "headers": {"Authorization": "Bearer test_token"}
    }
    
    # Handle request
    result = handle_request(
        request_data=request_data,
        config=test_api_config,
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify request result
    assert isinstance(result, Dict)
    assert "request_id" in result
    assert "timestamp" in result
    assert "response" in result
    
    # Verify request handling
    response = result["response"]
    assert "status_code" in response
    assert "headers" in response
    assert "body" in response
    
    # Verify request record
    request_record = db_session.query(APIRequest).filter_by(
        request_id=result["request_id"]
    ).first()
    assert request_record is not None
    assert request_record.status == "PROCESSED"
    assert request_record.error is None

def test_authenticate_request(db_session, test_user, test_api_config):
    """Test request authentication"""
    # Create test token
    token = jwt.encode(
        {
            "user_id": test_user.user_id,
            "exp": datetime.utcnow() + timedelta(hours=1)
        },
        "test_secret",
        algorithm="HS256"
    )
    
    # Create auth request
    auth_data = {
        "token": token,
        "method": "jwt",
        "scope": ["read"]
    }
    
    # Authenticate request
    result = authenticate_request(
        auth_data=auth_data,
        config=test_api_config["auth_config"],
        db_session=db_session
    )
    
    # Verify authentication result
    assert isinstance(result, Dict)
    assert "auth_id" in result
    assert "timestamp" in result
    assert "auth_info" in result
    
    # Verify auth info
    auth_info = result["auth_info"]
    assert "user_id" in auth_info
    assert "token" in auth_info
    assert "scope" in auth_info
    assert "expires_at" in auth_info

def test_authorize_request(db_session, test_user, test_api_config):
    """Test request authorization"""
    # Create auth info
    auth_info = {
        "user_id": test_user.user_id,
        "role": "reader",
        "scope": ["read"]
    }
    
    # Create resource info
    resource_info = {
        "path": "/api/v1/data",
        "method": "GET",
        "required_scope": ["read"]
    }
    
    # Authorize request
    result = authorize_request(
        auth_info=auth_info,
        resource_info=resource_info,
        config=test_api_config["auth_config"],
        db_session=db_session
    )
    
    # Verify authorization result
    assert isinstance(result, Dict)
    assert "authorized" in result
    assert result["authorized"] is True
    assert "scope" in result
    assert "role" in result

def test_rate_limit_request(db_session, test_user, test_api_config):
    """Test request rate limiting"""
    # Create request info
    request_info = {
        "user_id": test_user.user_id,
        "endpoint": "/api/v1/data",
        "method": "GET",
        "timestamp": datetime.utcnow()
    }
    
    # Check rate limit
    result = rate_limit_request(
        request_info=request_info,
        config=test_api_config["rate_limit_config"],
        db_session=db_session
    )
    
    # Verify rate limit result
    assert isinstance(result, Dict)
    assert "allowed" in result
    assert "remaining" in result
    assert "reset_at" in result
    assert "headers" in result
    
    # Verify rate limit record
    rate_limit_record = db_session.query(RateLimit).filter_by(
        user_id=test_user.user_id
    ).first()
    assert rate_limit_record is not None
    assert rate_limit_record.count > 0
    assert rate_limit_record.reset_at > datetime.utcnow()

def test_validate_request(db_session, test_api_config, test_endpoint_data):
    """Test request validation"""
    # Create request data
    request_data = {
        "method": "GET",
        "path": "/api/v1/data",
        "params": {
            "dataset_id": "test_dataset",
            "fields": ["id", "name"],
            "filter": {"status": "active"}
        },
        "headers": {
            "Content-Type": "application/json"
        }
    }
    
    # Validate request
    result = validate_request(
        request_data=request_data,
        endpoint_data=test_endpoint_data,
        config=test_api_config["validation_config"],
        db_session=db_session
    )
    
    # Verify validation result
    assert isinstance(result, Dict)
    assert "valid" in result
    assert result["valid"] is True
    assert "errors" in result
    assert len(result["errors"]) == 0

def test_process_response(db_session, test_api_config):
    """Test response processing"""
    # Create response data
    response_data = {
        "status_code": 200,
        "body": {
            "data": [{"id": 1, "name": "Test"}],
            "metadata": {"total": 1, "page": 1}
        },
        "headers": {
            "Content-Type": "application/json"
        }
    }
    
    # Process response
    result = process_response(
        response_data=response_data,
        config=test_api_config["response_config"],
        db_session=db_session
    )
    
    # Verify response result
    assert isinstance(result, Dict)
    assert "response_id" in result
    assert "processed_response" in result
    
    # Verify processed response
    processed = result["processed_response"]
    assert "status_code" in processed
    assert "body" in processed
    assert "headers" in processed
    
    # Verify response record
    response_record = db_session.query(APIResponse).filter_by(
        response_id=result["response_id"]
    ).first()
    assert response_record is not None
    assert response_record.status_code == 200
    assert response_record.error is None

def test_version_endpoint(db_session, test_user, test_api_config, test_endpoint_data):
    """Test endpoint versioning"""
    # Create version info
    version_info = {
        "endpoint_id": str(uuid.uuid4()),
        "version": "v2",
        "changes": ["Added new fields", "Updated response format"],
        "deprecated": False
    }
    
    # Version endpoint
    result = version_endpoint(
        endpoint_data=test_endpoint_data,
        version_info=version_info,
        config=test_api_config["versioning_config"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify version result
    assert isinstance(result, Dict)
    assert "version_id" in result
    assert "endpoint" in result
    
    # Verify versioned endpoint
    endpoint = result["endpoint"]
    assert endpoint["version"] == "v2"
    assert "v1" in endpoint["supported_versions"]
    assert not endpoint["deprecated"]

def test_document_endpoint(db_session, test_user, test_api_config, test_endpoint_data):
    """Test endpoint documentation"""
    # Document endpoint
    result = document_endpoint(
        endpoint_data=test_endpoint_data,
        config=test_api_config["documentation_config"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify documentation result
    assert isinstance(result, Dict)
    assert "doc_id" in result
    assert "documentation" in result
    
    # Verify documentation content
    docs = result["documentation"]
    assert "openapi" in docs
    assert "swagger" in docs
    assert "examples" in docs
    assert "schemas" in docs

def test_get_api_info(db_session, test_user, test_api_config, test_endpoint_data):
    """Test API information retrieval"""
    # First create an endpoint
    endpoint = create_endpoint(
        endpoint_data=test_endpoint_data,
        config=test_api_config["endpoint_config"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Get API info
    result = get_api_info(
        endpoint_id=endpoint["endpoint_id"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify info result
    assert isinstance(result, Dict)
    assert "endpoint_id" in result
    assert "info" in result
    
    # Verify API info
    info = result["info"]
    assert "path" in info
    assert "method" in info
    assert "auth_required" in info
    assert "rate_limit" in info
    assert "documentation" in info

def test_list_endpoints(db_session, test_user, test_api_config, test_endpoint_data):
    """Test endpoint listing"""
    # Create multiple endpoints
    for _ in range(3):
        create_endpoint(
            endpoint_data=test_endpoint_data,
            config=test_api_config["endpoint_config"],
            user_id=test_user.user_id,
            db_session=db_session
        )
    
    # List endpoints
    result = list_endpoints(
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify listing result
    assert isinstance(result, Dict)
    assert "endpoints" in result
    
    # Verify endpoints list
    endpoints = result["endpoints"]
    assert isinstance(endpoints, List)
    assert len(endpoints) == 3
    
    # Verify endpoint details
    for endpoint in endpoints:
        assert "endpoint_id" in endpoint
        assert "path" in endpoint
        assert "method" in endpoint
        assert "status" in endpoint

def test_delete_endpoint(db_session, test_user, test_api_config, test_endpoint_data):
    """Test endpoint deletion"""
    # First create an endpoint
    endpoint = create_endpoint(
        endpoint_data=test_endpoint_data,
        config=test_api_config["endpoint_config"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Delete endpoint
    result = delete_endpoint(
        endpoint_id=endpoint["endpoint_id"],
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify deletion result
    assert isinstance(result, Dict)
    assert "deleted" in result
    assert result["deleted"] is True
    
    # Verify endpoint record
    endpoint_record = db_session.query(APIEndpoint).filter_by(
        endpoint_id=endpoint["endpoint_id"]
    ).first()
    assert endpoint_record is not None
    assert endpoint_record.status == "DELETED"

def test_api_error_handling(db_session, test_user):
    """Test API error handling"""
    # Invalid endpoint data
    with pytest.raises(APIError) as excinfo:
        create_endpoint(
            endpoint_data={},
            config={},
            user_id=test_user.user_id,
            db_session=db_session
        )
    assert "Invalid endpoint data" in str(excinfo.value)
    
    # Invalid request data
    with pytest.raises(APIError) as excinfo:
        handle_request(
            request_data={},
            config={},
            user_id=test_user.user_id,
            db_session=db_session
        )
    assert "Invalid request data" in str(excinfo.value)
    
    # Invalid authentication data
    with pytest.raises(APIError) as excinfo:
        authenticate_request(
            auth_data={},
            config={},
            db_session=db_session
        )
    assert "Invalid authentication data" in str(excinfo.value)
    
    # Invalid authorization data
    with pytest.raises(APIError) as excinfo:
        authorize_request(
            auth_info={},
            resource_info={},
            config={},
            db_session=db_session
        )
    assert "Invalid authorization data" in str(excinfo.value)
    
    # Rate limit exceeded
    with pytest.raises(APIError) as excinfo:
        rate_limit_request(
            request_info={"user_id": test_user.user_id, "count": 1000000},
            config={"limits": {"requests_per_second": 1}},
            db_session=db_session
        )
    assert "Rate limit exceeded" in str(excinfo.value)
    
    # Database error
    with pytest.raises(DBAPIError) as excinfo:
        create_endpoint(
            endpoint_data={},
            config={},
            user_id=test_user.user_id,
            db_session=None
        )
    assert "Database error" in str(excinfo.value) 