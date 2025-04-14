import pytest
import hashlib
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, Any, List
from decimal import Decimal

from core.user_management import (
    register_user,
    authenticate_user,
    update_user_profile,
    change_password,
    reset_password,
    verify_email,
    get_user_profile,
    list_users,
    deactivate_user,
    activate_user,
    UserRole,
    UserStatus
)
from database.models import User, UserProfile, UserSession, UserRole as UserRoleModel
from database.exceptions import UserManagementError

@pytest.fixture
def test_user_data():
    """Create test user data"""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "TestPassword123!",
        "first_name": "Test",
        "last_name": "User",
        "role": UserRole.USER
    }

@pytest.fixture
def test_admin_data():
    """Create test admin data"""
    return {
        "username": "adminuser",
        "email": "admin@example.com",
        "password": "AdminPassword123!",
        "first_name": "Admin",
        "last_name": "User",
        "role": UserRole.ADMIN
    }

@pytest.fixture
def test_user(db_session, test_user_data):
    """Create test user"""
    # Hash password
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(test_user_data["password"].encode(), salt)
    
    # Create user
    user = User(
        user_id="test_user_id",
        username=test_user_data["username"],
        email=test_user_data["email"],
        password_hash=hashed_password.decode(),
        first_name=test_user_data["first_name"],
        last_name=test_user_data["last_name"],
        role=UserRoleModel.USER,
        status=UserStatus.ACTIVE,
        email_verified=True,
        created_at=datetime.utcnow(),
        last_login=datetime.utcnow()
    )
    
    # Create user profile
    profile = UserProfile(
        user_id=user.user_id,
        bio="Test user bio",
        location="Test location",
        timezone="UTC",
        preferences={
            "theme": "light",
            "notifications": True,
            "language": "en"
        }
    )
    
    db_session.add(user)
    db_session.add(profile)
    db_session.commit()
    
    return user

@pytest.fixture
def test_admin(db_session, test_admin_data):
    """Create test admin user"""
    # Hash password
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(test_admin_data["password"].encode(), salt)
    
    # Create admin user
    admin = User(
        user_id="admin_user_id",
        username=test_admin_data["username"],
        email=test_admin_data["email"],
        password_hash=hashed_password.decode(),
        first_name=test_admin_data["first_name"],
        last_name=test_admin_data["last_name"],
        role=UserRoleModel.ADMIN,
        status=UserStatus.ACTIVE,
        email_verified=True,
        created_at=datetime.utcnow(),
        last_login=datetime.utcnow()
    )
    
    # Create admin profile
    profile = UserProfile(
        user_id=admin.user_id,
        bio="Admin user bio",
        location="Admin location",
        timezone="UTC",
        preferences={
            "theme": "dark",
            "notifications": True,
            "language": "en"
        }
    )
    
    db_session.add(admin)
    db_session.add(profile)
    db_session.commit()
    
    return admin

def test_register_user(db_session, test_user_data):
    """Test user registration"""
    # Register user
    user = register_user(
        username=test_user_data["username"] + "_new",
        email="new_" + test_user_data["email"],
        password=test_user_data["password"],
        first_name=test_user_data["first_name"],
        last_name=test_user_data["last_name"],
        role=test_user_data["role"],
        db_session=db_session
    )
    
    # Verify user
    assert isinstance(user, Dict)
    assert "user_id" in user
    assert "username" in user
    assert "email" in user
    assert "first_name" in user
    assert "last_name" in user
    assert "role" in user
    assert "status" in user
    assert "email_verified" in user
    assert "created_at" in user
    
    # Verify user in database
    db_user = db_session.query(User).filter_by(user_id=user["user_id"]).first()
    assert db_user is not None
    assert db_user.username == test_user_data["username"] + "_new"
    assert db_user.email == "new_" + test_user_data["email"]
    assert db_user.first_name == test_user_data["first_name"]
    assert db_user.last_name == test_user_data["last_name"]
    assert db_user.role == UserRoleModel.USER
    assert db_user.status == UserStatus.ACTIVE
    assert db_user.email_verified is False
    
    # Verify user profile created
    db_profile = db_session.query(UserProfile).filter_by(user_id=user["user_id"]).first()
    assert db_profile is not None
    assert db_profile.bio == ""
    assert db_profile.location == ""
    assert db_profile.timezone == "UTC"
    assert "theme" in db_profile.preferences
    assert "notifications" in db_profile.preferences
    assert "language" in db_profile.preferences

def test_authenticate_user(db_session, test_user, test_user_data):
    """Test user authentication"""
    # Authenticate user
    auth_result = authenticate_user(
        username=test_user_data["username"],
        password=test_user_data["password"],
        db_session=db_session
    )
    
    # Verify authentication result
    assert isinstance(auth_result, Dict)
    assert "user_id" in auth_result
    assert "username" in auth_result
    assert "email" in auth_result
    assert "token" in auth_result
    assert "expires_at" in auth_result
    
    # Verify token
    token = auth_result["token"]
    decoded_token = jwt.decode(token, "secret_key", algorithms=["HS256"])
    assert decoded_token["user_id"] == test_user.user_id
    assert decoded_token["username"] == test_user.username
    assert decoded_token["role"] == test_user.role.value
    
    # Verify session created
    db_session_obj = db_session.query(UserSession).filter_by(
        user_id=test_user.user_id,
        token=token
    ).first()
    assert db_session_obj is not None
    assert db_session_obj.expires_at == datetime.fromisoformat(auth_result["expires_at"])
    
    # Verify last login updated
    db_user = db_session.query(User).filter_by(user_id=test_user.user_id).first()
    assert db_user.last_login is not None

def test_update_user_profile(db_session, test_user):
    """Test user profile update"""
    # Update profile
    updated_profile = update_user_profile(
        user_id=test_user.user_id,
        bio="Updated bio",
        location="Updated location",
        timezone="America/New_York",
        preferences={
            "theme": "dark",
            "notifications": False,
            "language": "es"
        },
        db_session=db_session
    )
    
    # Verify updated profile
    assert isinstance(updated_profile, Dict)
    assert "user_id" in updated_profile
    assert "bio" in updated_profile
    assert "location" in updated_profile
    assert "timezone" in updated_profile
    assert "preferences" in updated_profile
    
    # Verify values updated
    assert updated_profile["bio"] == "Updated bio"
    assert updated_profile["location"] == "Updated location"
    assert updated_profile["timezone"] == "America/New_York"
    assert updated_profile["preferences"]["theme"] == "dark"
    assert updated_profile["preferences"]["notifications"] is False
    assert updated_profile["preferences"]["language"] == "es"
    
    # Verify database updated
    db_profile = db_session.query(UserProfile).filter_by(user_id=test_user.user_id).first()
    assert db_profile.bio == "Updated bio"
    assert db_profile.location == "Updated location"
    assert db_profile.timezone == "America/New_York"
    assert db_profile.preferences["theme"] == "dark"
    assert db_profile.preferences["notifications"] is False
    assert db_profile.preferences["language"] == "es"

def test_change_password(db_session, test_user, test_user_data):
    """Test password change"""
    # Change password
    new_password = "NewTestPassword123!"
    result = change_password(
        user_id=test_user.user_id,
        current_password=test_user_data["password"],
        new_password=new_password,
        db_session=db_session
    )
    
    # Verify result
    assert isinstance(result, Dict)
    assert "success" in result
    assert result["success"] is True
    
    # Verify password updated
    db_user = db_session.query(User).filter_by(user_id=test_user.user_id).first()
    assert bcrypt.checkpw(new_password.encode(), db_user.password_hash.encode())
    
    # Verify can authenticate with new password
    auth_result = authenticate_user(
        username=test_user_data["username"],
        password=new_password,
        db_session=db_session
    )
    assert auth_result["user_id"] == test_user.user_id

def test_reset_password(db_session, test_user):
    """Test password reset"""
    # Reset password
    reset_token = "reset_token_123"
    new_password = "ResetPassword123!"
    result = reset_password(
        reset_token=reset_token,
        new_password=new_password,
        db_session=db_session
    )
    
    # Verify result
    assert isinstance(result, Dict)
    assert "success" in result
    assert result["success"] is True
    
    # Verify password updated
    db_user = db_session.query(User).filter_by(user_id=test_user.user_id).first()
    assert bcrypt.checkpw(new_password.encode(), db_user.password_hash.encode())
    
    # Verify can authenticate with new password
    auth_result = authenticate_user(
        username=test_user.username,
        password=new_password,
        db_session=db_session
    )
    assert auth_result["user_id"] == test_user.user_id

def test_verify_email(db_session, test_user):
    """Test email verification"""
    # Set email as unverified
    test_user.email_verified = False
    db_session.commit()
    
    # Verify email
    verification_token = "verification_token_123"
    result = verify_email(
        verification_token=verification_token,
        db_session=db_session
    )
    
    # Verify result
    assert isinstance(result, Dict)
    assert "success" in result
    assert result["success"] is True
    
    # Verify email marked as verified
    db_user = db_session.query(User).filter_by(user_id=test_user.user_id).first()
    assert db_user.email_verified is True

def test_get_user_profile(db_session, test_user):
    """Test user profile retrieval"""
    # Get user profile
    profile = get_user_profile(
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify profile
    assert isinstance(profile, Dict)
    assert "user_id" in profile
    assert "username" in profile
    assert "email" in profile
    assert "first_name" in profile
    assert "last_name" in profile
    assert "bio" in profile
    assert "location" in profile
    assert "timezone" in profile
    assert "preferences" in profile
    assert "role" in profile
    assert "status" in profile
    assert "email_verified" in profile
    assert "created_at" in profile
    assert "last_login" in profile
    
    # Verify values
    assert profile["user_id"] == test_user.user_id
    assert profile["username"] == test_user.username
    assert profile["email"] == test_user.email
    assert profile["first_name"] == test_user.first_name
    assert profile["last_name"] == test_user.last_name
    assert profile["bio"] == "Test user bio"
    assert profile["location"] == "Test location"
    assert profile["timezone"] == "UTC"
    assert profile["preferences"]["theme"] == "light"
    assert profile["preferences"]["notifications"] is True
    assert profile["preferences"]["language"] == "en"
    assert profile["role"] == test_user.role.value
    assert profile["status"] == test_user.status.value
    assert profile["email_verified"] == test_user.email_verified

def test_list_users(db_session, test_user, test_admin):
    """Test user listing"""
    # List users
    users = list_users(
        role=None,
        status=None,
        limit=10,
        offset=0,
        db_session=db_session
    )
    
    # Verify users
    assert isinstance(users, Dict)
    assert "users" in users
    assert "total" in users
    assert "limit" in users
    assert "offset" in users
    
    # Verify user list
    assert len(users["users"]) >= 2  # At least test_user and test_admin
    assert users["total"] >= 2
    assert users["limit"] == 10
    assert users["offset"] == 0
    
    # Verify user details
    user_ids = [user["user_id"] for user in users["users"]]
    assert test_user.user_id in user_ids
    assert test_admin.user_id in user_ids
    
    # List users by role
    admin_users = list_users(
        role=UserRole.ADMIN,
        status=None,
        limit=10,
        offset=0,
        db_session=db_session
    )
    assert len(admin_users["users"]) >= 1  # At least test_admin
    assert all(user["role"] == UserRole.ADMIN.value for user in admin_users["users"])
    
    # List users by status
    active_users = list_users(
        role=None,
        status=UserStatus.ACTIVE,
        limit=10,
        offset=0,
        db_session=db_session
    )
    assert len(active_users["users"]) >= 2  # At least test_user and test_admin
    assert all(user["status"] == UserStatus.ACTIVE.value for user in active_users["users"])

def test_deactivate_user(db_session, test_user):
    """Test user deactivation"""
    # Deactivate user
    result = deactivate_user(
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify result
    assert isinstance(result, Dict)
    assert "success" in result
    assert result["success"] is True
    
    # Verify user deactivated
    db_user = db_session.query(User).filter_by(user_id=test_user.user_id).first()
    assert db_user.status == UserStatus.INACTIVE
    
    # Verify cannot authenticate
    with pytest.raises(UserManagementError) as excinfo:
        authenticate_user(
            username=test_user.username,
            password="TestPassword123!",
            db_session=db_session
        )
    assert "User is inactive" in str(excinfo.value)

def test_activate_user(db_session, test_user):
    """Test user activation"""
    # Deactivate user first
    test_user.status = UserStatus.INACTIVE
    db_session.commit()
    
    # Activate user
    result = activate_user(
        user_id=test_user.user_id,
        db_session=db_session
    )
    
    # Verify result
    assert isinstance(result, Dict)
    assert "success" in result
    assert result["success"] is True
    
    # Verify user activated
    db_user = db_session.query(User).filter_by(user_id=test_user.user_id).first()
    assert db_user.status == UserStatus.ACTIVE
    
    # Verify can authenticate
    auth_result = authenticate_user(
        username=test_user.username,
        password="TestPassword123!",
        db_session=db_session
    )
    assert auth_result["user_id"] == test_user.user_id

def test_user_management_error_handling():
    """Test user management error handling"""
    # Invalid username
    with pytest.raises(UserManagementError) as excinfo:
        register_user(
            username="",  # Empty username
            email="test@example.com",
            password="TestPassword123!",
            first_name="Test",
            last_name="User",
            role=UserRole.USER,
            db_session=None
        )
    assert "Invalid username" in str(excinfo.value)
    
    # Invalid email
    with pytest.raises(UserManagementError) as excinfo:
        register_user(
            username="testuser",
            email="invalid_email",  # Invalid email
            password="TestPassword123!",
            first_name="Test",
            last_name="User",
            role=UserRole.USER,
            db_session=None
        )
    assert "Invalid email" in str(excinfo.value)
    
    # Invalid password
    with pytest.raises(UserManagementError) as excinfo:
        register_user(
            username="testuser",
            email="test@example.com",
            password="weak",  # Weak password
            first_name="Test",
            last_name="User",
            role=UserRole.USER,
            db_session=None
        )
    assert "Invalid password" in str(excinfo.value)
    
    # Username already exists
    with pytest.raises(UserManagementError) as excinfo:
        register_user(
            username="testuser",  # Existing username
            email="new@example.com",
            password="TestPassword123!",
            first_name="Test",
            last_name="User",
            role=UserRole.USER,
            db_session=None
        )
    assert "Username already exists" in str(excinfo.value)
    
    # Email already exists
    with pytest.raises(UserManagementError) as excinfo:
        register_user(
            username="newuser",
            email="test@example.com",  # Existing email
            password="TestPassword123!",
            first_name="Test",
            last_name="User",
            role=UserRole.USER,
            db_session=None
        )
    assert "Email already exists" in str(excinfo.value)
    
    # Invalid credentials
    with pytest.raises(UserManagementError) as excinfo:
        authenticate_user(
            username="testuser",
            password="WrongPassword123!",  # Wrong password
            db_session=None
        )
    assert "Invalid credentials" in str(excinfo.value)
    
    # User not found
    with pytest.raises(UserManagementError) as excinfo:
        get_user_profile(
            user_id="non_existent_user_id",
            db_session=None
        )
    assert "User not found" in str(excinfo.value)
    
    # Invalid reset token
    with pytest.raises(UserManagementError) as excinfo:
        reset_password(
            reset_token="invalid_token",
            new_password="NewPassword123!",
            db_session=None
        )
    assert "Invalid reset token" in str(excinfo.value)
    
    # Invalid verification token
    with pytest.raises(UserManagementError) as excinfo:
        verify_email(
            verification_token="invalid_token",
            db_session=None
        )
    assert "Invalid verification token" in str(excinfo.value) 