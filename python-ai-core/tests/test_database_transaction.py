import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, IntegrityError
from contextlib import contextmanager

from database.transaction import (
    begin_transaction,
    commit_transaction,
    rollback_transaction,
    with_transaction,
    get_transaction_isolation_level,
    set_transaction_isolation_level,
    get_transaction_status,
    get_transaction_savepoint,
    set_transaction_savepoint,
    rollback_to_savepoint
)
from database.exceptions import DatabaseError, TransactionError
from database.models import User, Activity, Reward, RiskAssessment, ActivityType, RewardType, RiskLevel
from database.transaction import TransactionManager

@pytest.fixture
def db_session():
    """Create database session"""
    session = Session()
    yield session
    session.close()

@pytest.fixture
def test_user_data() -> Dict[str, Any]:
    """Create test user data"""
    return {
        "user_id": "test_user",
        "username": "testuser",
        "email": "test@example.com",
        "created_at": datetime.utcnow(),
        "last_login": datetime.utcnow(),
        "is_active": True,
        "is_verified": True,
        "preferences": {
            "notification_enabled": True,
            "theme": "dark",
            "language": "en"
        }
    }

@pytest.fixture
def test_mining_activity_data() -> Dict[str, Any]:
    """Create test mining activity data"""
    return {
        "activity_id": "test_mining_activity",
        "user_id": "test_user",
        "activity_type": ActivityType.MINING,
        "start_time": datetime.utcnow() - timedelta(hours=1),
        "end_time": datetime.utcnow(),
        "status": "completed",
        "device_type": "gpu",
        "hash_rate": 100.0,
        "power_usage": 500.0,
        "efficiency_score": 0.85,
        "performance_metrics": {
            "hash_rate": 100.0,
            "power_usage": 500.0,
            "efficiency": 0.85,
            "temperature": 75.0,
            "fan_speed": 80.0
        }
    }

@pytest.fixture
def test_mining_reward_data() -> Dict[str, Any]:
    """Create test mining reward data"""
    return {
        "reward_id": "test_mining_reward",
        "user_id": "test_user",
        "activity_id": "test_mining_activity",
        "reward_type": RewardType.MINING,
        "amount": 0.001,
        "currency": "BTC",
        "timestamp": datetime.utcnow(),
        "status": "pending",
        "transaction_hash": "0x1234567890abcdef",
        "metadata": {
            "block_height": 12345678,
            "block_hash": "0xabcdef1234567890",
            "transaction_index": 0
        }
    }

@pytest.fixture
def transaction_manager(db_session: Session) -> TransactionManager:
    """Create a transaction manager"""
    return TransactionManager(db_session)

def test_begin_transaction(transaction_manager: TransactionManager):
    """Test beginning a transaction"""
    # Begin transaction
    transaction_manager.begin()

    # Check transaction state
    assert transaction_manager.is_active()
    assert not transaction_manager.is_committed()
    assert not transaction_manager.is_rolled_back()

def test_commit_transaction(transaction_manager: TransactionManager, test_user_data: Dict[str, Any]):
    """Test committing a transaction"""
    # Begin transaction
    transaction_manager.begin()

    # Create user
    user = User(**test_user_data)
    transaction_manager.session.add(user)

    # Commit transaction
    transaction_manager.commit()

    # Check transaction state
    assert not transaction_manager.is_active()
    assert transaction_manager.is_committed()
    assert not transaction_manager.is_rolled_back()

    # Check user was created
    retrieved_user = transaction_manager.session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
    assert retrieved_user is not None
    assert retrieved_user.user_id == test_user_data["user_id"]
    assert retrieved_user.username == test_user_data["username"]
    assert retrieved_user.email == test_user_data["email"]

def test_rollback_transaction(transaction_manager: TransactionManager, test_user_data: Dict[str, Any]):
    """Test rolling back a transaction"""
    # Begin transaction
    transaction_manager.begin()

    # Create user
    user = User(**test_user_data)
    transaction_manager.session.add(user)

    # Rollback transaction
    transaction_manager.rollback()

    # Check transaction state
    assert not transaction_manager.is_active()
    assert not transaction_manager.is_committed()
    assert transaction_manager.is_rolled_back()

    # Check user was not created
    retrieved_user = transaction_manager.session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
    assert retrieved_user is None

def test_nested_transaction(transaction_manager: TransactionManager, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any]):
    """Test nested transaction"""
    # Begin outer transaction
    transaction_manager.begin()

    # Create user
    user = User(**test_user_data)
    transaction_manager.session.add(user)

    # Begin inner transaction
    transaction_manager.begin()

    # Create activity
    activity = Activity(**test_mining_activity_data)
    transaction_manager.session.add(activity)

    # Commit inner transaction
    transaction_manager.commit()

    # Check inner transaction state
    assert transaction_manager.is_active()
    assert not transaction_manager.is_committed()
    assert not transaction_manager.is_rolled_back()

    # Commit outer transaction
    transaction_manager.commit()

    # Check outer transaction state
    assert not transaction_manager.is_active()
    assert transaction_manager.is_committed()
    assert not transaction_manager.is_rolled_back()

    # Check user and activity were created
    retrieved_user = transaction_manager.session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
    retrieved_activity = transaction_manager.session.query(Activity).filter_by(activity_id=test_mining_activity_data["activity_id"]).first()

    assert retrieved_user is not None
    assert retrieved_user.user_id == test_user_data["user_id"]
    assert retrieved_activity is not None
    assert retrieved_activity.activity_id == test_mining_activity_data["activity_id"]

def test_nested_transaction_rollback(transaction_manager: TransactionManager, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any]):
    """Test nested transaction rollback"""
    # Begin outer transaction
    transaction_manager.begin()

    # Create user
    user = User(**test_user_data)
    transaction_manager.session.add(user)

    # Begin inner transaction
    transaction_manager.begin()

    # Create activity
    activity = Activity(**test_mining_activity_data)
    transaction_manager.session.add(activity)

    # Rollback inner transaction
    transaction_manager.rollback()

    # Check inner transaction state
    assert transaction_manager.is_active()
    assert not transaction_manager.is_committed()
    assert not transaction_manager.is_rolled_back()

    # Commit outer transaction
    transaction_manager.commit()

    # Check outer transaction state
    assert not transaction_manager.is_active()
    assert transaction_manager.is_committed()
    assert not transaction_manager.is_rolled_back()

    # Check user was created but activity was not
    retrieved_user = transaction_manager.session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
    retrieved_activity = transaction_manager.session.query(Activity).filter_by(activity_id=test_mining_activity_data["activity_id"]).first()

    assert retrieved_user is not None
    assert retrieved_user.user_id == test_user_data["user_id"]
    assert retrieved_activity is None

def test_outer_transaction_rollback(transaction_manager: TransactionManager, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any]):
    """Test outer transaction rollback"""
    # Begin outer transaction
    transaction_manager.begin()

    # Create user
    user = User(**test_user_data)
    transaction_manager.session.add(user)

    # Begin inner transaction
    transaction_manager.begin()

    # Create activity
    activity = Activity(**test_mining_activity_data)
    transaction_manager.session.add(activity)

    # Commit inner transaction
    transaction_manager.commit()

    # Check inner transaction state
    assert transaction_manager.is_active()
    assert not transaction_manager.is_committed()
    assert not transaction_manager.is_rolled_back()

    # Rollback outer transaction
    transaction_manager.rollback()

    # Check outer transaction state
    assert not transaction_manager.is_active()
    assert not transaction_manager.is_committed()
    assert transaction_manager.is_rolled_back()

    # Check neither user nor activity were created
    retrieved_user = transaction_manager.session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
    retrieved_activity = transaction_manager.session.query(Activity).filter_by(activity_id=test_mining_activity_data["activity_id"]).first()

    assert retrieved_user is None
    assert retrieved_activity is None

def test_transaction_with_savepoint(transaction_manager: TransactionManager, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any]):
    """Test transaction with savepoint"""
    # Begin transaction
    transaction_manager.begin()

    # Create user
    user = User(**test_user_data)
    transaction_manager.session.add(user)

    # Create savepoint
    transaction_manager.savepoint("user_created")

    # Create activity
    activity = Activity(**test_mining_activity_data)
    transaction_manager.session.add(activity)

    # Create savepoint
    transaction_manager.savepoint("activity_created")

    # Create reward
    reward = Reward(**test_mining_reward_data)
    transaction_manager.session.add(reward)

    # Rollback to activity savepoint
    transaction_manager.rollback_to_savepoint("activity_created")

    # Check reward was not created
    retrieved_reward = transaction_manager.session.query(Reward).filter_by(reward_id=test_mining_reward_data["reward_id"]).first()
    assert retrieved_reward is None

    # Check user and activity were created
    retrieved_user = transaction_manager.session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
    retrieved_activity = transaction_manager.session.query(Activity).filter_by(activity_id=test_mining_activity_data["activity_id"]).first()

    assert retrieved_user is not None
    assert retrieved_user.user_id == test_user_data["user_id"]
    assert retrieved_activity is not None
    assert retrieved_activity.activity_id == test_mining_activity_data["activity_id"]

    # Commit transaction
    transaction_manager.commit()

    # Check transaction state
    assert not transaction_manager.is_active()
    assert transaction_manager.is_committed()
    assert not transaction_manager.is_rolled_back()

def test_transaction_with_error(transaction_manager: TransactionManager, test_user_data: Dict[str, Any]):
    """Test transaction with error"""
    # Begin transaction
    transaction_manager.begin()

    # Create user
    user = User(**test_user_data)
    transaction_manager.session.add(user)

    # Try to create another user with the same ID (should cause an error)
    duplicate_user = User(**test_user_data)
    transaction_manager.session.add(duplicate_user)

    # Check that committing raises an error
    with pytest.raises(IntegrityError):
        transaction_manager.commit()

    # Check transaction state
    assert not transaction_manager.is_active()
    assert not transaction_manager.is_committed()
    assert transaction_manager.is_rolled_back()

    # Check user was not created
    retrieved_user = transaction_manager.session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
    assert retrieved_user is None

def test_transaction_with_connection_error(transaction_manager: TransactionManager, test_user_data: Dict[str, Any]):
    """Test transaction with connection error"""
    # Begin transaction
    transaction_manager.begin()

    # Create user
    user = User(**test_user_data)
    transaction_manager.session.add(user)

    # Mock a connection error
    def mock_commit():
        raise OperationalError("Connection lost", None, None)

    transaction_manager.session.commit = mock_commit

    # Check that committing raises an error
    with pytest.raises(OperationalError):
        transaction_manager.commit()

    # Check transaction state
    assert not transaction_manager.is_active()
    assert not transaction_manager.is_committed()
    assert transaction_manager.is_rolled_back()

    # Check user was not created
    retrieved_user = transaction_manager.session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
    assert retrieved_user is None

def test_transaction_timeout(transaction_manager: TransactionManager, test_user_data: Dict[str, Any]):
    """Test transaction timeout"""
    # Set transaction timeout
    transaction_manager.set_timeout(1)  # 1 second timeout

    # Begin transaction
    transaction_manager.begin()

    # Create user
    user = User(**test_user_data)
    transaction_manager.session.add(user)

    # Simulate a long-running operation
    import time
    time.sleep(2)

    # Check that committing raises a timeout error
    with pytest.raises(DatabaseError) as excinfo:
        transaction_manager.commit()

    assert "Transaction timeout" in str(excinfo.value)

    # Check transaction state
    assert not transaction_manager.is_active()
    assert not transaction_manager.is_committed()
    assert transaction_manager.is_rolled_back()

    # Check user was not created
    retrieved_user = transaction_manager.session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
    assert retrieved_user is None

def test_transaction_isolation_level(transaction_manager: TransactionManager, test_user_data: Dict[str, Any]):
    """Test transaction isolation level"""
    # Set transaction isolation level
    transaction_manager.set_isolation_level("READ COMMITTED")

    # Begin transaction
    transaction_manager.begin()

    # Check isolation level
    assert transaction_manager.get_isolation_level() == "READ COMMITTED"

    # Create user
    user = User(**test_user_data)
    transaction_manager.session.add(user)

    # Commit transaction
    transaction_manager.commit()

    # Check user was created
    retrieved_user = transaction_manager.session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
    assert retrieved_user is not None
    assert retrieved_user.user_id == test_user_data["user_id"]

def test_transaction_read_only(transaction_manager: TransactionManager, test_user_data: Dict[str, Any]):
    """Test transaction read-only mode"""
    # Create user first (outside the read-only transaction)
    user = User(**test_user_data)
    transaction_manager.session.add(user)
    transaction_manager.session.commit()

    # Begin read-only transaction
    transaction_manager.begin_read_only()

    # Try to create another user (should not be allowed)
    another_user = User(**{"user_id": "another_user", "username": "anotheruser", "email": "another@example.com"})
    transaction_manager.session.add(another_user)

    # Check that committing raises an error
    with pytest.raises(DatabaseError) as excinfo:
        transaction_manager.commit()

    assert "Read-only transaction" in str(excinfo.value)

    # Check transaction state
    assert not transaction_manager.is_active()
    assert not transaction_manager.is_committed()
    assert transaction_manager.is_rolled_back()

    # Check another user was not created
    retrieved_another_user = transaction_manager.session.query(User).filter_by(user_id="another_user").first()
    assert retrieved_another_user is None

    # Check original user still exists
    retrieved_user = transaction_manager.session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
    assert retrieved_user is not None
    assert retrieved_user.user_id == test_user_data["user_id"]

def test_transaction_deferred_constraints(transaction_manager: TransactionManager, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any]):
    """Test transaction with deferred constraints"""
    # Begin transaction with deferred constraints
    transaction_manager.begin_deferred()

    # Create activity (should fail without user due to foreign key constraint)
    activity = Activity(**test_mining_activity_data)
    transaction_manager.session.add(activity)

    # Create user
    user = User(**test_user_data)
    transaction_manager.session.add(user)

    # Commit transaction (should succeed because constraints are checked at commit time)
    transaction_manager.commit()

    # Check user and activity were created
    retrieved_user = transaction_manager.session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
    retrieved_activity = transaction_manager.session.query(Activity).filter_by(activity_id=test_mining_activity_data["activity_id"]).first()

    assert retrieved_user is not None
    assert retrieved_user.user_id == test_user_data["user_id"]
    assert retrieved_activity is not None
    assert retrieved_activity.activity_id == test_mining_activity_data["activity_id"]

def test_transaction_with_multiple_operations(transaction_manager: TransactionManager, test_user_data: Dict[str, Any], test_mining_activity_data: Dict[str, Any], test_mining_reward_data: Dict[str, Any]):
    """Test transaction with multiple operations"""
    # Begin transaction
    transaction_manager.begin()

    # Create user
    user = User(**test_user_data)
    transaction_manager.session.add(user)

    # Create activity
    activity = Activity(**test_mining_activity_data)
    transaction_manager.session.add(activity)

    # Create reward
    reward = Reward(**test_mining_reward_data)
    transaction_manager.session.add(reward)

    # Update user
    user.is_active = False
    user.last_login = datetime.utcnow()

    # Delete activity
    transaction_manager.session.delete(activity)

    # Commit transaction
    transaction_manager.commit()

    # Check transaction state
    assert not transaction_manager.is_active()
    assert transaction_manager.is_committed()
    assert not transaction_manager.is_rolled_back()

    # Check user was updated
    retrieved_user = transaction_manager.session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
    assert retrieved_user is not None
    assert retrieved_user.user_id == test_user_data["user_id"]
    assert not retrieved_user.is_active

    # Check activity was deleted
    retrieved_activity = transaction_manager.session.query(Activity).filter_by(activity_id=test_mining_activity_data["activity_id"]).first()
    assert retrieved_activity is None

    # Check reward was created
    retrieved_reward = transaction_manager.session.query(Reward).filter_by(reward_id=test_mining_reward_data["reward_id"]).first()
    assert retrieved_reward is not None
    assert retrieved_reward.reward_id == test_mining_reward_data["reward_id"]

def test_transaction_with_batch_operations(transaction_manager: TransactionManager):
    """Test transaction with batch operations"""
    # Begin transaction
    transaction_manager.begin()

    # Create multiple users
    users = []
    for i in range(10):
        user_data = {
            "user_id": f"test_user_{i}",
            "username": f"testuser_{i}",
            "email": f"test_{i}@example.com",
            "created_at": datetime.utcnow(),
            "last_login": datetime.utcnow(),
            "is_active": True,
            "is_verified": True,
            "preferences": {
                "notification_enabled": True,
                "theme": "dark",
                "language": "en"
            }
        }
        user = User(**user_data)
        transaction_manager.session.add(user)
        users.append(user)

    # Commit transaction
    transaction_manager.commit()

    # Check transaction state
    assert not transaction_manager.is_active()
    assert transaction_manager.is_committed()
    assert not transaction_manager.is_rolled_back()

    # Check users were created
    for i in range(10):
        retrieved_user = transaction_manager.session.query(User).filter_by(user_id=f"test_user_{i}").first()
        assert retrieved_user is not None
        assert retrieved_user.user_id == f"test_user_{i}"
        assert retrieved_user.username == f"testuser_{i}"
        assert retrieved_user.email == f"test_{i}@example.com"

def test_transaction_with_large_data(transaction_manager: TransactionManager):
    """Test transaction with large data"""
    # Begin transaction
    transaction_manager.begin()

    # Create a user with large preferences
    large_preferences = {
        "notification_enabled": True,
        "theme": "dark",
        "language": "en",
        "data": "x" * 1000000  # 1MB of data
    }

    user_data = {
        "user_id": "test_user",
        "username": "testuser",
        "email": "test@example.com",
        "created_at": datetime.utcnow(),
        "last_login": datetime.utcnow(),
        "is_active": True,
        "is_verified": True,
        "preferences": large_preferences
    }

    user = User(**user_data)
    transaction_manager.session.add(user)

    # Commit transaction
    transaction_manager.commit()

    # Check transaction state
    assert not transaction_manager.is_active()
    assert transaction_manager.is_committed()
    assert not transaction_manager.is_rolled_back()

    # Check user was created
    retrieved_user = transaction_manager.session.query(User).filter_by(user_id="test_user").first()
    assert retrieved_user is not None
    assert retrieved_user.user_id == "test_user"
    assert retrieved_user.username == "testuser"
    assert retrieved_user.email == "test@example.com"
    assert len(retrieved_user.preferences["data"]) == 1000000

def test_transaction_with_concurrent_operations(transaction_manager: TransactionManager, test_user_data: Dict[str, Any]):
    """Test transaction with concurrent operations"""
    # Create user first
    user = User(**test_user_data)
    transaction_manager.session.add(user)
    transaction_manager.session.commit()

    # Begin transaction
    transaction_manager.begin()

    # Update user
    user.is_active = False
    user.last_login = datetime.utcnow()

    # Simulate another session updating the same user
    another_session = transaction_manager.session.get_bind().connect()
    another_session.execute(
        "UPDATE users SET is_active = TRUE WHERE user_id = :user_id",
        {"user_id": test_user_data["user_id"]}
    )
    another_session.commit()

    # Commit transaction (should succeed with optimistic locking)
    transaction_manager.commit()

    # Check transaction state
    assert not transaction_manager.is_active()
    assert transaction_manager.is_committed()
    assert not transaction_manager.is_rolled_back()

    # Check user was updated
    retrieved_user = transaction_manager.session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
    assert retrieved_user is not None
    assert retrieved_user.user_id == test_user_data["user_id"]
    assert not retrieved_user.is_active

def test_transaction_with_error_handling(transaction_manager: TransactionManager, test_user_data: Dict[str, Any]):
    """Test transaction with error handling"""
    # Begin transaction
    transaction_manager.begin()

    # Create user
    user = User(**test_user_data)
    transaction_manager.session.add(user)

    # Try to create another user with the same ID (should cause an error)
    duplicate_user = User(**test_user_data)
    transaction_manager.session.add(duplicate_user)

    # Check that committing raises an error
    with pytest.raises(IntegrityError):
        transaction_manager.commit()

    # Check transaction state
    assert not transaction_manager.is_active()
    assert not transaction_manager.is_committed()
    assert transaction_manager.is_rolled_back()

    # Check user was not created
    retrieved_user = transaction_manager.session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
    assert retrieved_user is None

    # Begin a new transaction
    transaction_manager.begin()

    # Create user
    user = User(**test_user_data)
    transaction_manager.session.add(user)

    # Commit transaction
    transaction_manager.commit()

    # Check user was created
    retrieved_user = transaction_manager.session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
    assert retrieved_user is not None
    assert retrieved_user.user_id == test_user_data["user_id"]

def test_transaction_with_retry(transaction_manager: TransactionManager, test_user_data: Dict[str, Any]):
    """Test transaction with retry"""
    # Set retry count
    transaction_manager.set_retry_count(3)

    # Begin transaction
    transaction_manager.begin()

    # Create user
    user = User(**test_user_data)
    transaction_manager.session.add(user)

    # Mock a transient error
    error_count = 0
    original_commit = transaction_manager.session.commit

    def mock_commit():
        nonlocal error_count
        if error_count < 2:
            error_count += 1
            raise OperationalError("Transient error", None, None)
        return original_commit()

    transaction_manager.session.commit = mock_commit

    # Commit transaction (should retry and succeed)
    transaction_manager.commit()

    # Check transaction state
    assert not transaction_manager.is_active()
    assert transaction_manager.is_committed()
    assert not transaction_manager.is_rolled_back()

    # Check user was created
    retrieved_user = transaction_manager.session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
    assert retrieved_user is not None
    assert retrieved_user.user_id == test_user_data["user_id"]

def test_transaction_with_timeout_and_retry(transaction_manager: TransactionManager, test_user_data: Dict[str, Any]):
    """Test transaction with timeout and retry"""
    # Set timeout and retry count
    transaction_manager.set_timeout(1)  # 1 second timeout
    transaction_manager.set_retry_count(3)

    # Begin transaction
    transaction_manager.begin()

    # Create user
    user = User(**test_user_data)
    transaction_manager.session.add(user)

    # Mock a timeout error
    error_count = 0
    original_commit = transaction_manager.session.commit

    def mock_commit():
        nonlocal error_count
        if error_count < 2:
            error_count += 1
            import time
            time.sleep(2)  # Simulate a long-running operation
            raise DatabaseError("Transaction timeout")
        return original_commit()

    transaction_manager.session.commit = mock_commit

    # Commit transaction (should retry and succeed)
    transaction_manager.commit()

    # Check transaction state
    assert not transaction_manager.is_active()
    assert transaction_manager.is_committed()
    assert not transaction_manager.is_rolled_back()

    # Check user was created
    retrieved_user = transaction_manager.session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
    assert retrieved_user is not None
    assert retrieved_user.user_id == test_user_data["user_id"] 