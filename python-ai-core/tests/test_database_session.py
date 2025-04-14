import pytest
from contextlib import contextmanager
from typing import Generator, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from database.session import (
    get_db,
    get_db_session,
    init_db,
    close_db,
    with_db_session,
    with_transaction
)
from database.models import User, Activity, Reward
from database.exceptions import DatabaseError

@pytest.fixture
def db_connection():
    """Create a database connection for testing"""
    # Initialize test database
    init_db()
    
    # Get database session
    session = next(get_db())
    
    try:
        yield session
    finally:
        # Close database connection
        close_db()

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

def test_get_db_session(db_connection):
    """Test getting a database session"""
    # Get database session
    session = get_db_session()
    
    # Check session was created
    assert session is not None
    assert isinstance(session, Session)
    
    # Check session is active
    assert session.is_active

def test_with_db_session(db_connection, test_user_data):
    """Test using database session context manager"""
    # Use database session context manager
    with with_db_session() as session:
        # Create user
        user = User(**test_user_data)
        session.add(user)
        session.commit()
        
        # Check user was created
        db_user = session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
        assert db_user is not None
        assert db_user.user_id == test_user_data["user_id"]
        assert db_user.username == test_user_data["username"]
        assert db_user.email == test_user_data["email"]
    
    # Check session was closed
    assert not session.is_active

def test_with_transaction(db_connection, test_user_data):
    """Test using transaction context manager"""
    # Use transaction context manager
    with with_transaction() as session:
        # Create user
        user = User(**test_user_data)
        session.add(user)
        
        # Check user was not committed yet
        db_user = session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
        assert db_user is None
    
    # Check transaction was committed
    with with_db_session() as session:
        db_user = session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
        assert db_user is not None
        assert db_user.user_id == test_user_data["user_id"]

def test_transaction_rollback(db_connection, test_user_data):
    """Test transaction rollback on error"""
    # Use transaction context manager with error
    with pytest.raises(DatabaseError):
        with with_transaction() as session:
            # Create user
            user = User(**test_user_data)
            session.add(user)
            
            # Raise error to trigger rollback
            raise DatabaseError("Test error")
    
    # Check transaction was rolled back
    with with_db_session() as session:
        db_user = session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
        assert db_user is None

def test_nested_transactions(db_connection, test_user_data):
    """Test nested transactions"""
    # Use outer transaction
    with with_transaction() as outer_session:
        # Create user in outer transaction
        user = User(**test_user_data)
        outer_session.add(user)
        
        # Use inner transaction
        with with_transaction() as inner_session:
            # Update user in inner transaction
            inner_user = inner_session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
            inner_user.username = "updated_user"
        
        # Check changes from inner transaction are visible in outer transaction
        outer_user = outer_session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
        assert outer_user.username == "updated_user"
    
    # Check all changes were committed
    with with_db_session() as session:
        db_user = session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
        assert db_user is not None
        assert db_user.username == "updated_user"

def test_concurrent_sessions(db_connection, test_user_data):
    """Test concurrent database sessions"""
    # Create first session
    with with_db_session() as session1:
        # Create user in first session
        user = User(**test_user_data)
        session1.add(user)
        session1.commit()
        
        # Create second session
        with with_db_session() as session2:
            # Check user is visible in second session
            db_user = session2.query(User).filter_by(user_id=test_user_data["user_id"]).first()
            assert db_user is not None
            assert db_user.user_id == test_user_data["user_id"]
            
            # Update user in second session
            db_user.username = "updated_user"
            session2.commit()
        
        # Check changes from second session are visible in first session
        session1.refresh(user)
        assert user.username == "updated_user"

def test_session_cleanup(db_connection, test_user_data):
    """Test session cleanup on error"""
    # Use database session with error
    with pytest.raises(DatabaseError):
        with with_db_session() as session:
            # Create user
            user = User(**test_user_data)
            session.add(user)
            
            # Raise error
            raise DatabaseError("Test error")
    
    # Check session was cleaned up
    assert not session.is_active
    
    # Check user was not created
    with with_db_session() as session:
        db_user = session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
        assert db_user is None

def test_connection_pool(db_connection):
    """Test database connection pool"""
    # Create multiple sessions
    sessions = []
    for _ in range(5):
        session = get_db_session()
        sessions.append(session)
    
    # Check all sessions are active
    for session in sessions:
        assert session.is_active
    
    # Close all sessions
    for session in sessions:
        session.close()
    
    # Check all sessions are closed
    for session in sessions:
        assert not session.is_active

def test_session_timeout(db_connection):
    """Test session timeout"""
    # Create session with short timeout
    with with_db_session(timeout=1) as session:
        # Check session is active
        assert session.is_active
        
        # Wait for timeout
        time.sleep(2)
        
        # Check session is closed
        assert not session.is_active

def test_connection_retry(db_connection):
    """Test connection retry on failure"""
    # Close database connection
    close_db()
    
    # Try to get database session with retry
    session = get_db_session(retry_count=3, retry_delay=1)
    
    # Check session was created
    assert session is not None
    assert isinstance(session, Session)
    assert session.is_active

def test_transaction_isolation(db_connection, test_user_data):
    """Test transaction isolation level"""
    # Use transaction with read committed isolation
    with with_transaction(isolation_level="READ COMMITTED") as session1:
        # Create user
        user = User(**test_user_data)
        session1.add(user)
        
        # Use another transaction
        with with_transaction(isolation_level="READ COMMITTED") as session2:
            # Check user is not visible until commit
            db_user = session2.query(User).filter_by(user_id=test_user_data["user_id"]).first()
            assert db_user is None
    
    # Check user is visible after commit
    with with_db_session() as session:
        db_user = session.query(User).filter_by(user_id=test_user_data["user_id"]).first()
        assert db_user is not None
        assert db_user.user_id == test_user_data["user_id"]

def test_session_events(db_connection):
    """Test session events"""
    events = []
    
    # Define event handlers
    def before_commit(session):
        events.append("before_commit")
    
    def after_commit(session):
        events.append("after_commit")
    
    def before_rollback(session):
        events.append("before_rollback")
    
    def after_rollback(session):
        events.append("after_rollback")
    
    # Register event handlers
    Session.event.listen(Session, "before_commit", before_commit)
    Session.event.listen(Session, "after_commit", after_commit)
    Session.event.listen(Session, "before_rollback", before_rollback)
    Session.event.listen(Session, "after_rollback", after_rollback)
    
    # Use transaction
    with with_transaction() as session:
        # Create user
        user = User(**test_user_data)
        session.add(user)
    
    # Check events were triggered
    assert "before_commit" in events
    assert "after_commit" in events
    assert "before_rollback" not in events
    assert "after_rollback" not in events
    
    # Use transaction with error
    events = []
    with pytest.raises(DatabaseError):
        with with_transaction() as session:
            # Create user
            user = User(**test_user_data)
            session.add(user)
            
            # Raise error
            raise DatabaseError("Test error")
    
    # Check events were triggered
    assert "before_commit" not in events
    assert "after_commit" not in events
    assert "before_rollback" in events
    assert "after_rollback" in events 