import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, TimeoutError, SQLAlchemyError
from contextlib import contextmanager
import time
from typing import Generator

from database.connection import (
    get_db_connection,
    get_db_session,
    close_db_connection,
    check_db_connection,
    retry_db_operation,
    with_db_session,
    get_connection_pool,
    release_connection,
    get_connection_stats,
    DatabaseConnection,
    get_connection,
    close_connection,
    is_connection_alive,
    create_session,
    close_all_connections
)
from database.exceptions import DatabaseError, ConnectionError, TimeoutError
from database.config import DatabaseConfig

@pytest.fixture
def db_config():
    """Create database configuration"""
    return DatabaseConfig(
        host="localhost",
        port=5432,
        database="test_db",
        user="test_user",
        password="test_password",
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800
    )

@pytest.fixture
def mock_db_connection(mocker):
    """Mock database connection"""
    mock_conn = mocker.Mock()
    mock_conn.execute.return_value = mocker.Mock()
    mock_conn.begin.return_value.__enter__.return_value = mock_conn
    mock_conn.begin.return_value.__exit__.return_value = None
    return mock_conn

@pytest.fixture
def mock_db_session(mocker):
    """Mock database session"""
    mock_session = mocker.Mock()
    mock_session.execute.return_value = mocker.Mock()
    mock_session.begin.return_value.__enter__.return_value = mock_session
    mock_session.begin.return_value.__exit__.return_value = None
    return mock_session

@pytest.fixture
def db_connection() -> DatabaseConnection:
    """Create a database connection for testing"""
    connection = get_connection()
    yield connection
    close_connection(connection)

@pytest.fixture
def db_session_factory(db_connection: DatabaseConnection) -> sessionmaker:
    """Create a session factory"""
    return sessionmaker(bind=db_connection.engine)

@pytest.fixture
def db_session(db_session_factory: sessionmaker):
    """Create a database session"""
    session = db_session_factory()
    yield session
    session.close()

def test_get_db_connection(db_config, mock_db_connection, mocker):
    """Test getting database connection"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_connection)
    
    # Get database connection
    connection = get_db_connection(db_config)
    
    # Check connection was created
    assert connection is not None
    assert connection == mock_db_connection

def test_get_db_session(db_config, mock_db_session, mocker):
    """Test getting database session"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_session)
    
    # Get database session
    session = get_db_session(db_config)
    
    # Check session was created
    assert session is not None
    assert session == mock_db_session

def test_close_db_connection(db_config, mock_db_connection, mocker):
    """Test closing database connection"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_connection)
    
    # Get database connection
    connection = get_db_connection(db_config)
    
    # Close database connection
    close_db_connection(connection)
    
    # Check connection was closed
    mock_db_connection.close.assert_called_once()

def test_check_db_connection(db_config, mock_db_connection, mocker):
    """Test checking database connection"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_connection)
    
    # Get database connection
    connection = get_db_connection(db_config)
    
    # Check database connection
    is_connected = check_db_connection(connection)
    
    # Check connection is valid
    assert is_connected is True
    mock_db_connection.execute.assert_called_once()

def test_check_db_connection_error(db_config, mock_db_connection, mocker):
    """Test checking database connection with error"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_connection)
    
    # Get database connection
    connection = get_db_connection(db_config)
    
    # Mock connection error
    mock_db_connection.execute.side_effect = OperationalError("connection error", None, None)
    
    # Check database connection
    is_connected = check_db_connection(connection)
    
    # Check connection is invalid
    assert is_connected is False

def test_retry_db_operation(db_config, mock_db_connection, mocker):
    """Test retrying database operation"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_connection)
    
    # Get database connection
    connection = get_db_connection(db_config)
    
    # Define operation
    def operation():
        return connection.execute(text("SELECT 1"))
    
    # Retry database operation
    result = retry_db_operation(operation, max_retries=3, retry_delay=1)
    
    # Check operation was successful
    assert result is not None
    mock_db_connection.execute.assert_called_once()

def test_retry_db_operation_error(db_config, mock_db_connection, mocker):
    """Test retrying database operation with error"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_connection)
    
    # Get database connection
    connection = get_db_connection(db_config)
    
    # Define operation
    def operation():
        raise OperationalError("connection error", None, None)
    
    # Retry database operation
    with pytest.raises(OperationalError):
        retry_db_operation(operation, max_retries=3, retry_delay=1)
    
    # Check operation was retried
    assert mock_db_connection.execute.call_count == 0

def test_with_db_session(db_config, mock_db_session, mocker):
    """Test using database session context manager"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_session)
    
    # Use database session
    with with_db_session(db_config) as session:
        # Check session was created
        assert session is not None
        assert session == mock_db_session
        
        # Execute query
        session.execute(text("SELECT 1"))
    
    # Check session was closed
    mock_db_session.close.assert_called_once()

def test_with_db_session_error(db_config, mock_db_session, mocker):
    """Test using database session context manager with error"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_session)
    
    # Use database session
    with pytest.raises(OperationalError):
        with with_db_session(db_config) as session:
            # Check session was created
            assert session is not None
            assert session == mock_db_session
            
            # Execute query
            raise OperationalError("connection error", None, None)
    
    # Check session was closed
    mock_db_session.close.assert_called_once()

def test_get_connection_pool(db_config, mock_db_connection, mocker):
    """Test getting connection pool"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_connection)
    
    # Get connection pool
    pool = get_connection_pool(db_config)
    
    # Check pool was created
    assert pool is not None
    assert pool.size() == db_config.pool_size

def test_release_connection(db_config, mock_db_connection, mocker):
    """Test releasing connection"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_connection)
    
    # Get connection pool
    pool = get_connection_pool(db_config)
    
    # Get connection
    connection = pool.connect()
    
    # Release connection
    release_connection(connection)
    
    # Check connection was released
    assert pool.size() == db_config.pool_size

def test_get_connection_stats(db_config, mock_db_connection, mocker):
    """Test getting connection statistics"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_connection)
    
    # Get connection pool
    pool = get_connection_pool(db_config)
    
    # Get connection statistics
    stats = get_connection_stats(pool)
    
    # Check statistics were retrieved
    assert stats is not None
    assert "size" in stats
    assert "checkedin" in stats
    assert "checkedout" in stats
    assert "overflow" in stats

def test_connection_timeout(db_config, mock_db_connection, mocker):
    """Test connection timeout"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_connection)
    
    # Get connection pool
    pool = get_connection_pool(db_config)
    
    # Mock connection timeout
    mock_db_connection.execute.side_effect = TimeoutError("connection timeout")
    
    # Get connection
    with pytest.raises(TimeoutError):
        pool.connect()
    
    # Check connection was not created
    assert pool.size() == db_config.pool_size

def test_connection_pool_exhaustion(db_config, mock_db_connection, mocker):
    """Test connection pool exhaustion"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_connection)
    
    # Get connection pool
    pool = get_connection_pool(db_config)
    
    # Exhaust connection pool
    connections = []
    for _ in range(db_config.pool_size + db_config.max_overflow + 1):
        connections.append(pool.connect())
    
    # Check pool is exhausted
    with pytest.raises(OperationalError):
        pool.connect()
    
    # Release connections
    for connection in connections:
        release_connection(connection)
    
    # Check pool is available
    assert pool.size() == db_config.pool_size

def test_connection_pool_recycle(db_config, mock_db_connection, mocker):
    """Test connection pool recycling"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_connection)
    
    # Get connection pool
    pool = get_connection_pool(db_config)
    
    # Get connection
    connection = pool.connect()
    
    # Mock connection expiration
    connection._start_time = datetime.utcnow() - timedelta(seconds=db_config.pool_recycle + 1)
    
    # Release connection
    release_connection(connection)
    
    # Check connection was recycled
    assert pool.size() == db_config.pool_size

def test_connection_pool_pre_ping(db_config, mock_db_connection, mocker):
    """Test connection pool pre-ping"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_connection)
    
    # Get connection pool
    pool = get_connection_pool(db_config)
    
    # Get connection
    connection = pool.connect()
    
    # Mock connection error
    mock_db_connection.execute.side_effect = OperationalError("connection error", None, None)
    
    # Release connection
    release_connection(connection)
    
    # Check connection was invalidated
    assert pool.size() == db_config.pool_size

def test_connection_pool_reset(db_config, mock_db_connection, mocker):
    """Test connection pool reset"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_connection)
    
    # Get connection pool
    pool = get_connection_pool(db_config)
    
    # Reset connection pool
    pool.dispose()
    
    # Check pool was reset
    assert pool.size() == 0

def test_connection_pool_overflow(db_config, mock_db_connection, mocker):
    """Test connection pool overflow"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_connection)
    
    # Get connection pool
    pool = get_connection_pool(db_config)
    
    # Get connections up to max overflow
    connections = []
    for _ in range(db_config.pool_size + db_config.max_overflow):
        connections.append(pool.connect())
    
    # Check pool is at max overflow
    assert pool.size() == db_config.pool_size + db_config.max_overflow
    
    # Release connections
    for connection in connections:
        release_connection(connection)
    
    # Check pool is available
    assert pool.size() == db_config.pool_size

def test_connection_pool_timeout(db_config, mock_db_connection, mocker):
    """Test connection pool timeout"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_connection)
    
    # Get connection pool
    pool = get_connection_pool(db_config)
    
    # Get connection with timeout
    with pytest.raises(TimeoutError):
        pool.connect(timeout=1)
    
    # Check connection was not created
    assert pool.size() == db_config.pool_size

def test_connection_pool_checkout(db_config, mock_db_connection, mocker):
    """Test connection pool checkout"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_connection)
    
    # Get connection pool
    pool = get_connection_pool(db_config)
    
    # Get connection
    connection = pool.connect()
    
    # Check connection was checked out
    assert connection in pool._checkedout
    
    # Release connection
    release_connection(connection)
    
    # Check connection was checked in
    assert connection not in pool._checkedout

def test_connection_pool_checkin(db_config, mock_db_connection, mocker):
    """Test connection pool checkin"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_connection)
    
    # Get connection pool
    pool = get_connection_pool(db_config)
    
    # Get connection
    connection = pool.connect()
    
    # Release connection
    release_connection(connection)
    
    # Check connection was checked in
    assert connection in pool._checkedin

def test_connection_pool_invalidation(db_config, mock_db_connection, mocker):
    """Test connection pool invalidation"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_connection)
    
    # Get connection pool
    pool = get_connection_pool(db_config)
    
    # Get connection
    connection = pool.connect()
    
    # Invalidate connection
    connection.invalidate()
    
    # Release connection
    release_connection(connection)
    
    # Check connection was invalidated
    assert connection not in pool._checkedin

def test_connection_pool_reconnection(db_config, mock_db_connection, mocker):
    """Test connection pool reconnection"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_connection)
    
    # Get connection pool
    pool = get_connection_pool(db_config)
    
    # Get connection
    connection = pool.connect()
    
    # Mock connection error
    mock_db_connection.execute.side_effect = OperationalError("connection error", None, None)
    
    # Release connection
    release_connection(connection)
    
    # Get new connection
    new_connection = pool.connect()
    
    # Check new connection was created
    assert new_connection is not None
    assert new_connection != connection

def test_connection_pool_cleanup(db_config, mock_db_connection, mocker):
    """Test connection pool cleanup"""
    # Mock database engine
    mocker.patch("database.connection.create_engine", return_value=mock_db_connection)
    
    # Get connection pool
    pool = get_connection_pool(db_config)
    
    # Get connections
    connections = []
    for _ in range(db_config.pool_size):
        connections.append(pool.connect())
    
    # Release connections
    for connection in connections:
        release_connection(connection)
    
    # Cleanup connection pool
    pool.dispose()
    
    # Check pool was cleaned up
    assert pool.size() == 0
    assert len(pool._checkedin) == 0
    assert len(pool._checkedout) == 0

def test_get_connection():
    """Test getting a database connection"""
    connection = get_connection()
    try:
        # Check connection was created
        assert connection is not None
        assert connection.engine is not None
        assert connection.pool is not None
        
        # Check connection is alive
        assert is_connection_alive(connection)
    finally:
        close_connection(connection)

def test_connection_pool():
    """Test connection pool management"""
    # Get connection pool
    pool = get_connection_pool()
    
    # Check pool was created
    assert pool is not None
    assert pool.size() > 0
    assert pool.checkedin() == pool.size()
    
    # Get multiple connections
    connections = []
    for _ in range(3):
        connection = get_connection()
        assert connection is not None
        connections.append(connection)
    
    # Check pool status
    assert pool.checkedout() == 3
    assert pool.checkedin() == pool.size() - 3
    
    # Close connections
    for connection in connections:
        close_connection(connection)
    
    # Check all connections were returned to pool
    assert pool.checkedout() == 0
    assert pool.checkedin() == pool.size()

def test_connection_timeout():
    """Test connection timeout"""
    connection = get_connection()
    try:
        # Set connection timeout
        connection.engine.dialect.server_side_cursors = True
        connection.engine.dialect.supports_server_side_cursors = True
        
        # Execute query with timeout
        with pytest.raises(OperationalError):
            with connection.engine.connect() as conn:
                conn.execute(text("SELECT pg_sleep(2)")).fetchall()
    finally:
        close_connection(connection)

def test_connection_retry():
    """Test connection retry mechanism"""
    # Mock connection error
    error_count = 0
    original_connect = create_engine
    
    def mock_connect(*args, **kwargs):
        nonlocal error_count
        if error_count < 2:
            error_count += 1
            raise OperationalError("Connection failed", None, None)
        return original_connect(*args, **kwargs)
    
    # Replace connection function
    create_engine = mock_connect
    
    try:
        # Get connection (should retry and succeed)
        connection = get_connection()
        assert connection is not None
        assert error_count == 2
    finally:
        close_connection(connection)
        create_engine = original_connect

def test_connection_error_handling():
    """Test connection error handling"""
    # Try to connect with invalid credentials
    with pytest.raises(DatabaseError) as excinfo:
        get_connection(
            username="invalid",
            password="invalid",
            host="invalid",
            port=1234,
            database="invalid"
        )
    
    assert "Failed to establish database connection" in str(excinfo.value)

def test_session_management():
    """Test session management"""
    connection = get_connection()
    try:
        # Create session
        session = create_session(connection)
        
        # Check session was created
        assert session is not None
        
        # Execute test query
        result = session.execute(text("SELECT 1")).scalar()
        assert result == 1
        
        # Close session
        session.close()
    finally:
        close_connection(connection)

def test_scoped_session():
    """Test scoped session management"""
    connection = get_connection()
    try:
        # Create scoped session
        Session = scoped_session(sessionmaker(bind=connection.engine))
        
        # Get session
        session1 = Session()
        session2 = Session()
        
        # Check sessions are the same
        assert session1 is session2
        
        # Remove session
        Session.remove()
    finally:
        close_connection(connection)

def test_connection_context_manager():
    """Test connection context manager"""
    @contextmanager
    def managed_connection() -> Generator[DatabaseConnection, None, None]:
        connection = get_connection()
        try:
            yield connection
        finally:
            close_connection(connection)
    
    # Use connection context manager
    with managed_connection() as connection:
        assert connection is not None
        assert is_connection_alive(connection)
    
    # Check connection was closed
    assert not is_connection_alive(connection)

def test_close_all_connections():
    """Test closing all connections"""
    # Create multiple connections
    connections = [get_connection() for _ in range(3)]
    
    # Check connections are alive
    for connection in connections:
        assert is_connection_alive(connection)
    
    # Close all connections
    close_all_connections()
    
    # Check connections were closed
    for connection in connections:
        assert not is_connection_alive(connection)

def test_connection_max_overflow():
    """Test connection pool max overflow"""
    pool = get_connection_pool()
    max_overflow = pool._pool.max_overflow
    
    # Get connections up to max overflow
    connections = []
    try:
        for _ in range(pool.size() + max_overflow):
            connection = get_connection()
            assert connection is not None
            connections.append(connection)
        
        # Try to get one more connection
        with pytest.raises(SQLAlchemyError):
            get_connection()
    finally:
        # Close all connections
        for connection in connections:
            close_connection(connection)

def test_connection_recycle():
    """Test connection recycling"""
    connection = get_connection()
    try:
        # Get creation time
        creation_time = connection.info.get('created_at')
        
        # Wait for recycle interval
        time.sleep(1)
        
        # Execute query to trigger recycle
        with connection.engine.connect() as conn:
            conn.execute(text("SELECT 1")).scalar()
        
        # Check connection was recycled
        assert connection.info.get('created_at') > creation_time
    finally:
        close_connection(connection)

def test_connection_invalidation():
    """Test connection invalidation"""
    connection = get_connection()
    try:
        # Invalidate connection
        connection.invalidate()
        
        # Check connection is not alive
        assert not is_connection_alive(connection)
        
        # Try to use connection
        with pytest.raises(SQLAlchemyError):
            with connection.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
    finally:
        close_connection(connection)

def test_connection_info():
    """Test connection information"""
    connection = get_connection()
    try:
        # Check connection info
        assert connection.info is not None
        assert 'created_at' in connection.info
        assert 'pool_id' in connection.info
        assert 'process_id' in connection.info
    finally:
        close_connection(connection)

def test_connection_events():
    """Test connection events"""
    events = []
    
    def on_connect(dbapi_connection, connection_record):
        events.append('connect')
    
    def on_checkout(dbapi_connection, connection_record, connection_proxy):
        events.append('checkout')
    
    def on_checkin(dbapi_connection, connection_record):
        events.append('checkin')
    
    # Register event listeners
    from sqlalchemy import event
    connection = get_connection()
    try:
        event.listen(connection.engine.pool, 'connect', on_connect)
        event.listen(connection.engine.pool, 'checkout', on_checkout)
        event.listen(connection.engine.pool, 'checkin', on_checkin)
        
        # Use connection
        with connection.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # Check events were fired
        assert 'connect' in events
        assert 'checkout' in events
        assert 'checkin' in events
    finally:
        close_connection(connection) 