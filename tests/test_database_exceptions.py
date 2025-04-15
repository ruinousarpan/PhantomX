import pytest
from src.database.exceptions import (
    DatabaseError,
    ConnectionError,
    QueryError,
    ValidationError,
    MigrationError,
    ConfigurationError,
    TransactionError,
    IntegrityError
)

def test_database_error():
    """Test base DatabaseError exception"""
    error_msg = "Test database error"
    error = DatabaseError(error_msg)
    assert str(error) == error_msg
    assert isinstance(error, Exception)

def test_connection_error():
    """Test ConnectionError exception"""
    error_msg = "Failed to connect to database"
    error = ConnectionError(error_msg)
    assert str(error) == error_msg
    assert isinstance(error, DatabaseError)
    assert isinstance(error, Exception)

def test_query_error():
    """Test QueryError exception"""
    error_msg = "Invalid SQL query"
    error = QueryError(error_msg)
    assert str(error) == error_msg
    assert isinstance(error, DatabaseError)
    assert isinstance(error, Exception)

def test_validation_error():
    """Test ValidationError exception"""
    error_msg = "Data validation failed"
    error = ValidationError(error_msg)
    assert str(error) == error_msg
    assert isinstance(error, DatabaseError)
    assert isinstance(error, Exception)

def test_migration_error():
    """Test MigrationError exception"""
    error_msg = "Migration failed"
    error = MigrationError(error_msg)
    assert str(error) == error_msg
    assert isinstance(error, DatabaseError)
    assert isinstance(error, Exception)

def test_configuration_error():
    """Test ConfigurationError exception"""
    error_msg = "Invalid database configuration"
    error = ConfigurationError(error_msg)
    assert str(error) == error_msg
    assert isinstance(error, DatabaseError)
    assert isinstance(error, Exception)

def test_transaction_error():
    """Test TransactionError exception"""
    error_msg = "Transaction rollback failed"
    error = TransactionError(error_msg)
    assert str(error) == error_msg
    assert isinstance(error, DatabaseError)
    assert isinstance(error, Exception)

def test_integrity_error():
    """Test IntegrityError exception"""
    error_msg = "Unique constraint violation"
    error = IntegrityError(error_msg)
    assert str(error) == error_msg
    assert isinstance(error, DatabaseError)
    assert isinstance(error, Exception)

def test_exception_inheritance():
    """Test exception inheritance hierarchy"""
    # Test that all specific exceptions inherit from DatabaseError
    exceptions = [
        ConnectionError,
        QueryError,
        ValidationError,
        MigrationError,
        ConfigurationError,
        TransactionError,
        IntegrityError
    ]
    
    for exc in exceptions:
        assert issubclass(exc, DatabaseError)
        assert issubclass(exc, Exception)

def test_exception_with_details():
    """Test exceptions with additional details"""
    details = {"code": "ERR001", "severity": "high"}
    error = DatabaseError("Test error", details)
    assert str(error) == "Test error"
    assert hasattr(error, "details")
    assert error.details == details

def test_exception_chaining():
    """Test exception chaining"""
    original_error = ValueError("Original error")
    error = DatabaseError("Database error", cause=original_error)
    assert error.__cause__ == original_error
    assert isinstance(error.__cause__, ValueError) 