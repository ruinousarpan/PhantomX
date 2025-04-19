import pytest

from src.database import (
    DatabaseError,
    ConnectionError,
    QueryError,
    ValidationError,
    MigrationError,
    ConfigurationError,
    TransactionError,
    IntegrityError
)


def test_database_error_basic():
    """Test basic DatabaseError functionality."""
    error = DatabaseError("Database error occurred")
    assert str(error) == "Database error occurred"
    assert error.details is None
    assert error.__cause__ is None


def test_database_error_with_details():
    """Test DatabaseError with additional details."""
    details = {"code": "DB001", "severity": "high"}
    error = DatabaseError("Database error occurred", details=details)
    assert str(error) == "Database error occurred"
    assert error.details == details
    assert error.__cause__ is None


def test_database_error_with_cause():
    """Test DatabaseError with a cause."""
    original_error = ValueError("Original error")
    error = DatabaseError("Database error occurred", cause=original_error)
    assert str(error) == "Database error occurred"
    assert error.details is None
    assert error.__cause__ == original_error


def test_database_error_with_details_and_cause():
    """Test DatabaseError with both details and cause."""
    details = {"code": "DB001", "severity": "high"}
    original_error = ValueError("Original error")
    error = DatabaseError("Database error occurred", details=details, cause=original_error)
    assert str(error) == "Database error occurred"
    assert error.details == details
    assert error.__cause__ == original_error


def test_connection_error():
    """Test ConnectionError functionality."""
    error = ConnectionError("Failed to connect to database")
    assert isinstance(error, DatabaseError)
    assert str(error) == "Failed to connect to database"


def test_query_error():
    """Test QueryError functionality."""
    error = QueryError("Invalid SQL query")
    assert isinstance(error, DatabaseError)
    assert str(error) == "Invalid SQL query"


def test_validation_error():
    """Test ValidationError functionality."""
    error = ValidationError("Invalid data format")
    assert isinstance(error, DatabaseError)
    assert str(error) == "Invalid data format"


def test_migration_error():
    """Test MigrationError functionality."""
    error = MigrationError("Migration failed")
    assert isinstance(error, DatabaseError)
    assert str(error) == "Migration failed"


def test_configuration_error():
    """Test ConfigurationError functionality."""
    error = ConfigurationError("Invalid configuration")
    assert isinstance(error, DatabaseError)
    assert str(error) == "Invalid configuration"


def test_transaction_error():
    """Test TransactionError functionality."""
    error = TransactionError("Transaction rollback failed")
    assert isinstance(error, DatabaseError)
    assert str(error) == "Transaction rollback failed"


def test_integrity_error():
    """Test IntegrityError functionality."""
    error = IntegrityError("Unique constraint violation")
    assert isinstance(error, DatabaseError)
    assert str(error) == "Unique constraint violation"


def test_error_inheritance():
    """Test that all database errors inherit from DatabaseError."""
    error_classes = [
        ConnectionError,
        QueryError,
        ValidationError,
        MigrationError,
        ConfigurationError,
        TransactionError,
        IntegrityError
    ]
    
    for error_class in error_classes:
        error = error_class("Test error")
        assert isinstance(error, DatabaseError)
        assert issubclass(error_class, DatabaseError)


def test_error_chaining():
    """Test error chaining functionality."""
    original_error = ValueError("Original error")
    db_error = DatabaseError("Database error", cause=original_error)
    connection_error = ConnectionError("Connection failed", cause=db_error)
    
    assert connection_error.__cause__ == db_error
    assert db_error.__cause__ == original_error 