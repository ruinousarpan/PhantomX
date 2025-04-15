class DatabaseError(Exception):
    """Base exception for all database-related errors.
    
    This exception serves as the parent class for all database-specific exceptions.
    It can include additional details about the error and the original cause.
    
    Args:
        message (str): A descriptive error message
        details (dict, optional): Additional error details. Defaults to None.
        cause (Exception, optional): The original exception that caused this error. Defaults to None.
    """
    def __init__(self, message: str, details: dict = None, cause: Exception = None):
        super().__init__(message)
        self.details = details
        self.__cause__ = cause

class ConnectionError(DatabaseError):
    """Raised when there is an error connecting to the database.
    
    This exception is raised when the application fails to establish a connection
    to the database, such as when the database server is down or the connection
    credentials are invalid.
    """
    pass

class QueryError(DatabaseError):
    """Raised when there is an error executing a database query.
    
    This exception is raised when a database query fails to execute properly,
    such as when the SQL syntax is invalid or the query references non-existent
    tables or columns.
    """
    pass

class ValidationError(DatabaseError):
    """Raised when there is an error validating database data.
    
    This exception is raised when data fails validation checks before being
    inserted or updated in the database, such as when required fields are missing
    or data types are incorrect.
    """
    pass

class MigrationError(DatabaseError):
    """Raised when there is an error during database migrations.
    
    This exception is raised when database schema migrations fail, such as when
    there are conflicts between migration versions or when migration scripts
    contain errors.
    """
    pass

class ConfigurationError(DatabaseError):
    """Raised when there is an error in database configuration.
    
    This exception is raised when there are issues with the database configuration,
    such as missing required settings or invalid configuration values.
    """
    pass

class TransactionError(DatabaseError):
    """Raised when there is an error during database transactions.
    
    This exception is raised when a database transaction fails, such as when a
    rollback operation fails or when there are issues with transaction isolation
    levels.
    """
    pass

class IntegrityError(DatabaseError):
    """Raised when there is a database integrity constraint violation.
    
    This exception is raised when a database operation violates integrity
    constraints, such as unique key violations, foreign key constraints, or
    check constraints.
    """
    pass 