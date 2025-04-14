import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import text
from alembic import command
from alembic.config import Config

from database.models import (
    User,
    Activity,
    Reward,
    MiningActivity,
    StakingActivity,
    TradingActivity,
    MiningReward,
    StakingReward,
    TradingReward
)
from database.exceptions import DatabaseError
from database.migrations import (
    get_current_version,
    upgrade_database,
    downgrade_database,
    create_migration,
    apply_migration,
    rollback_migration
)

@pytest.fixture
def alembic_config():
    """Create Alembic configuration"""
    config = Config()
    config.set_main_option("script_location", "database/migrations")
    config.set_main_option("sqlalchemy.url", "postgresql://user:password@localhost/test_db")
    return config

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
        "activity_type": "mining",
        "start_time": datetime.utcnow(),
        "end_time": datetime.utcnow() + timedelta(hours=1),
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
def setup_test_data(db_connection, test_user_data, test_mining_activity_data):
    """Set up test data for migration tests"""
    # Create user
    user = User(**test_user_data)
    db_connection.add(user)
    db_connection.commit()
    
    # Create mining activity
    mining_activity = MiningActivity(**test_mining_activity_data)
    db_connection.add(mining_activity)
    db_connection.commit()
    
    return {
        "user": user,
        "mining_activity": mining_activity
    }

def test_get_current_version(alembic_config):
    """Test getting current database version"""
    # Get current version
    version = get_current_version(alembic_config)
    
    # Check version was retrieved
    assert version is not None
    assert isinstance(version, str)

def test_upgrade_database(alembic_config):
    """Test upgrading database"""
    # Get current version
    current_version = get_current_version(alembic_config)
    
    # Upgrade database
    upgrade_database(alembic_config)
    
    # Get new version
    new_version = get_current_version(alembic_config)
    
    # Check database was upgraded
    assert new_version != current_version

def test_downgrade_database(alembic_config):
    """Test downgrading database"""
    # Get current version
    current_version = get_current_version(alembic_config)
    
    # Downgrade database
    downgrade_database(alembic_config)
    
    # Get new version
    new_version = get_current_version(alembic_config)
    
    # Check database was downgraded
    assert new_version != current_version

def test_create_migration(alembic_config):
    """Test creating migration"""
    # Create migration
    migration = create_migration(alembic_config, "test_migration")
    
    # Check migration was created
    assert migration is not None
    assert isinstance(migration, str)

def test_apply_migration(alembic_config):
    """Test applying migration"""
    # Create migration
    migration = create_migration(alembic_config, "test_migration")
    
    # Apply migration
    apply_migration(alembic_config, migration)
    
    # Get current version
    current_version = get_current_version(alembic_config)
    
    # Check migration was applied
    assert current_version == migration

def test_rollback_migration(alembic_config):
    """Test rolling back migration"""
    # Create migration
    migration = create_migration(alembic_config, "test_migration")
    
    # Apply migration
    apply_migration(alembic_config, migration)
    
    # Rollback migration
    rollback_migration(alembic_config, migration)
    
    # Get current version
    current_version = get_current_version(alembic_config)
    
    # Check migration was rolled back
    assert current_version != migration

def test_migration_add_column(alembic_config, db_connection, setup_test_data):
    """Test migration adding column"""
    # Create migration
    migration = create_migration(alembic_config, "add_test_column")
    
    # Add column to User model
    with db_connection.begin():
        db_connection.execute(text("ALTER TABLE users ADD COLUMN test_column VARCHAR(255)"))
    
    # Apply migration
    apply_migration(alembic_config, migration)
    
    # Check column was added
    with db_connection.begin():
        result = db_connection.execute(text("SELECT test_column FROM users")).fetchone()
        assert result is not None

def test_migration_remove_column(alembic_config, db_connection, setup_test_data):
    """Test migration removing column"""
    # Create migration
    migration = create_migration(alembic_config, "remove_test_column")
    
    # Remove column from User model
    with db_connection.begin():
        db_connection.execute(text("ALTER TABLE users DROP COLUMN test_column"))
    
    # Apply migration
    apply_migration(alembic_config, migration)
    
    # Check column was removed
    with db_connection.begin():
        with pytest.raises(Exception):
            db_connection.execute(text("SELECT test_column FROM users"))

def test_migration_modify_column(alembic_config, db_connection, setup_test_data):
    """Test migration modifying column"""
    # Create migration
    migration = create_migration(alembic_config, "modify_test_column")
    
    # Modify column in User model
    with db_connection.begin():
        db_connection.execute(text("ALTER TABLE users ALTER COLUMN email TYPE VARCHAR(512)"))
    
    # Apply migration
    apply_migration(alembic_config, migration)
    
    # Check column was modified
    with db_connection.begin():
        result = db_connection.execute(text("SELECT column_name, data_type, character_maximum_length FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'email'")).fetchone()
        assert result is not None
        assert result.data_type == "character varying"
        assert result.character_maximum_length == 512

def test_migration_add_index(alembic_config, db_connection, setup_test_data):
    """Test migration adding index"""
    # Create migration
    migration = create_migration(alembic_config, "add_test_index")
    
    # Add index to User model
    with db_connection.begin():
        db_connection.execute(text("CREATE INDEX idx_users_email ON users (email)"))
    
    # Apply migration
    apply_migration(alembic_config, migration)
    
    # Check index was added
    with db_connection.begin():
        result = db_connection.execute(text("SELECT indexname FROM pg_indexes WHERE tablename = 'users' AND indexname = 'idx_users_email'")).fetchone()
        assert result is not None

def test_migration_remove_index(alembic_config, db_connection, setup_test_data):
    """Test migration removing index"""
    # Create migration
    migration = create_migration(alembic_config, "remove_test_index")
    
    # Remove index from User model
    with db_connection.begin():
        db_connection.execute(text("DROP INDEX idx_users_email"))
    
    # Apply migration
    apply_migration(alembic_config, migration)
    
    # Check index was removed
    with db_connection.begin():
        result = db_connection.execute(text("SELECT indexname FROM pg_indexes WHERE tablename = 'users' AND indexname = 'idx_users_email'")).fetchone()
        assert result is None

def test_migration_add_constraint(alembic_config, db_connection, setup_test_data):
    """Test migration adding constraint"""
    # Create migration
    migration = create_migration(alembic_config, "add_test_constraint")
    
    # Add constraint to User model
    with db_connection.begin():
        db_connection.execute(text("ALTER TABLE users ADD CONSTRAINT chk_users_email CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$')"))
    
    # Apply migration
    apply_migration(alembic_config, migration)
    
    # Check constraint was added
    with db_connection.begin():
        result = db_connection.execute(text("SELECT constraint_name FROM information_schema.table_constraints WHERE table_name = 'users' AND constraint_name = 'chk_users_email'")).fetchone()
        assert result is not None

def test_migration_remove_constraint(alembic_config, db_connection, setup_test_data):
    """Test migration removing constraint"""
    # Create migration
    migration = create_migration(alembic_config, "remove_test_constraint")
    
    # Remove constraint from User model
    with db_connection.begin():
        db_connection.execute(text("ALTER TABLE users DROP CONSTRAINT chk_users_email"))
    
    # Apply migration
    apply_migration(alembic_config, migration)
    
    # Check constraint was removed
    with db_connection.begin():
        result = db_connection.execute(text("SELECT constraint_name FROM information_schema.table_constraints WHERE table_name = 'users' AND constraint_name = 'chk_users_email'")).fetchone()
        assert result is None

def test_migration_add_foreign_key(alembic_config, db_connection, setup_test_data):
    """Test migration adding foreign key"""
    # Create migration
    migration = create_migration(alembic_config, "add_test_foreign_key")
    
    # Add foreign key to Activity model
    with db_connection.begin():
        db_connection.execute(text("ALTER TABLE activities ADD CONSTRAINT fk_activities_user_id FOREIGN KEY (user_id) REFERENCES users (user_id)"))
    
    # Apply migration
    apply_migration(alembic_config, migration)
    
    # Check foreign key was added
    with db_connection.begin():
        result = db_connection.execute(text("SELECT constraint_name FROM information_schema.table_constraints WHERE table_name = 'activities' AND constraint_name = 'fk_activities_user_id'")).fetchone()
        assert result is not None

def test_migration_remove_foreign_key(alembic_config, db_connection, setup_test_data):
    """Test migration removing foreign key"""
    # Create migration
    migration = create_migration(alembic_config, "remove_test_foreign_key")
    
    # Remove foreign key from Activity model
    with db_connection.begin():
        db_connection.execute(text("ALTER TABLE activities DROP CONSTRAINT fk_activities_user_id"))
    
    # Apply migration
    apply_migration(alembic_config, migration)
    
    # Check foreign key was removed
    with db_connection.begin():
        result = db_connection.execute(text("SELECT constraint_name FROM information_schema.table_constraints WHERE table_name = 'activities' AND constraint_name = 'fk_activities_user_id'")).fetchone()
        assert result is None

def test_migration_data_transformation(alembic_config, db_connection, setup_test_data):
    """Test migration data transformation"""
    # Create migration
    migration = create_migration(alembic_config, "transform_test_data")
    
    # Transform data in User model
    with db_connection.begin():
        db_connection.execute(text("UPDATE users SET email = LOWER(email)"))
    
    # Apply migration
    apply_migration(alembic_config, migration)
    
    # Check data was transformed
    with db_connection.begin():
        result = db_connection.execute(text("SELECT email FROM users WHERE user_id = 'test_user'")).fetchone()
        assert result is not None
        assert result.email == "test@example.com"

def test_migration_data_validation(alembic_config, db_connection, setup_test_data):
    """Test migration data validation"""
    # Create migration
    migration = create_migration(alembic_config, "validate_test_data")
    
    # Validate data in User model
    with db_connection.begin():
        result = db_connection.execute(text("SELECT COUNT(*) FROM users WHERE email !~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'")).fetchone()
        assert result[0] == 0
    
    # Apply migration
    apply_migration(alembic_config, migration)
    
    # Check data was validated
    with db_connection.begin():
        result = db_connection.execute(text("SELECT COUNT(*) FROM users WHERE email !~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'")).fetchone()
        assert result[0] == 0

def test_migration_rollback_data(alembic_config, db_connection, setup_test_data):
    """Test migration rollback data"""
    # Create migration
    migration = create_migration(alembic_config, "rollback_test_data")
    
    # Backup data
    with db_connection.begin():
        db_connection.execute(text("CREATE TABLE users_backup AS SELECT * FROM users"))
    
    # Transform data
    with db_connection.begin():
        db_connection.execute(text("UPDATE users SET email = LOWER(email)"))
    
    # Apply migration
    apply_migration(alembic_config, migration)
    
    # Rollback migration
    rollback_migration(alembic_config, migration)
    
    # Restore data
    with db_connection.begin():
        db_connection.execute(text("TRUNCATE TABLE users"))
        db_connection.execute(text("INSERT INTO users SELECT * FROM users_backup"))
        db_connection.execute(text("DROP TABLE users_backup"))
    
    # Check data was restored
    with db_connection.begin():
        result = db_connection.execute(text("SELECT email FROM users WHERE user_id = 'test_user'")).fetchone()
        assert result is not None
        assert result.email == "test@example.com"

def test_migration_error_handling(alembic_config, db_connection, setup_test_data):
    """Test migration error handling"""
    # Create migration
    migration = create_migration(alembic_config, "error_test_migration")
    
    # Try to apply invalid migration
    with pytest.raises(Exception):
        with db_connection.begin():
            db_connection.execute(text("ALTER TABLE users ADD COLUMN invalid_column INVALID_TYPE"))
    
    # Check migration was not applied
    with db_connection.begin():
        with pytest.raises(Exception):
            db_connection.execute(text("SELECT invalid_column FROM users"))

def test_migration_concurrent_access(alembic_config, db_connection, setup_test_data):
    """Test migration concurrent access"""
    # Create migration
    migration = create_migration(alembic_config, "concurrent_test_migration")
    
    # Start transaction
    with db_connection.begin():
        # Lock table
        db_connection.execute(text("LOCK TABLE users IN SHARE MODE"))
        
        # Try to apply migration in another transaction
        with pytest.raises(Exception):
            with db_connection.begin():
                db_connection.execute(text("ALTER TABLE users ADD COLUMN test_column VARCHAR(255)"))
    
    # Check migration was not applied
    with db_connection.begin():
        with pytest.raises(Exception):
            db_connection.execute(text("SELECT test_column FROM users"))

def test_migration_performance(alembic_config, db_connection, setup_test_data):
    """Test migration performance"""
    # Create migration
    migration = create_migration(alembic_config, "performance_test_migration")
    
    # Measure migration time
    start_time = datetime.utcnow()
    
    # Apply migration
    apply_migration(alembic_config, migration)
    
    end_time = datetime.utcnow()
    migration_time = (end_time - start_time).total_seconds()
    
    # Check migration time is within acceptable range
    assert migration_time < 5.0  # 5 seconds

def test_migration_idempotency(alembic_config, db_connection, setup_test_data):
    """Test migration idempotency"""
    # Create migration
    migration = create_migration(alembic_config, "idempotency_test_migration")
    
    # Apply migration first time
    apply_migration(alembic_config, migration)
    
    # Apply migration second time
    apply_migration(alembic_config, migration)
    
    # Check database state is consistent
    with db_connection.begin():
        result = db_connection.execute(text("SELECT version_num FROM alembic_version")).fetchone()
        assert result is not None
        assert result.version_num == migration 