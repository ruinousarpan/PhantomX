from sqlalchemy import create_engine, Column, String, Boolean, DateTime, Float, JSON, ForeignKey, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
import os
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/ai_core"
)

# Create engine
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Create base class for models
Base = declarative_base()

# Database models
class UserModel(Base):
    """User database model"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    scopes = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sessions = relationship("SessionModel", back_populates="user")
    activities = relationship("ActivityModel", back_populates="user")

class SessionModel(Base):
    """User session database model"""
    __tablename__ = "sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    device_info = Column(JSON, nullable=False)
    ip_address = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("UserModel", back_populates="sessions")
    activities = relationship("ActivityModel", back_populates="session")

class ActivityModel(Base):
    """User activity database model"""
    __tablename__ = "activities"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    activity_type = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    duration = Column(Integer, nullable=False)  # in seconds
    metrics = Column(JSON, nullable=False)
    status = Column(String, nullable=False)
    rewards = Column(Float, nullable=True)
    risk_score = Column(Float, nullable=True)
    efficiency_score = Column(Float, nullable=True)
    
    # Relationships
    user = relationship("UserModel", back_populates="activities")
    session = relationship("SessionModel", back_populates="activities")

# Database dependency
def get_db():
    """
    Get database session
    
    Yields:
        Session: Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Database operations
class DatabaseOperations:
    """Database operations class"""
    
    @staticmethod
    def create_user(db, user_data: Dict[str, Any]) -> UserModel:
        """Create a new user"""
        db_user = UserModel(**user_data)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    
    @staticmethod
    def get_user_by_id(db, user_id: str) -> Optional[UserModel]:
        """Get user by ID"""
        return db.query(UserModel).filter(UserModel.id == user_id).first()
    
    @staticmethod
    def get_user_by_username(db, username: str) -> Optional[UserModel]:
        """Get user by username"""
        return db.query(UserModel).filter(UserModel.username == username).first()
    
    @staticmethod
    def get_user_by_email(db, email: str) -> Optional[UserModel]:
        """Get user by email"""
        return db.query(UserModel).filter(UserModel.email == email).first()
    
    @staticmethod
    def update_user(db, user_id: str, user_data: Dict[str, Any]) -> Optional[UserModel]:
        """Update user"""
        db_user = DatabaseOperations.get_user_by_id(db, user_id)
        if db_user:
            for key, value in user_data.items():
                setattr(db_user, key, value)
            db.commit()
            db.refresh(db_user)
        return db_user
    
    @staticmethod
    def delete_user(db, user_id: str) -> bool:
        """Delete user"""
        db_user = DatabaseOperations.get_user_by_id(db, user_id)
        if db_user:
            db.delete(db_user)
            db.commit()
            return True
        return False
    
    @staticmethod
    def create_session(db, session_data: Dict[str, Any]) -> SessionModel:
        """Create a new session"""
        db_session = SessionModel(**session_data)
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        return db_session
    
    @staticmethod
    def get_session_by_id(db, session_id: str) -> Optional[SessionModel]:
        """Get session by ID"""
        return db.query(SessionModel).filter(SessionModel.id == session_id).first()
    
    @staticmethod
    def update_session(db, session_id: str, session_data: Dict[str, Any]) -> Optional[SessionModel]:
        """Update session"""
        db_session = DatabaseOperations.get_session_by_id(db, session_id)
        if db_session:
            for key, value in session_data.items():
                setattr(db_session, key, value)
            db.commit()
            db.refresh(db_session)
        return db_session
    
    @staticmethod
    def create_activity(db, activity_data: Dict[str, Any]) -> ActivityModel:
        """Create a new activity"""
        db_activity = ActivityModel(**activity_data)
        db.add(db_activity)
        db.commit()
        db.refresh(db_activity)
        return db_activity
    
    @staticmethod
    def get_user_activities(db, user_id: str, limit: int = 100) -> List[ActivityModel]:
        """Get user activities"""
        return db.query(ActivityModel).filter(ActivityModel.user_id == user_id).limit(limit).all()
    
    @staticmethod
    def get_session_activities(db, session_id: str) -> List[ActivityModel]:
        """Get session activities"""
        return db.query(ActivityModel).filter(ActivityModel.session_id == session_id).all()

def init_db():
    """
    Initialize database
    
    Creates all tables if they don't exist
    """
    try:
        # Import models to ensure they are registered with Base
        from .models import (
            User, UserActivity, UserSession,
            MiningStats, StakingStats, TradingStats, ReferralStats
        )
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

def check_db_connection():
    """
    Check database connection
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        return True
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        return False 