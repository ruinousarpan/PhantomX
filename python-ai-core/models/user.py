from pydantic import BaseModel, EmailStr, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class UserBase(BaseModel):
    """Base user model with common attributes"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: Optional[str] = None
    is_active: bool = True
    scopes: List[str] = []

class UserCreate(UserBase):
    """User creation model with password"""
    password: str = Field(..., min_length=8)
    
    @validator('password')
    def password_strength(cls, v):
        """Validate password strength"""
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in v):
            raise ValueError('Password must contain at least one special character')
        return v

class UserUpdate(BaseModel):
    """User update model with optional fields"""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    scopes: Optional[List[str]] = None

class UserInDB(UserBase):
    """User model for database storage"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        orm_mode = True

class User(UserBase):
    """User model for API responses"""
    id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class UserProfile(User):
    """Extended user profile with additional information"""
    mining_stats: Optional[Dict[str, Any]] = None
    staking_stats: Optional[Dict[str, Any]] = None
    trading_stats: Optional[Dict[str, Any]] = None
    referral_stats: Optional[Dict[str, Any]] = None
    total_rewards: float = 0.0
    risk_score: float = 0.0
    efficiency_score: float = 0.0

class UserActivity(BaseModel):
    """User activity tracking model"""
    user_id: str
    activity_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration: int  # in seconds
    metrics: Dict[str, Any]
    status: str
    rewards: Optional[float] = None
    risk_score: Optional[float] = None
    efficiency_score: Optional[float] = None

class UserSession(BaseModel):
    """User session tracking model"""
    user_id: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    device_info: Dict[str, Any]
    ip_address: str
    is_active: bool = True
    activities: List[UserActivity] = [] 