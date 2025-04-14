import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import os
import logging
from typing import Generator, Dict, Any

from database.db import Base, get_db
from database.models import User, UserActivity, ActivityType
from main import app
from auth.jwt import create_access_token
from core.neural_mining import NeuralMiningEngine
from core.optimization_engine import OptimizationEngine
from core.prediction_engine import PredictionEngine
from core.reward_engine import RewardEngine
from core.risk_engine import RiskEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test database URL
TEST_DATABASE_URL = "sqlite:///:memory:"

# Create test engine
engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

# Create test session
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session")
def db() -> Generator:
    """Create test database session"""
    Base.metadata.create_all(bind=engine)
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="module")
def client() -> Generator:
    """Create test client"""
    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture(scope="module")
def test_user(db: TestingSessionLocal) -> Dict[str, Any]:
    """Create test user"""
    from auth.jwt import get_password_hash
    
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password=get_password_hash("testpassword"),
        full_name="Test User",
        is_active=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "is_active": user.is_active
    }

@pytest.fixture(scope="module")
def test_user_token(test_user: Dict[str, Any]) -> str:
    """Create test user token"""
    return create_access_token(
        data={
            "sub": test_user["username"],
            "user_id": str(test_user["id"]),
            "scopes": ["user"]
        }
    )

@pytest.fixture(scope="module")
def authorized_client(client: TestClient, test_user_token: str) -> TestClient:
    """Create authorized test client"""
    client.headers = {
        **client.headers,
        "Authorization": f"Bearer {test_user_token}"
    }
    return client

@pytest.fixture(scope="module")
def test_mining_activity(db: TestingSessionLocal, test_user: Dict[str, Any]) -> Dict[str, Any]:
    """Create test mining activity"""
    activity = UserActivity(
        user_id=test_user["id"],
        activity_type=ActivityType.MINING,
        metrics={
            "device_type": "gpu",
            "hash_rate": 100.0,
            "power_usage": 500.0,
            "temperature": 75.0
        },
        rewards=10.0,
        risk_score=0.2,
        efficiency_score=0.8
    )
    db.add(activity)
    db.commit()
    db.refresh(activity)
    
    return {
        "id": activity.id,
        "user_id": activity.user_id,
        "activity_type": activity.activity_type,
        "metrics": activity.metrics,
        "rewards": activity.rewards,
        "risk_score": activity.risk_score,
        "efficiency_score": activity.efficiency_score
    }

@pytest.fixture(scope="module")
def neural_mining_engine() -> NeuralMiningEngine:
    """Create neural mining engine"""
    engine = NeuralMiningEngine()
    return engine

@pytest.fixture(scope="module")
def optimization_engine() -> OptimizationEngine:
    """Create optimization engine"""
    engine = OptimizationEngine()
    return engine

@pytest.fixture(scope="module")
def prediction_engine() -> PredictionEngine:
    """Create prediction engine"""
    engine = PredictionEngine()
    return engine

@pytest.fixture(scope="module")
def reward_engine() -> RewardEngine:
    """Create reward engine"""
    engine = RewardEngine()
    return engine

@pytest.fixture(scope="module")
def risk_engine() -> RiskEngine:
    """Create risk engine"""
    engine = RiskEngine()
    return engine 