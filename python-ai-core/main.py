from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError
import logging
import time
from typing import Callable
import os

from database.db import init_db, check_db_connection
from routers import auth, neural_mining, optimization, prediction, reward, risk
from core.neural_mining import NeuralMiningEngine
from core.optimization_engine import OptimizationEngine
from core.prediction_engine import PredictionEngine
from core.reward_engine import RewardEngine
from core.risk_engine import RiskEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Core API",
    description="AI-powered mining and optimization platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI engines
neural_mining_engine = NeuralMiningEngine()
optimization_engine = OptimizationEngine()
prediction_engine = PredictionEngine()
reward_engine = RewardEngine()
risk_engine = RiskEngine()

# Add request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next: Callable):
    request_id = str(time.time_ns())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next: Callable):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"Status: {response.status_code} "
        f"Duration: {duration:.2f}s "
        f"Request ID: {request.state.request_id}"
    )
    return response

# Add error handling middleware
@app.middleware("http")
async def error_handling(request: Request, call_next: Callable):
    try:
        return await call_next(request)
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Database error occurred"}
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected error occurred"}
        )

# Register routers
app.include_router(auth.router)
app.include_router(neural_mining.router)
app.include_router(optimization.router)
app.include_router(prediction.router)
app.include_router(reward.router)
app.include_router(risk.router)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Initialize database
        init_db()
        if not check_db_connection():
            raise Exception("Database connection failed")
        logger.info("Database initialized successfully")
        
        # Initialize AI engines
        await neural_mining_engine.initialize_model()
        await optimization_engine.initialize_model()
        await prediction_engine.initialize_model()
        await reward_engine.initialize_model()
        await risk_engine.initialize_model()
        logger.info("AI engines initialized successfully")
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        # Cleanup AI engines
        neural_mining_engine.cleanup()
        optimization_engine.cleanup()
        prediction_engine.cleanup()
        reward_engine.cleanup()
        risk_engine.cleanup()
        logger.info("AI engines cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db_healthy = check_db_connection()
        
        # Check AI engines
        mining_healthy = neural_mining_engine.get_status()["is_operational"]
        optimization_healthy = optimization_engine.get_status()["is_operational"]
        prediction_healthy = prediction_engine.get_status()["is_operational"]
        reward_healthy = reward_engine.get_status()["is_operational"]
        risk_healthy = risk_engine.get_status()["is_operational"]
        
        # Overall health
        healthy = all([
            db_healthy,
            mining_healthy,
            optimization_healthy,
            prediction_healthy,
            reward_healthy,
            risk_healthy
        ])
        
        return {
            "status": "healthy" if healthy else "unhealthy",
            "components": {
                "database": "healthy" if db_healthy else "unhealthy",
                "neural_mining": "healthy" if mining_healthy else "unhealthy",
                "optimization": "healthy" if optimization_healthy else "unhealthy",
                "prediction": "healthy" if prediction_healthy else "unhealthy",
                "reward": "healthy" if reward_healthy else "unhealthy",
                "risk": "healthy" if risk_healthy else "unhealthy"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "AI Core API",
        "version": "1.0.0",
        "description": "AI-powered mining and optimization platform",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

# Validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "request_id": request.state.request_id
        }
    ) 