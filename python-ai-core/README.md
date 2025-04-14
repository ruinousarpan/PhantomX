# AI Core API

AI-powered mining and optimization platform with neural mining, prediction, and risk management capabilities.

## Features

- Neural Mining Engine for mining analysis and optimization
- Optimization Engine for activity optimization
- Prediction Engine for performance forecasting
- Reward Engine for reward calculation and distribution
- Risk Engine for risk assessment and management
- User authentication and authorization
- Activity tracking and statistics
- Real-time monitoring and health checks

## Project Structure

```
python-ai-core/
├── core/                   # Core AI components
│   ├── neural_mining.py    # Neural mining engine
│   ├── optimization_engine.py
│   ├── prediction_engine.py
│   ├── reward_engine.py
│   └── risk_engine.py
├── database/              # Database components
│   ├── db.py             # Database connection
│   ├── models.py         # SQLAlchemy models
│   └── operations.py     # Database operations
├── routers/              # API routers
│   ├── auth.py          # Authentication router
│   ├── neural_mining.py
│   ├── optimization.py
│   ├── prediction.py
│   ├── reward.py
│   └── risk.py
├── config.py            # Configuration settings
├── main.py             # FastAPI application
└── requirements.txt    # Project dependencies
```

## Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Redis (optional, for caching)
- CUDA-capable GPU (optional, for faster AI processing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-core.git
cd ai-core
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file:
```env
# Application
DEBUG=True
SECRET_KEY=your-secret-key

# Database
POSTGRES_SERVER=localhost
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=ai_core
POSTGRES_PORT=5432

# AI Models
MODEL_CACHE_DIR=./models
MODEL_DEVICE=cuda  # or cpu

# Redis (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
```

## Database Setup

1. Create PostgreSQL database:
```bash
createdb ai_core
```

2. Initialize database:
```bash
python -c "from database.db import init_db; init_db()"
```

## Running the Application

1. Start the server:
```bash
uvicorn main:app --reload
```

2. Access the API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Authentication
- POST `/auth/register` - Register new user
- POST `/auth/token` - Login and get access token
- POST `/auth/refresh` - Refresh access token
- GET `/auth/me` - Get current user profile

### Neural Mining
- POST `/mining/analyze` - Analyze mining session
- POST `/mining/optimize` - Optimize mining configuration
- GET `/mining/status` - Get mining engine status

### Optimization
- POST `/optimization/optimize` - Optimize activity
- GET `/optimization/status` - Get optimization engine status

### Prediction
- POST `/prediction/predict` - Predict activity performance
- GET `/prediction/status` - Get prediction engine status

### Reward
- POST `/reward/calculate` - Calculate rewards
- GET `/reward/status` - Get reward engine status

### Risk
- POST `/risk/assess` - Assess activity risk
- GET `/risk/status` - Get risk engine status

## Development

### Code Style
- Follow PEP 8 guidelines
- Use Black for code formatting
- Use isort for import sorting
- Use flake8 for linting
- Use mypy for type checking

### Testing
```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_neural_mining.py
```

### Documentation
```bash
# Generate documentation
mkdocs build

# Serve documentation
mkdocs serve
```

## Monitoring

### Health Checks
- GET `/health` - Check system health
- Monitors database connection and AI engine status

### Metrics
- Prometheus metrics available at `/metrics`
- Request duration, error rates, and system metrics

## Security

- JWT-based authentication
- Password hashing with bcrypt
- CORS protection
- Rate limiting
- Input validation
- SQL injection protection
- XSS protection

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 