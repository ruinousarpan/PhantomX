# Getting Started

This guide will help you get up and running with Python AI Core quickly.

## Prerequisites

- Python 3.9 or higher
- PostgreSQL 12 or higher
- pip (Python package installer)

## Installation

1. Install the package using pip:

```bash
pip install python-ai-core
```

2. Set up your environment variables:

```bash
export AI_CORE_DB_URL="postgresql://user:password@localhost:5432/ai_core"
export AI_CORE_SECRET_KEY="your-secret-key"
export AI_CORE_ENV="development"
```

3. Initialize the database:

```bash
python -m python_ai_core db init
python -m python_ai_core db migrate
python -m python_ai_core db upgrade
```

## Basic Usage

### 1. Data Validation

```python
from python_ai_core import DataValidator

# Create a validator
validator = DataValidator()

# Define a schema
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0}
    }
}

# Validate data
data = {"name": "John", "age": 30}
result = validator.validate(data, schema)
```

### 2. Data Lineage

```python
from python_ai_core import LineageTracker

# Initialize tracker
tracker = LineageTracker()

# Track data transformation
with tracker.track("transform_data"):
    # Your data transformation code here
    transformed_data = process_data(input_data)
```

### 3. Data Versioning

```python
from python_ai_core import VersionManager

# Initialize version manager
version_mgr = VersionManager()

# Create new version
version = version_mgr.create_version(
    data=processed_data,
    version="1.0.0",
    description="Initial version"
)
```

## Configuration

Create a configuration file `config.yml`:

```yaml
database:
  url: postgresql://user:password@localhost:5432/ai_core
  pool_size: 5

security:
  secret_key: your-secret-key
  token_expiry: 3600

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Next Steps

- Learn about [advanced features](../api/index.md)
- Set up [monitoring](../api/monitoring.md)
- Configure [backup strategies](../api/backup.md)
- Explore [analytics capabilities](../api/analytics.md)

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   ```
   Check your database URL and ensure PostgreSQL is running
   ```

2. **Authentication Failed**
   ```
   Verify your secret key and token configuration
   ```

3. **Version Conflict**
   ```
   Ensure you're using compatible versions of dependencies
   ```

## Getting Help

- Check our [FAQ](../guides/faq.md)
- Join our [Discord community](https://discord.gg/example)
- Open an issue on [GitHub](https://github.com/example/python-ai-core) 