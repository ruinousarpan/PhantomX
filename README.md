# PhantomX Network 🌌

A next-generation blockchain platform combining AI, multi-device mining, quantum-safe cryptography, and interplanetary networking capabilities.

## 🚀 Features

- **Advanced Blockchain Core**: Custom PoS/hybrid consensus mechanism
- **Multi-Device Mining**: Support for mobile, web, desktop, and IoT devices
- **AI Integration**: Neural mining, security analysis, and smart rewards
- **Quantum-Safe Security**: Post-quantum cryptographic layer
- **Interplanetary Support**: Delay-tolerant networking for space nodes
- **VPN Network**: Decentralized VPN with reward mechanisms
- **Bio-Authentication**: DNA-based wallet derivation (experimental)
- **Neural Mining**: BCI integration for proof-of-focus mining

## 🛠️ Project Structure

```
phantomx/
├── blockchain-node/     # Core blockchain implementation
├── api-gateway/         # REST API interface
├── wallet-ui/          # Web wallet interface
├── phantomx-app/       # Mobile application
├── python-ai-core/     # AI services
├── mining-node-system/ # Mining infrastructure
├── phantomx-neural/    # Neural mining integration
├── phantomx-galactic/  # Space networking components
├── phantomx-bioauth/   # Biometric authentication
├── phantomx-vpn/       # Decentralized VPN system
└── smart-contracts/    # Core protocol contracts
```

## 🚦 Getting Started

### Prerequisites

- Go 1.20+
- Node.js 18+
- Python 3.9+
- Docker & Docker Compose

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/phantomx/phantomx-network.git
cd phantomx-network
```

2. Install dependencies:
```bash
# Install Go dependencies
cd blockchain-node
go mod tidy

# Install Node.js dependencies
cd ../wallet-ui
npm install

# Install Python dependencies
cd ../python-ai-core
pip install -r requirements.txt
```

3. Start the development environment:
```bash
docker-compose up
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## 🌟 Roadmap

See ROADMAP.md for the detailed development plan and upcoming features.

# Fee Prediction System

A machine learning-based system for predicting transaction fees based on network conditions and transaction metrics.

## Overview

This system provides tools for:

1. **Data Collection**: Gathering transaction data and network conditions
2. **Data Preprocessing**: Cleaning and preparing data for model training
3. **Model Training**: Training machine learning models to predict fees
4. **Fee Prediction**: Making predictions for new transactions
5. **API and CLI**: Interfaces for interacting with the system

## Components

### Data Collection

The `DataCollector` class collects transaction data and network conditions from various sources.

### Data Preprocessing

The `DataPreprocessor` class handles data cleaning, feature engineering, and preparation for model training.

### Model Training

The `ModelTrainer` class trains machine learning models using preprocessed data.

### Fee Prediction

The `FeePredictionService` class uses trained models to predict fees for new transactions.

### API and CLI

- **API**: FastAPI endpoints for fee prediction
- **CLI**: Command-line interface for fee prediction

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/phantomx.git
   cd phantomx
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   ```

3. Set up environment variables:
   Create a `.env` file in the project root with the following variables:
   ```
   ETH_RPC_URL=your_ethereum_rpc_url
   ETHERSCAN_API_KEY=your_etherscan_api_key
   ```

## Usage

The system provides a unified command-line interface for all operations:

### Data Collection

Collect transaction data from the blockchain:

```
python src/ai/cli.py collect --days 7 --max-transactions 10000
```

### Model Training

Train fee prediction models:

```
python src/ai/cli.py train --n-estimators 100 --max-depth 10 --evaluate
```

### Model Evaluation

Evaluate model performance:

```
python src/ai/cli.py evaluate --save-results
```

### Model Management

Manage model lifecycle:

```
# Register a new model
python src/ai/cli.py model register --model-path models/saved/model_20230101 --model-info '{"version": "1.0.0", "description": "Initial model"}'

# Activate a model
python src/ai/cli.py model activate --model-id model_20230101

# List all models
python src/ai/cli.py model list

# Get model information
python src/ai/cli.py model info --model-id model_20230101
```

### API Server

Run the API server:

```
python src/ai/cli.py api --host 0.0.0.0 --port 8000
```

### Complete Pipeline

Run the entire pipeline from data collection to model evaluation:

```
python src/ai/cli.py pipeline --days 7 --max-transactions 10000 --evaluate
```

## API Endpoints

The API server provides the following endpoints:

- `POST /predict`: Predict fee for a single transaction
- `POST /batch-predict`: Predict fees for multiple transactions
- `POST /update-network-conditions`: Update network conditions
- `GET /fee-history`: Get fee history

Example request:

```json
{
  "transaction_type": "transfer",
  "gas_limit": 21000,
  "data_size": 0,
  "complexity_score": 0.0,
  "priority_level": "medium"
}
```

Example response:

```json
{
  "base_fee": 15.2,
  "priority_fee": 1.5,
  "total_fee": 16.7,
  "confidence_score": 0.85
}
```

## Project Structure

```
phantomx/
├── data/                  # Data directory
│   ├── raw/               # Raw transaction data
│   └── processed/         # Processed data for training
├── models/                # Models directory
│   └── saved/             # Saved trained models
├── src/                   # Source code
│   └── ai/                # AI module
│       ├── api/           # API module
│       ├── data/          # Data collection and preprocessing
│       ├── models/        # Model training and evaluation
│       ├── services/      # Service layer
│       └── schemas/       # Data schemas
├── tests/                 # Test files
├── .env                   # Environment variables
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## Development

### Running Tests

```
pytest
```

### Code Style

The project uses Black for code formatting and isort for import sorting:

```
black src tests
isort src tests
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the Ethereum community for providing the data
- Thanks to the scikit-learn team for the machine learning tools
- Thanks to the FastAPI team for the web framework 