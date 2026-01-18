# Personal Finance System for Expense Tracking and Analysis

A containerized financial analytics engine that performs multivariate anomaly detection using LOF, One-Class SVM, and a deep-learning autoencoder—delivering high-precision identification of irregular spending patterns across large transaction datasets. The platform exposes clean REST APIs for ingestion, forecasting, drift analysis, semantic category clustering, and visual diagnostics, with typical response times around 5 seconds even at 10,000+ records.

## Features

* **Multivariate Anomaly Detection**: Detects irregular spending patterns using LOF, One-Class SVM, and a deep-learning autoencoder
* **Forecasting & Drift Analysis**: Predicts category-level expenses and measures data drift using JSD and divergence metrics
* **Semantic Category Clustering**: Generates sentence-transformer embeddings to group similar expense categories
* **User Authentication**: Secure JWT-based authentication with bcrypt password hashing and session management
* **Expense Management**: Complete CRUD operations for expense tracking with filtering and history
* **Budget Tracking**: Category-based budget allocation and monitoring
* **RESTful APIs**: Unified endpoints for ingestion, analytics, and visualization
* **Containerized Deployment**: Docker-based architecture for reproducible, scalable execution
* **Interactive Analytics**: Generate comprehensive Excel reports with trend charts, heatmaps, and anomaly plots

## Tech Stack

* **Backend**: Node.js with Express.js
* **ML Analysis API**: FastAPI (Python)
* **ML Frameworks**: Scikit-learn, PyTorch, XGBoost, sentence-transformers
* **Database**: PostgreSQL 16
* **Authentication**: JWT, bcrypt, express-session
* **Containerization**: Docker & Docker Compose

## Prerequisites

Before you begin, ensure you have the following installed:

* **Docker**: Version 20.10 or higher
* **Docker Compose**: Version 2.0 or higher
* **Git**: For cloning the repository

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/theankitdash/Personal-Finance-System-for-Expense-Tracking-and-Analysis.git
cd Personal-Finance-System-for-Expense-Tracking-and-Analysis
```

### 2. Build and Start Services

```bash
docker-compose up --build
```

This will start three services:
- **PostgreSQL Database** on port `5432`
- **Python ML API** on port `8000`
- **Node.js Backend** on port `3000`

### 3. Access the Application

Open your browser and navigate to:
```
http://localhost:3000
```

## Architecture

```
┌─────────────────┐
│   Client/UI     │
│  (Port 3000)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Node.js Backend (Express)     │
│   - Authentication (JWT)        │
│   - Expense Management          │
│   - Budget Management           │
│   - User Profile                │
└──────┬──────────────────┬───────┘
       │                  │
       ▼                  ▼
┌──────────────┐   ┌─────────────────────┐
│  PostgreSQL  │   │  Python ML API      │
│  Database    │   │  (FastAPI)          │
│              │   │  - Anomaly Detection│
│  - Users     │   │  - Forecasting      │
│  - Expenses  │   │  - Clustering       │
│  - Budgets   │   │  - Excel Reports    │
└──────────────┘   └─────────────────────┘
```

## Directory Structure

```
Personal-Finance-System-for-Expense-Tracking-and-Analysis/
│
├── node-backend/              # Node.js Backend Service
├── pythonapi/                 # Python ML Analysis Service
├── .env                       # Environment variables
├── .gitignore                 # Git ignore rules
└── docker-compose.yml         # Multi-container orchestration
```

### Running in Development Mode

For local development without Docker:

#### Backend (Node.js)

```bash
cd node-backend
npm install
# Update .env to use localhost instead of service names
node server.js
```

#### Python ML API

```bash
cd pythonapi
pip install -r requirements.txt
# Update .env to use localhost instead of service names
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Database

Make sure PostgreSQL is running locally and update `.env`:
```env
PG_HOST=localhost
PYTHON_API_URL=http://localhost:8000
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or support, please open an issue on GitHub.
