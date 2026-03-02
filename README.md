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
                     ┌──────────────────────────────────────┐
                     │         Client / Browser UI          │
                     │  index.html ─ home.html ─ expenses   │
                     │  account-settings ─ report-analysis  │
                     └──────────────────┬───────────────────┘
                                        │  HTTP (Port 3000)
                                        ▼
              ┌─────────────────────────────────────────────────────┐
              │              Node.js Backend (Express 5)            │
              │                                                     │
              │  Middleware           Routes                        │
              │  ┌──────────────┐    ┌────────────────────────┐    │
              │  │ auth.js      │    │ auth.routes.js         │    │
              │  │ rateLimiter  │    │ user.routes.js         │    │
              │  │ validation   │    │ expense.routes.js      │    │
              │  │ session      │    │ budget.routes.js       │    │
              │  │ cookieParser │    └────────────────────────┘    │
              │  └──────────────┘                                   │
              └──────────┬────────────────────────┬─────────────────┘
                         │                        │
          SQL (Port 5432)│                        │ HTTP POST /analyze
                         ▼                        ▼
    ┌────────────────────────────┐   ┌──────────────────────────────────────┐
    │     PostgreSQL 16          │   │       Python ML API (FastAPI)        │
    │                            │   │                                      │
    │  Tables:                   │   │  ┌─────────────────────────────────┐ │
    │  ┌──────────────────────┐  │   │  │          ML Pipeline            │ │
    │  │ credentials          │  │   │  │  ┌───────────┐ ┌────────────┐  │ │
    │  │  phone, password     │  │   │  │  │ DataPrep  │ │ AnomalyML  │  │ │
    │  ├──────────────────────┤  │   │  │  │           │ │ LOF, OCSVM │  │ │
    │  │ personal_details     │  │   │  │  │ Features  │ │ Autoencoder│  │ │
    │  │  phone, name, gender │  │   │  │  └───────────┘ └────────────┘  │ │
    │  │  date_of_birth       │  │   │  │  ┌───────────┐ ┌────────────┐  │ │
    │  ├──────────────────────┤  │   │  │  │RegresserML│ │ ClusterML  │  │ │
    │  │ budget               │  │   │  │  │ RF, GB,   │ │ KMeans +   │  │ │
    │  │  phone, category,    │  │   │  │  │ XGBoost   │ │ Sentence   │  │ │
    │  │  amount              │  │   │  │  └───────────┘ │ Transformer│  │ │
    │  ├──────────────────────┤  │   │  │  ┌───────────┐ └────────────┘  │ │
    │  │ expenses             │  │   │  │  │ DriftML   │ ┌────────────┐  │ │
    │  │  id, phone, date,    │  │   │  │  │ Jensen-   │ │ Visualizer │  │ │
    │  │  amount, description │  │   │  │  │ Shannon   │ │ Matplotlib │  │ │
    │  │  category            │  │   │  │  └───────────┘ └────────────┘  │ │
    │  └──────────────────────┘  │   │  └─────────────────────────────────┘ │
    │                            │   │                                      │
    │  Indexes:                  │   │  Reports ──► Excel Generation        │
    │  idx_expenses_phone_date   │   │              (openpyxl + charts)     │
    │  idx_budget_phone          │   │                                      │
    └────────────────────────────┘   └──────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────┐
    │                     Docker Compose Orchestration                     │
    │                                                                      │
    │  postgresql ──► python-api ──► node-server                           │
    │  (health: pg_isready)  (health: /health)  (health: /health)         │
    │  CPU: 1.0 / 1G         CPU: 2.0 / 2G     CPU: 1.0 / 1G            │
    │                                                                      │
    │  Volume: postgres_data        Volume: ./pythonapi/ml_models          │
    └──────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
Personal-Finance-System-for-Expense-Tracking-and-Analysis/
│
├── docker-compose.yml               # Multi-container orchestration (3 services)
├── .env                              # Environment variables
├── .env.example                      # Environment template
├── .gitignore                        # Git ignore rules
│
├── node-backend/                     # Node.js Backend Service
│   ├── Dockerfile                    # Node container build config
│   ├── .dockerignore
│   ├── package.json                  # Dependencies & scripts
│   ├── server.js                     # Express app entry point
│   │
│   ├── config/
│   │   ├── database.js               # PostgreSQL pool + table creation
│   │   └── session.js                # Express session config
│   │
│   ├── middleware/
│   │   ├── auth.js                   # JWT authentication middleware
│   │   ├── rateLimiter.js            # API rate limiting
│   │   └── validation.js             # Request validation
│   │
│   ├── routes/
│   │   ├── auth.routes.js            # Login / Register / Logout
│   │   ├── user.routes.js            # User profile management
│   │   ├── expense.routes.js         # CRUD operations for expenses
│   │   └── budget.routes.js          # Budget allocation & tracking
│   │
│   ├── utils/
│   │   └── envValidator.js           # Environment variable checks
│   │
│   └── public/                       # Static frontend assets
│       ├── index.html                # Login / Register page
│       ├── home.html                 # Dashboard
│       ├── expenses.html             # Expense management UI
│       ├── account-settings.html     # User settings
│       ├── report-analysis.html      # ML analysis trigger
│       │
│       ├── css/
│       │   ├── common.css            # Shared styles
│       │   ├── style.css             # Login / Register styles
│       │   ├── home.css              # Dashboard styles
│       │   ├── expenses.css          # Expense page styles
│       │   ├── account-settings.css  # Settings styles
│       │   └── report-analysis.css   # Report page styles
│       │
│       └── js/
│           ├── utils.js              # Shared utilities & auth helpers
│           ├── script.js             # Login / Register logic
│           ├── home.js               # Dashboard logic
│           ├── expenses.js           # Expense CRUD logic
│           ├── account-settings.js   # Settings logic
│           └── report-analysis.js    # ML report trigger
│
└── pythonapi/                        # Python ML Analysis Service
    ├── Dockerfile                    # Python container build config
    ├── .dockerignore
    ├── requirements.txt              # Python dependencies (pinned)
    ├── main.py                       # FastAPI app entry point + CORS
    │
    ├── config/
    │   └── config_db.py              # DB connection & env config
    │
    ├── models/
    │   └── models.py                 # Pydantic request schemas
    │
    ├── routes/
    │   └── router.py                 # POST /analyze endpoint
    │
    ├── utils/
    │   ├── data_prep.py              # DataPrepML: feature engineering
    │   └── pipeline.py               # Orchestrates all ML services
    │
    ├── services/
    │   ├── anomaly.py                # AnomalyML: LOF, OCSVM, Autoencoder
    │   ├── regresser.py              # RegresserML: RF, GB, XGBoost
    │   ├── cluster.py                # ClusterML: KMeans + SentenceTransformer
    │   ├── drift.py                  # DriftDetectionML: Jensen-Shannon
    │   └── visualizer.py             # Matplotlib chart generation
    │
    ├── reports/
    │   ├── excel_gen.py              # Excel workbook builder (openpyxl)
    │   └── charts/                   # Generated chart images
    │
    └── ml_models/                    # Persisted trained models
        ├── lof.joblib                # Local Outlier Factor
        ├── ocsvm.joblib              # One-Class SVM
        ├── autoencoder_pytorch.pth   # PyTorch Autoencoder weights
        ├── scaler.joblib             # StandardScaler
        └── regressor_*.joblib        # Per-category regressors
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
