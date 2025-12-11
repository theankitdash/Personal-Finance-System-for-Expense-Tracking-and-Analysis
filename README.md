# Personal Finance System for Expense Tracking and Analysis

A containerized financial analytics engine that performs multivariate anomaly detection using LOF, One-Class SVM, and a deep-learning autoencoder—delivering high-precision identification of irregular spending patterns across large transaction datasets. The platform exposes clean REST APIs for ingestion, forecasting, drift analysis, semantic category clustering, and visual diagnostics, with typical response times around 2–3 seconds even at 10,000+ records.

## Features

* **Multivariate Anomaly Detection**: Detects irregular spending patterns using LOF, One-Class SVM, and a deep-learning autoencoder.

* **Forecasting & Drift Analysis**: Predicts category-level expenses and measures data drift using JSD and divergence metrics.

* **Semantic Category Clustering**: Generates sentence-transformer embeddings to group similar expense categories.

* **RESTful APIs**: Provides unified endpoints for ingestion, analytics, and visualization.

* **Containerized Deployment**: Uses Docker for reproducible, scalable execution across environments.

* **Interactive Visualizations**: Offers trend charts, heatmaps, anomaly plots, and category insights.

## Tech Stack

* **Backend**: Node.js
* **API Analysis**: FastAPI (Python)
* **ML frameworks**: Scikit-learn, PyTorch, XGBoost, sentence-transformers
* **Database**: PostgreSQL
* **Containerization**: Docker & Docker Compose

## Getting Started

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/theankitdash/Personal-Finance-System-for-Expense-Tracking-and-Analysis.git
   cd Personal-Finance-System-for-Expense-Tracking-and-Analysis
   ```

2. **Build and Start Containers**:

   ```bash
   docker-compose up --build
   ```

3. **Access the Application**:

   Open your browser and navigate to `http://localhost:3000`.

## Directory Structure

```
Personal-Finance-System-for-Expense-Tracking-and-Analysis/
│
├── node-backend/       # Front-end services (Node.js)
├── pythonapi/          # Backend Services (FastAPI: Python with ML Analysis)
├── .gitignore          # Git ignore file
└── docker-compose.yml  # Docker Compose configuration
```
