# Personal Finance System for Expense Tracking and Analysis

A containerized financial analytics platform that utilizes the Isolation Forest algorithm to detect anomalies with 80% precision across 10,000+ transactions. The system employs REST APIs for data retrieval and visualization, offering a response time of approximately 2.5 seconds.

## Features

* **Anomaly Detection**: Applies Isolation Forest to identify outliers in financial transactions.
* **RESTful APIs**: Facilitates seamless data retrieval and interaction.
* **Data Visualization**: Presents financial data through intuitive charts and graphs.
* **Containerization**: Utilizes Docker for consistent and scalable deployment.

## Tech Stack

* **Backend**: FastAPI
* **Data Analysis**: Python (scikit-learn)
* **Containerization**: Docker & Docker Compose
* **API Framework**: FastAPI (Python)
* **Frontend**: Node.js
* **Database**: PostgreSQL

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
├── node-backend/       # Backend services (Node.js)
├── pythonapi/          # Python-based anomaly detection and analysis
├── .gitignore          # Git ignore file
└── docker-compose.yml  # Docker Compose configuration
```

* financial-data
* finance-management
* anomaly-detection
* fastapi
* expense-analysis
