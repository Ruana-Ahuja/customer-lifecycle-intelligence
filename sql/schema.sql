-- Customer Lifecycle Intelligence Platform
-- Database Schema Documentation

-- Raw transaction data (cleaned)
CREATE TABLE IF NOT EXISTS transactions (
    Invoice         TEXT,
    StockCode       TEXT,
    Description     TEXT,
    Quantity        INTEGER,
    InvoiceDate     DATETIME,
    Price           REAL,
    "Customer ID"   REAL,
    Country         TEXT,
    TotalPrice      REAL
);

-- RFM features with customer segments
CREATE TABLE IF NOT EXISTS rfm_segments (
    "Customer ID"   REAL PRIMARY KEY,
    Recency         INTEGER,
    Frequency       INTEGER,
    Monetary        REAL,
    Cluster         INTEGER,
    Segment         TEXT
);

-- Churn prediction features and labels
CREATE TABLE IF NOT EXISTS churn_features (
    "Customer ID"           REAL PRIMARY KEY,
    Recency                 INTEGER,
    Frequency               INTEGER,
    Monetary                REAL,
    AvgOrderValue           REAL,
    Tenure                  INTEGER,
    AvgDaysBetweenOrders    REAL,
    Churned                 INTEGER
);

-- CLV prediction features and targets
CREATE TABLE IF NOT EXISTS clv_features (
    "Customer ID"           REAL PRIMARY KEY,
    Recency                 INTEGER,
    Frequency               INTEGER,
    Monetary                REAL,
    AvgOrderValue           REAL,
    Tenure                  INTEGER,
    AvgDaysBetweenOrders    REAL,
    CLV                     REAL
);