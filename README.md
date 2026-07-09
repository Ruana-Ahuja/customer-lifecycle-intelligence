**Live Dashboard:** [Customer Lifecycle Intelligence Platform](https://customer-lifecycle-intelligence-iz9biuue45lgtp5xkoe29t.streamlit.app/)

# Customer Lifecycle Intelligence Platform

An end-to-end Data Science solution built on real retail transaction data to segment customers, predict churn, forecast customer lifetime value, and explain model decisions using SHAP.

---

## Business Problem

Retail businesses lose revenue by treating all customers identically. This platform uses machine learning to:
- Identify distinct customer segments based on purchasing behavior
- Predict which customers are at risk of churning
- Forecast future revenue per customer (CLV)
- Explain every prediction in human-interpretable terms

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.13 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Database | SQLite, SQL |
| Dashboard | Streamlit |
| Version Control | Git, GitHub |

---

## Project Architecture

- `src/data_cleaning.py` — Data ingestion and cleaning pipeline
- `src/eda.py` — Exploratory data analysis and visualization
- `src/feature_engineering.py` — RFM, churn, and CLV feature construction
- `src/segmentation.py` — K-Means clustering with elbow method
- `src/churn_model.py` — XGBoost churn classifier with sklearn Pipeline
- `src/clv_model.py` — XGBoost CLV regressor with sklearn Pipeline
- `src/shap_explainer.py` — SHAP explainability for both models
- `sql/schema.sql` — Database schema documentation
- `sql/analytics.sql` — 6 business analytics queries
- `dashboard/app.py` — Streamlit interactive dashboard
- `main.py` — End-to-end pipeline runner

---

## Dataset

**UCI Online Retail II Dataset**
- 500,000+ transactions from a UK-based gift retailer (2009-2011)
- Source: UCI Machine Learning Repository
- After cleaning: 392,693 rows, 4,338 unique customers

---

## Methodology

### 1. Data Cleaning
Removed missing Customer IDs, cancelled transactions (Invoice starting with C), negative quantities, zero prices, and duplicates. Engineered `TotalPrice = Quantity × Price`.

### 2. Exploratory Data Analysis
Identified strong Q4 seasonality, UK market dominance (90%+ revenue), and Pareto distribution in customer spending — top 5% of customers generate disproportionate revenue.

### 3. RFM Segmentation
Engineered Recency, Frequency, and Monetary features per customer. Applied K-Means clustering (K=4 selected via Elbow Method) to produce four actionable segments:

| Segment | Customers | Avg Monetary | Churn Rate |
|---|---|---|---|
| Champions | 3 | £207,507 | 0% |
| Loyal | 52 | £31,132 | 1.9% |
| At Risk | 2,876 | £1,573 | 15.9% |
| Lost | 989 | £525 | 100% |

### 4. Churn Prediction
XGBoost Classifier wrapped in a sklearn Pipeline (StandardScaler + XGBClassifier). Handled class imbalance via `scale_pos_weight`. Achieved ROC-AUC of 0.71 and churn recall of 0.69.

### 5. CLV Prediction
XGBoost Regressor wrapped in a sklearn Pipeline. Applied log transformation to handle right-skewed CLV distribution. Trained exclusively on active customers (CLV > 0) to separate the buy/no-buy decision from the spend prediction. Achieved R² of 0.23.

### 6. Explainable AI
SHAP TreeExplainer applied to both models. Key finding: Monetary value is the dominant predictor in both churn and CLV models — high-spending customers are less likely to churn and more likely to generate future revenue.

### 7. SQL Analytics
SQLite database created from processed CSVs. Six business queries covering segment profiling, churn rates, CLV by segment, monthly revenue trends, and top product performance.

### 8. Streamlit Dashboard
Five-page interactive dashboard with real-time predictions, SHAP visualizations, and customer-level drill-down.

---

## Key Results

- Identified 55 high-value customers (Champions + Loyal) generating avg £31k-£207k revenue
- Churn model correctly identifies 69% of churning customers for targeted retention
- SHAP analysis revealed Monetary as the dominant predictor across both models
- SQL analysis showed At Risk segment has only 15.9% actual churn rate — high retention opportunity

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/Ruana-Ahuja/customer-lifecycle-intelligence.git
cd customer-lifecycle-intelligence

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add dataset to data/raw/online_retail_II.xlsx
# Download from: https://archive.ics.uci.edu/dataset/502/online+retail+ii

# Run full pipeline
python main.py

# Launch dashboard
streamlit run dashboard/app.py
```
