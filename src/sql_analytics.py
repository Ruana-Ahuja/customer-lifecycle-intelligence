import sqlite3
import pandas as pd
import os


DB_PATH = 'data/processed/retail_analytics.db'


def create_database(cleaned_path, rfm_path, churn_path, clv_path):
    conn = sqlite3.connect(DB_PATH)

    pd.read_csv(cleaned_path).to_sql('transactions', conn, if_exists='replace', index=False)
    pd.read_csv(rfm_path).to_sql('rfm_segments', conn, if_exists='replace', index=False)
    pd.read_csv(churn_path).to_sql('churn_features', conn, if_exists='replace', index=False)
    pd.read_csv(clv_path).to_sql('clv_features', conn, if_exists='replace', index=False)

    conn.close()
    print("Database created with 4 tables: transactions, rfm_segments, churn_features, clv_features")


def run_queries():
    conn = sqlite3.connect(DB_PATH)

    queries = {
        "Customer Segment Summary": """
            SELECT
                Segment,
                COUNT(*) as CustomerCount,
                ROUND(AVG(Recency), 1) as AvgRecency,
                ROUND(AVG(Frequency), 1) as AvgFrequency,
                ROUND(AVG(Monetary), 2) as AvgMonetary
            FROM rfm_segments
            GROUP BY Segment
            ORDER BY AvgMonetary DESC
        """,

        "Top 10 Customers by Revenue": """
            SELECT
                "Customer ID",
                ROUND(SUM(TotalPrice), 2) as TotalRevenue,
                COUNT(DISTINCT Invoice) as TotalOrders
            FROM transactions
            GROUP BY "Customer ID"
            ORDER BY TotalRevenue DESC
            LIMIT 10
        """,

        "Churn Rate by Segment": """
            SELECT
                r.Segment,
                COUNT(*) as TotalCustomers,
                SUM(c.Churned) as ChurnedCustomers,
                ROUND(100.0 * SUM(c.Churned) / COUNT(*), 1) as ChurnRate
            FROM rfm_segments r
            JOIN churn_features c ON r."Customer ID" = c."Customer ID"
            GROUP BY r.Segment
            ORDER BY ChurnRate DESC
        """,

        "Monthly Revenue": """
            SELECT
                STRFTIME('%Y-%m', InvoiceDate) as Month,
                ROUND(SUM(TotalPrice), 2) as Revenue,
                COUNT(DISTINCT "Customer ID") as ActiveCustomers
            FROM transactions
            GROUP BY Month
            ORDER BY Month
        """,

        "Average CLV by Segment": """
            SELECT
                r.Segment,
                ROUND(AVG(cl.CLV), 2) as AvgCLV,
                ROUND(MAX(cl.CLV), 2) as MaxCLV,
                COUNT(*) as CustomerCount
            FROM rfm_segments r
            JOIN clv_features cl ON r."Customer ID" = cl."Customer ID"
            GROUP BY r.Segment
            ORDER BY AvgCLV DESC
        """,

        "Top 10 Products by Revenue": """
            SELECT
                Description,
                ROUND(SUM(TotalPrice), 2) as TotalRevenue,
                SUM(Quantity) as UnitsSold
            FROM transactions
            WHERE Description IS NOT NULL
            GROUP BY Description
            ORDER BY TotalRevenue DESC
            LIMIT 10
        """
    }

    results = {}
    for name, query in queries.items():
        print(f"\n{'='*50}")
        print(f"{name}")
        print('='*50)
        df = pd.read_sql_query(query, conn)
        print(df.to_string(index=False))
        results[name] = df

    conn.close()
    return results