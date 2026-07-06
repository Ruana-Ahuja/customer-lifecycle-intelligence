import pandas as pd
import numpy as np


def build_rfm(df):
    df = df[df['Country'] == 'United Kingdom'].copy()

    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('Customer ID').agg(
        Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency=('Invoice', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index()

    return rfm


def build_churn_features(df):
    df = df[df['Country'] == 'United Kingdom'].copy()

    snapshot_date = df['InvoiceDate'].max()
    cutoff_date = snapshot_date - pd.Timedelta(days=90)

    observation_df = df[df['InvoiceDate'] <= cutoff_date]
    future_df = df[df['InvoiceDate'] > cutoff_date]

    active_customers = set(future_df['Customer ID'].unique())

    features = observation_df.groupby('Customer ID').agg(
        Recency=('InvoiceDate', lambda x: (cutoff_date - x.max()).days),
        Frequency=('Invoice', 'nunique'),
        Monetary=('TotalPrice', 'sum'),
        AvgOrderValue=('TotalPrice', 'mean'),
        Tenure=('InvoiceDate', lambda x: (x.max() - x.min()).days),
    ).reset_index()

    features['AvgDaysBetweenOrders'] = features.apply(
        lambda row: row['Tenure'] / row['Frequency'] if row['Frequency'] > 1 else row['Tenure'],
        axis=1
    )

    features['Churned'] = features['Customer ID'].apply(
        lambda x: 0 if x in active_customers else 1
    )

    return features