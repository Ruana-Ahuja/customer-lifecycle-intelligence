import pandas as pd


def build_rfm(df):
    df = df[df['Country'] == 'United Kingdom'].copy()

    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('Customer ID').agg(
        Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency=('Invoice', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index()

    return rfm