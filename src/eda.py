import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


def revenue_trend(df, output_path):
    monthly_revenue = df.set_index('InvoiceDate').resample('ME')['TotalPrice'].sum()

    plt.figure(figsize=(10, 5))
    monthly_revenue.plot(kind='line', marker='o')
    plt.title('Monthly Revenue Trend')
    plt.xlabel('Month')
    plt.ylabel('Revenue (GBP)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def revenue_by_country(df, output_path, top_n=10):
    country_revenue = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=country_revenue.values, y=country_revenue.index)
    plt.title(f'Top {top_n} Countries by Revenue')
    plt.xlabel('Revenue (GBP)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def customer_spending_distribution(df, output_path):
    customer_revenue = df.groupby('Customer ID')['TotalPrice'].sum()

    plt.figure(figsize=(10, 5))
    sns.histplot(customer_revenue, bins=50)
    plt.title('Customer Spending Distribution')
    plt.xlabel('Total Spend (GBP)')
    plt.xlim(0, customer_revenue.quantile(0.95))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def purchase_frequency_distribution(df, output_path):
    purchase_freq = df.groupby('Customer ID')['Invoice'].nunique()

    plt.figure(figsize=(10, 5))
    sns.histplot(purchase_freq, bins=50)
    plt.title('Customer Purchase Frequency Distribution')
    plt.xlabel('Number of Purchases')
    plt.xlim(0, purchase_freq.quantile(0.95))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def top_products(df, output_path, top_n=10):
    product_sales = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=product_sales.values, y=product_sales.index)
    plt.title(f'Top {top_n} Best-Selling Products')
    plt.xlabel('Units Sold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()