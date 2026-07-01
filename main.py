from src.data_cleaning import load_data, clean_data, save_cleaned_data
from src.eda import (
    revenue_trend,
    revenue_by_country,
    customer_spending_distribution,
    purchase_frequency_distribution,
    top_products
)

RAW_DATA_PATH = 'data/raw/online_retail_II.xlsx'
CLEANED_DATA_PATH = 'data/processed/cleaned_data.csv'
FIGURES_PATH = 'reports/figures'


def main():
    df_raw = load_data(RAW_DATA_PATH)
    df_clean = clean_data(df_raw)
    save_cleaned_data(df_clean, CLEANED_DATA_PATH)

    print(f"Cleaned dataset: {df_clean.shape[0]} rows, {df_clean['Customer ID'].nunique()} unique customers")

    revenue_trend(df_clean, f'{FIGURES_PATH}/revenue_trend.png')
    revenue_by_country(df_clean, f'{FIGURES_PATH}/revenue_by_country.png')
    customer_spending_distribution(df_clean, f'{FIGURES_PATH}/customer_spending_distribution.png')
    purchase_frequency_distribution(df_clean, f'{FIGURES_PATH}/purchase_frequency_distribution.png')
    top_products(df_clean, f'{FIGURES_PATH}/top_products.png')

    print("EDA figures saved to reports/figures/")


if __name__ == "__main__":
    main()