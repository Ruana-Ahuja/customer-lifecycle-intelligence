from src.data_cleaning import load_data, clean_data, save_cleaned_data
from src.eda import (
    revenue_trend,
    revenue_by_country,
    customer_spending_distribution,
    purchase_frequency_distribution,
    top_products
)
from src.feature_engineering import build_rfm
from src.segmentation import run_elbow_method, assign_clusters

RAW_DATA_PATH = 'data/raw/online_retail_II.xlsx'
CLEANED_DATA_PATH = 'data/processed/cleaned_data.csv'
RFM_PATH = 'data/processed/rfm_features.csv'
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

    rfm = build_rfm(df_clean)
    rfm_scaled, scaler = run_elbow_method(rfm, output_path=f'{FIGURES_PATH}/elbow_plot.png')
    rfm, km = assign_clusters(rfm, rfm_scaled, n_clusters=4, output_path=f'{FIGURES_PATH}/segment_profiles.png')
    rfm.to_csv(RFM_PATH, index=False)
    print(f"Segmentation complete:\n{rfm['Segment'].value_counts().to_string()}")


if __name__ == "__main__":
    main()