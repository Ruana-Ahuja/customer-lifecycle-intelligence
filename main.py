from src.data_cleaning import load_data, clean_data, save_cleaned_data
from src.eda import (
    revenue_trend,
    revenue_by_country,
    customer_spending_distribution,
    purchase_frequency_distribution,
    top_products
)
from src.feature_engineering import build_rfm, build_churn_features, build_clv_features
from src.segmentation import run_elbow_method, assign_clusters
from src.churn_model import train_churn_model, evaluate_churn_model, save_churn_model
from src.clv_model import train_clv_model, evaluate_clv_model, save_clv_model
from src.shap_explainer import explain_churn_model, explain_clv_model
from src.sql_analytics import create_database, run_queries

RAW_DATA_PATH = 'data/raw/online_retail_II.xlsx'
CLEANED_DATA_PATH = 'data/processed/cleaned_data.csv'
RFM_PATH = 'data/processed/rfm_features.csv'
CHURN_FEATURES_PATH = 'data/processed/churn_features.csv'
CLV_FEATURES_PATH = 'data/processed/clv_features.csv'
CHURN_MODEL_PATH = 'models/churn_model.pkl'
CLV_MODEL_PATH = 'models/clv_model.pkl'
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
    rfm, km = assign_clusters(rfm, rfm_scaled, n_clusters=4,
                              output_path=f'{FIGURES_PATH}/segment_profiles.png')
    rfm.to_csv(RFM_PATH, index=False)
    print(f"Segmentation complete:\n{rfm['Segment'].value_counts().to_string()}")

    churn_features = build_churn_features(df_clean)
    churn_features.to_csv(CHURN_FEATURES_PATH, index=False)
    churn_model, X_test_churn, y_test_churn = train_churn_model(churn_features)
    evaluate_churn_model(churn_model, X_test_churn, y_test_churn,
                         f'{FIGURES_PATH}/churn_confusion_matrix.png')
    save_churn_model(churn_model, CHURN_MODEL_PATH)
    print("Churn model saved.")

    clv_features = build_clv_features(df_clean)
    clv_features.to_csv(CLV_FEATURES_PATH, index=False)
    clv_model, X_test_clv, y_test_clv = train_clv_model(clv_features)
    evaluate_clv_model(clv_model, X_test_clv, y_test_clv,
                       f'{FIGURES_PATH}/clv_actual_vs_predicted.png')
    save_clv_model(clv_model, CLV_MODEL_PATH)
    print("CLV model saved.")

    explain_churn_model(churn_model, X_test_churn, FIGURES_PATH)
    explain_clv_model(clv_model, X_test_clv, FIGURES_PATH)
    print("SHAP explanations saved.")

    create_database(CLEANED_DATA_PATH, RFM_PATH, CHURN_FEATURES_PATH, CLV_FEATURES_PATH)
    run_queries()
    print("\nSQL analytics complete.")


if __name__ == "__main__":
    main()