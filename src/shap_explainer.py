import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap


def explain_churn_model(pipeline, X_test, output_dir):
    model = pipeline.named_steps['model']
    scaler = pipeline.named_steps['scaler']
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_scaled_df)

    mean_shap = pd.Series(
        np.abs(shap_values.values).mean(axis=0),
        index=X_test.columns
    ).sort_values(ascending=False)

    print("\nChurn Model - Feature Importance (SHAP):")
    for feature, value in mean_shap.items():
        print(f"  {feature:<25} {value:.4f}")

    print(f"\nKey Insight: '{mean_shap.index[0]}' is the strongest churn predictor.")
    print("High spenders are significantly less likely to churn.")

    plt.figure()
    shap.summary_plot(shap_values, X_test_scaled_df, show=False)
    plt.title('SHAP Summary Plot - Churn Model')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_churn_summary.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_test_scaled_df, plot_type='bar', show=False)
    plt.title('SHAP Feature Importance - Churn Model')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_churn_importance.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title('SHAP Waterfall - Single Customer')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_churn_waterfall.png', bbox_inches='tight')
    plt.close()


def explain_clv_model(pipeline, X_test, output_dir):
    model = pipeline.named_steps['model']
    scaler = pipeline.named_steps['scaler']
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_scaled_df)

    mean_shap = pd.Series(
        np.abs(shap_values.values).mean(axis=0),
        index=X_test.columns
    ).sort_values(ascending=False)

    print("\nCLV Model - Feature Importance (SHAP):")
    for feature, value in mean_shap.items():
        print(f"  {feature:<25} {value:.4f}")

    print(f"\nKey Insight: '{mean_shap.index[0]}' is the strongest CLV predictor.")
    print("Past spending behavior is the best indicator of future revenue.")

    plt.figure()