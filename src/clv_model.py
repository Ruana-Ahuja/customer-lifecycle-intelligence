import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

sns.set_style('whitegrid')

FEATURES = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'Tenure', 'AvgDaysBetweenOrders']


def train_clv_model(features_df):
    active_df = features_df[features_df['CLV'] > 0].copy()

    clv_cap = active_df['CLV'].quantile(0.95)
    active_df = active_df[active_df['CLV'] <= clv_cap].copy()

    X = active_df[FEATURES]
    y = np.log1p(active_df['CLV'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_clv_model(model, X_test, y_test, output_path):
    y_pred_log = model.predict(X_test)

    y_pred = np.expm1(y_pred_log)
    y_actual = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)

    print(f"\nCLV Model Performance:")
    print(f"RMSE: £{rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual, y_pred, alpha=0.4)
    plt.plot([y_actual.min(), y_actual.max()],
             [y_actual.min(), y_actual.max()], 'r--')
    plt.title('CLV: Actual vs Predicted (Active Customers)')
    plt.xlabel('Actual CLV (GBP)')
    plt.ylabel('Predicted CLV (GBP)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return y_pred


def save_clv_model(model, path):
    joblib.dump(model, path)