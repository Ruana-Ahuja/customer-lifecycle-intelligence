import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import joblib

sns.set_style('whitegrid')

FEATURES = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'Tenure', 'AvgDaysBetweenOrders']


def train_churn_model(features_df):
    X = features_df[FEATURES]
    y = features_df['Churned']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_churn_model(model, X_test, y_test, output_path):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nChurn Model Performance:")
    print(classification_report(y_test, y_pred, target_names=['Active', 'Churned']))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Active', 'Churned'],
                yticklabels=['Active', 'Churned'])
    plt.title('Churn Model Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return y_pred, y_prob


def save_churn_model(model, path):
    joblib.dump(model, path)