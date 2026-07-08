import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Customer Lifecycle Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

# ── Data Loading ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/cleaned_data.csv', parse_dates=['InvoiceDate'])
    rfm = pd.read_csv('data/processed/rfm_features.csv')
    churn = pd.read_csv('data/processed/churn_features.csv')
    clv = pd.read_csv('data/processed/clv_features.csv')
    return df, rfm, churn, clv


@st.cache_resource
def load_models():
    churn_model = joblib.load('models/churn_model.pkl')
    clv_model = joblib.load('models/clv_model.pkl')
    return churn_model, clv_model


df, rfm, churn_df, clv_df = load_data()
churn_model, clv_model = load_models()

FEATURES = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'Tenure', 'AvgDaysBetweenOrders']

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview",
    "Customer Segments",
    "Churn Analysis",
    "CLV Analysis",
    "SHAP Explainability"
])

# ── Page 1: Overview ──────────────────────────────────────────
if page == "Overview":
    st.title("Customer Lifecycle Intelligence Platform")
    st.markdown("End-to-end customer analytics: segmentation, churn prediction, and lifetime value forecasting.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{rfm['Customer ID'].nunique():,}")
    col2.metric("Total Revenue", f"£{df['TotalPrice'].sum():,.0f}")
    col3.metric("Avg Order Value", f"£{df['TotalPrice'].mean():,.2f}")
    col4.metric("Churn Rate", f"{churn_df['Churned'].mean()*100:.1f}%")

    st.subheader("Monthly Revenue Trend")
    monthly = df.set_index('InvoiceDate').resample('ME')['TotalPrice'].sum().reset_index()
    monthly.columns = ['Month', 'Revenue']
    fig = px.line(monthly, x='Month', y='Revenue', markers=True)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Revenue by Country")
        country_rev = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(country_rev, x='TotalPrice', y='Country', orientation='h')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 Products")
        top_prod = df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(top_prod, x='TotalPrice', y='Description', orientation='h')
        st.plotly_chart(fig, use_container_width=True)

# ── Page 2: Customer Segments ─────────────────────────────────
elif page == "Customer Segments":
    st.title("Customer Segmentation")
    st.markdown("RFM-based K-Means clustering into 4 behavioral segments.")

    col1, col2 = st.columns(2)
    with col1:
        seg_counts = rfm['Segment'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']
        fig = px.pie(seg_counts, names='Segment', values='Count', title='Customer Distribution by Segment')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_summary = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(2).reset_index()
        st.subheader("Segment Profiles")
        st.dataframe(seg_summary, use_container_width=True)

    st.subheader("RFM Distribution by Segment")
    metric = st.selectbox("Select Metric", ['Recency', 'Frequency', 'Monetary'])
    fig = px.box(rfm, x='Segment', y=metric, color='Segment')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("3D RFM Scatter Plot")
    fig = px.scatter_3d(
        rfm, x='Recency', y='Frequency', z='Monetary',
        color='Segment', opacity=0.6,
        title='Customer Segments in RFM Space'
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Page 3: Churn Analysis ────────────────────────────────────
elif page == "Churn Analysis":
    st.title("Churn Prediction")
    st.markdown("XGBoost classifier predicting customer churn with sklearn Pipeline.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers Analyzed", len(churn_df))
    col2.metric("Churned Customers", int(churn_df['Churned'].sum()))
    col3.metric("Churn Rate", f"{churn_df['Churned'].mean()*100:.1f}%")

    churn_predictions = churn_model.predict(churn_df[FEATURES])
    churn_proba = churn_model.predict_proba(churn_df[FEATURES])[:, 1]

    churn_df = churn_df.copy()
    churn_df['ChurnProbability'] = churn_proba
    churn_df['ChurnPrediction'] = churn_predictions

    st.subheader("Churn Probability Distribution")
    fig = px.histogram(churn_df, x='ChurnProbability', nbins=50,
                       color='Churned', barmode='overlay')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("High Risk Customers")
    high_risk = churn_df[churn_df['ChurnProbability'] > 0.7].sort_values(
        'ChurnProbability', ascending=False
    )[['Customer ID', 'Recency', 'Frequency', 'Monetary', 'ChurnProbability']].head(20)
    st.dataframe(high_risk.round(3), use_container_width=True)

# ── Page 4: CLV Analysis ──────────────────────────────────────
elif page == "CLV Analysis":
    st.title("Customer Lifetime Value Prediction")
    st.markdown("XGBoost regression predicting 90-day future revenue per customer.")

    active_clv = clv_df[clv_df['CLV'] > 0].copy()
    clv_cap = active_clv['CLV'].quantile(0.95)
    active_clv = active_clv[active_clv['CLV'] <= clv_cap]

    clv_pred = np.expm1(clv_model.predict(active_clv[FEATURES]))
    active_clv = active_clv.copy()
    active_clv['PredictedCLV'] = clv_pred

    col1, col2, col3 = st.columns(3)
    col1.metric("Active Customers", len(active_clv))
    col2.metric("Avg Predicted CLV", f"£{active_clv['PredictedCLV'].mean():,.2f}")
    col3.metric("Total Predicted Revenue", f"£{active_clv['PredictedCLV'].sum():,.0f}")

    st.subheader("Actual vs Predicted CLV")
    fig = px.scatter(active_clv, x='CLV', y='PredictedCLV', opacity=0.5,
                     labels={'CLV': 'Actual CLV', 'PredictedCLV': 'Predicted CLV'})
    fig.add_shape(type='line',
                  x0=active_clv['CLV'].min(), y0=active_clv['CLV'].min(),
                  x1=active_clv['CLV'].max(), y1=active_clv['CLV'].max(),
                  line=dict(color='red', dash='dash'))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("CLV Distribution")
    fig = px.histogram(active_clv, x='PredictedCLV', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

# ── Page 5: SHAP Explainability ───────────────────────────────
elif page == "SHAP Explainability":
    st.title("Explainable AI — SHAP Analysis")
    st.markdown("SHAP values reveal which features drive each model's predictions.")

    model_choice = st.radio("Select Model", ["Churn Model", "CLV Model"])

    if model_choice == "Churn Model":
        model = churn_model.named_steps['model']
        scaler = churn_model.named_steps['scaler']
        X = churn_df[FEATURES].copy()
    else:
        active_clv = clv_df[clv_df['CLV'] > 0].copy()
        clv_cap = active_clv['CLV'].quantile(0.95)
        active_clv = active_clv[active_clv['CLV'] <= clv_cap]
        model = clv_model.named_steps['model']
        scaler = clv_model.named_steps['scaler']
        X = active_clv[FEATURES].copy()

    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURES)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_scaled_df)

    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(12, 5))
    shap.summary_plot(shap_values, X_scaled_df, plot_type='bar', show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("SHAP Summary Plot")
    fig, ax = plt.subplots(figsize=(12, 6))
    shap.summary_plot(shap_values, X_scaled_df, show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()