# Data-Prediction-Household-Energy-Usage
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Streamlit App Title
st.title("⚡ Power Consumption Prediction")

# Upload CSV File
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Display First Few Rows
    st.subheader("🔍 Preview of Uploaded Data")
    st.dataframe(df.head())

    # Select Target Variable
    target = st.selectbox("🎯 Select Target Variable", df.columns)

    # Features (X) and Target (y)
    X = df.drop(columns=[target, "Datetime"], errors="ignore")  # Features
    y = df[target]  # Target

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Compute Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display Metrics
    st.subheader("📊 Model Evaluation Metrics")
    st.write(f"✅ **Root Mean Squared Error (RMSE):** {rmse:.4f}")
    st.write(f"✅ **Mean Absolute Error (MAE):** {mae:.4f}")
    st.write(f"✅ **R-Squared (R²):** {r2:.4f}")

    # Feature Importance
    feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False).head(10)  # 🔥 Top 10 Features

    # Feature Importance Visualization
    st.subheader("🔍 Top 10 Feature Importance Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="viridis", ax=ax)
    ax.set_title("Top 10 Important Features")
    st.pyplot(fig)

    # Actual vs Predicted Plot
    st.subheader("📈 Actual vs Predicted Values")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax2)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")  # 45-degree line
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.set_title("Actual vs Predicted Power Consumption")
    st.pyplot(fig2)

    st.success("✅ Model training and prediction completed!")
else:
    st.warning("📤 Please upload a CSV file to start.")

