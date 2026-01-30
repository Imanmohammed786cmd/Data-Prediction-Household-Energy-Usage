# =====================================
# IMPORTS
# =====================================
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="⚡ Energy Consumption Prediction",
    layout="wide"
)
st.title("⚡ Household Energy Consumption Prediction")

# =====================================
# CSV PATH
# =====================================
CSV_PATH = "/Users/imanmohammed/Downloads/Mini3/cleaned_output.csv"

if not os.path.exists(CSV_PATH):
    st.error("❌ CSV file not found")
    st.stop()

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(CSV_PATH)
st.success(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# =====================================
# FILTER SECTION
# =====================================
st.sidebar.subheader("🔍 Filters")
filter_start = st.sidebar.number_input(
    "Start Index", min_value=0, max_value=len(df)-1, value=0
)
filter_end = st.sidebar.number_input(
    "End Index", min_value=0, max_value=len(df)-1, value=min(2000, len(df)-1)
)

df_filtered = df.iloc[filter_start:filter_end].copy()

st.write(f"Showing rows from {filter_start} to {filter_end}")
st.dataframe(df_filtered.head(1000), use_container_width=True)

# =====================================
# FEATURE ENGINEERING
# =====================================
@st.cache_data
def create_features(df):
    df = df.copy()
    df["power_lag_1"] = df["Global_active_power"].shift(1)
    df["power_lag_2"] = df["Global_active_power"].shift(2)
    df["intensity_lag_1"] = df["Global_intensity"].shift(1)
    df = df.dropna()
    return df

df_features = create_features(df_filtered)

FEATURES = ["power_lag_1", "power_lag_2", "intensity_lag_1"]
TARGET = "Global_active_power"

X = df_features[FEATURES]
y = df_features[TARGET]

# =====================================
# TIME SERIES SPLIT
# =====================================
split_index = int(len(df_features) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

st.info(f"📁 Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# =====================================
# MODEL TRAINING
# =====================================
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

with st.spinner("🔄 Training model..."):
    model = train_model(X_train, y_train)

st.success("✅ Model trained successfully")

# =====================================
# PREDICTIONS
# =====================================
y_pred = model.predict(X_test)

y_test_arr = np.asarray(y_test, dtype=float).reshape(-1)
y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)

min_len = min(len(y_test_arr), len(y_pred_arr))
y_test_arr = y_test_arr[:min_len]
y_pred_arr = y_pred_arr[:min_len]

# =====================================
# METRICS
# =====================================
rmse = np.sqrt(mean_squared_error(y_test_arr, y_pred_arr))
r2 = r2_score(y_test_arr, y_pred_arr)

col1, col2 = st.columns(2)
col1.metric("📉 RMSE", f"{rmse:.4f}")
col2.metric("📊 R² Score", f"{r2:.4f}")

# =====================================
# VISUALIZATIONS
# =====================================

# 1️⃣ Actual vs Predicted
st.subheader("📈 Actual vs Predicted Energy Consumption")

plot_len = min(500, min_len)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_test_arr[:plot_len], label="Actual")
ax.plot(y_pred_arr[:plot_len], label="Predicted")
ax.set_xlabel("Time Index")
ax.set_ylabel("Global Active Power")
ax.legend()
st.pyplot(fig)

# 2️⃣ Prediction Error Over Time
st.subheader("📉 Prediction Error Over Time")

errors = y_test_arr - y_pred_arr
errors = errors[np.isfinite(errors)]  # remove NaN / inf

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(errors[:plot_len])
ax.axhline(0, linestyle="--", color="red")
ax.set_xlabel("Time Index")
ax.set_ylabel("Prediction Error")
st.pyplot(fig)

# 3️⃣ Actual vs Predicted Scatter
st.subheader("🔍 Actual vs Predicted Scatter")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test_arr, y_pred_arr, alpha=0.5)
ax.plot(
    [y_test_arr.min(), y_test_arr.max()],
    [y_test_arr.min(), y_test_arr.max()],
    linestyle="--",
    color="red"
)
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
st.pyplot(fig)

# 4️⃣ Feature Importance
st.subheader("🔥 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(importance_df["Feature"], importance_df["Importance"])
ax.set_xlabel("Importance Score")
st.pyplot(fig)

# 5️⃣ Energy Consumption Trend
st.subheader("⚡ Energy Consumption Trend")

trend_len = min(2000, len(df_filtered))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_filtered["Global_active_power"].iloc[:trend_len])
ax.set_xlabel("Time Index")
ax.set_ylabel("Global Active Power")
st.pyplot(fig)
