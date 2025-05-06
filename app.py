import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
import matplotlib.pyplot as plt

# --------------------------
# Download latest metro data
# --------------------------
@st.cache_data(show_spinner=False)
def download_gzip_file():
    url = "https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/redfin_metro_market_tracker.tsv000.gz"
    r = requests.get(url)
    file_path = "metro.tsv.gz"
    with open(file_path, "wb") as f:
        f.write(r.content)
    return file_path

# --------------------------
# Load and clean data
# --------------------------
@st.cache_data(show_spinner=False)
def load_metro_data():
    file_path = download_gzip_file()
    df = pd.read_csv(file_path, sep='\t', compression='gzip')
    df.columns = df.columns.str.strip().str.upper()
    df = df[df['REGION_TYPE'] == 'metro']
    df['PERIOD_BEGIN'] = pd.to_datetime(df['PERIOD_BEGIN'], errors='coerce')
    df['MEDIAN_SALE_PRICE'] = pd.to_numeric(df['MEDIAN_SALE_PRICE'], errors='coerce')
    df = df.dropna(subset=['REGION', 'PERIOD_BEGIN', 'MEDIAN_SALE_PRICE'])
    return df

# --------------------------
# Safe name formatter
# --------------------------
def format_model_name(name):
    return name.replace(",", "").replace(" ", "_").replace("/", "_") + ".pkl"

# --------------------------
# Load model
# --------------------------
def load_model(metro_name):
    safe_name = format_model_name(metro_name)
    model_url = f"https://raw.githubusercontent.com/tylermaire/housing-price-streamlit/main/metro_models/{safe_name}"
    model_path = f"temp_model.pkl"
    r = requests.get(model_url)
    if r.status_code != 200:
        return None
    with open(model_path, "wb") as f:
        f.write(r.content)
    return joblib.load(model_path)

# --------------------------
# Start Streamlit App
# --------------------------
st.set_page_config(page_title="Housing Price Predictor", layout="wide")
st.title("🏠 Housing Price Predictor - Metro Level")

with st.spinner("📥 Downloading Redfin metro dataset..."):
    df = load_metro_data()
st.success("✅ Redfin data downloaded.")

# Metro Search
all_metros = sorted(df['REGION'].unique())
selected = st.selectbox("🔍 Choose a metro area", all_metros)

# Filter
sub = df[df['REGION'] == selected].copy().sort_values('PERIOD_BEGIN')
sub['ROLLING_AVG'] = sub['MEDIAN_SALE_PRICE'].rolling(3).mean()
sub['YOY_CHANGE'] = sub['MEDIAN_SALE_PRICE'].pct_change(12)
sub['LAG_1'] = sub['MEDIAN_SALE_PRICE'].shift(1)

if sub.dropna().empty:
    st.warning("⚠️ Not enough data to make a prediction for this metro.")
    st.stop()

# Display trend chart
st.subheader(f"📈 Median Sale Price in {selected}")
st.line_chart(sub.set_index('PERIOD_BEGIN')['MEDIAN_SALE_PRICE'])

# Load model from GitHub
model = load_model(selected)
if model is None:
    st.error(f"🚫 Model not found for {selected}. Expected GitHub file: {format_model_name(selected)}")
    st.stop()

# Use latest values for prediction
latest = sub.dropna().iloc[-1]
f1 = float(latest['MEDIAN_SALE_PRICE'])
f2 = float(latest['ROLLING_AVG'])
f3 = float(latest['YOY_CHANGE'])
f4 = float(latest['LAG_1'])
X_pred = np.array([[f1, f2, f3, f4]])
pred = model.predict(X_pred)[0]

# Display prediction
st.subheader("💰 Predicted Median Price Next Month")
st.success(f"${pred:,.0f}")

# Optional: Forecast chart (sample visualization)
st.subheader("📉 Forecast Visualization")
fig, ax = plt.subplots()
ax.plot(sub['PERIOD_BEGIN'], sub['MEDIAN_SALE_PRICE'], label="Actual")
ax.axhline(pred, color='green', linestyle='--', label='Prediction')
ax.set_title("Median Price with Forecast")
ax.legend()
st.pyplot(fig)

# Download option
csv_data = pd.DataFrame({"Next Month Forecast": [pred]})
st.download_button("Download Prediction as CSV", csv_data.to_csv(index=False), file_name="prediction.csv")
