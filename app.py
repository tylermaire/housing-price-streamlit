import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import io

st.set_page_config(page_title="🏠 Metro Price Predictor", layout="wide")

@st.cache_data
def load_data():
    """Load Redfin metro-level data from local or AWS."""
    try:
        df = pd.read_csv("metro.tsv.gz", sep="\t", compression="gzip")
    except FileNotFoundError:
        url = (
            "https://redfin-public-data.s3.us-west-2.amazonaws.com/"
            "redfin_market_tracker/redfin_metro_market_tracker.tsv000.gz"
        )
        df = pd.read_csv(url, sep="\t", compression="gzip")
    df.columns = df.columns.str.strip().str.lower()
    df = df[df["region_type"] == "metro"]
    df["period_begin"] = pd.to_datetime(df["period_begin"], errors="coerce")
    df["median_sale_price"] = pd.to_numeric(df["median_sale_price"], errors="coerce")
    df.dropna(subset=["region", "period_begin", "median_sale_price"], inplace=True)
    return df

df = load_data()

st.title("🏙️ Metro Housing Market Explorer")
st.write("Select a metro to view trends and predict next month's median sale price.")

# Metro selector with search
metros = sorted(df["region"].unique())
search = st.text_input("🔍 Search metro", "")
choices = [m for m in metros if search.lower() in m.lower()] if search else metros
selected = st.selectbox("Choose metro:", choices)

# Filter and plot history
metro_df = df[df["region"] == selected].sort_values("period_begin")
if metro_df.empty:
    st.error("No data for this metro.")
    st.stop()

st.subheader(f"📊 Historical Median Sale Price — {selected}")
st.line_chart(metro_df.set_index("period_begin")["median_sale_price"])

# Compute features
metro_df["rolling_avg"] = metro_df["median_sale_price"].rolling(3).mean()
metro_df["yoy_pct"] = metro_df["median_sale_price"].pct_change(12) * 100
metro_df["lag_1"] = metro_df["median_sale_price"].shift(1)
features_df = metro_df.dropna(subset=["rolling_avg", "yoy_pct", "lag_1"])

if features_df.empty:
    st.warning("Not enough complete data to compute prediction features.")
    st.stop()

latest = features_df.iloc[-1]

@st.cache_resource
def load_model(metro_name):
    """Fetch the .pkl model named <Metro>_metro_area.pkl from GitHub raw."""
    base = metro_name.replace(", ", "_").replace(" ", "_")
    filename = f"{base}_metro_area.pkl"
    url = (
        "https://raw.githubusercontent.com/"
        "tylermaire/housing-price-streamlit/"
        "main/metro_models/"
        + filename
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return joblib.load(io.BytesIO(r.content))
    except Exception:
        return None

model = load_model(selected)

st.subheader("🔮 Predict Next Month’s Median Price")
if model is None:
    st.error(f"No model found for {selected}.")
    st.stop()

# Inputs with defaults
f1 = st.number_input("Current Median Price", value=float(latest["median_sale_price"]))
f2 = st.number_input("3-Month Rolling Avg", value=float(latest["rolling_avg"]))
f3 = st.number_input("YoY % Change", value=float(latest["yoy_pct"]))
f4 = st.number_input("Last Month’s Price", value=float(latest["lag_1"]))

X = np.array([[f1, f2, f3, f4]])
try:
    pred = model.predict(X)[0]
    st.metric("💰 Predicted Median Price Next Month", f"${pred:,.0f}")
except Exception as e:
    st.error(f"Prediction error: {e}")
