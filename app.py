import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import io

st.set_page_config(page_title="🏠 Metro Price Predictor", layout="centered")

@st.cache_data
def load_data():
    """Load Redfin metro-level data."""
    try:
        df = pd.read_csv("metro.tsv.gz", sep="\t", compression="gzip")
    except FileNotFoundError:
        url = "https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/redfin_metro_market_tracker.tsv000.gz"
        df = pd.read_csv(url, sep="\t", compression="gzip")
    df.columns = df.columns.str.strip().str.lower()
    df = df[df["region_type"] == "metro"]
    df["period_begin"] = pd.to_datetime(df["period_begin"], errors="coerce")
    df["median_sale_price"] = pd.to_numeric(df["median_sale_price"], errors="coerce")
    df = df.dropna(subset=["region", "period_begin", "median_sale_price"])
    return df

data = load_data()

st.title("🏙️ Metro Housing Market Explorer")

metros = sorted(data["region"].unique())
selected_metro = st.selectbox("Select a metro area:", metros)

metro_df = data[data["region"] == selected_metro].copy()
metro_df.sort_values("period_begin", inplace=True)

# Show the chart
st.subheader(f"Median Sale Price – {selected_metro}")
st.line_chart(metro_df.set_index("period_begin")["median_sale_price"])

@st.cache_resource
def load_model(metro_name):
    """Download and load model file for selected metro."""
    safe_name = metro_name.replace(", ", "_").replace(" ", "_") + ".pkl"
    model_url = f"https://raw.githubusercontent.com/tylermaire/housing-price-streamlit/main/metro_models/{safe_name}"
    try:
        r = requests.get(model_url, timeout=5)
        r.raise_for_status()
        model = joblib.load(io.BytesIO(r.content))
        return model
    except Exception:
        return None

model = load_model(selected_metro)

# Prepare features
metro_clean = metro_df.dropna(subset=["median_sale_price"])
if metro_clean.empty:
    st.warning("🚫 Not enough valid data available for prediction in this metro.")
    st.stop()

metro_clean["rolling_avg_price"] = metro_clean["median_sale_price"].rolling(3).mean()
metro_clean["yoy_price_change"] = metro_clean["median_sale_price"].pct_change(12) * 100
metro_clean["lag_1"] = metro_clean["median_sale_price"].shift(1)

latest = metro_clean.dropna().iloc[-1]

st.subheader("📈 Prediction Input Features")
f1 = st.number_input("Current Median Price", value=float(latest["median_sale_price"]), step=1000.0)
f2 = st.number_input("3-Month Rolling Avg", value=float(latest["rolling_avg_price"]), step=1000.0)
f3 = st.number_input("Year-over-Year Change (%)", value=float(latest["yoy_price_change"]), step=0.1, format="%.2f")
f4 = st.number_input("Last Month's Price", value=float(latest["lag_1"]), step=1000.0)

if model:
    input_array = np.array([[f1, f2, f3, f4]])
    try:
        pred = model.predict(input_array)[0]
        st.metric(label="🔮 Predicted Next Month Price", value=f"${pred:,.0f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
else:
    st.warning("🚫 No prediction model found for this metro.")
