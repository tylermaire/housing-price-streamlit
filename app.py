import streamlit as st
import pandas as pd
import numpy as np
import requests, io, joblib

st.set_page_config(page_title="🏠 Metro Price Predictor", layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("metro.tsv.gz", sep="\t", compression="gzip", low_memory=False)
    except FileNotFoundError:
        url = (
            "https://redfin-public-data.s3.us-west-2.amazonaws.com/"
            "redfin_market_tracker/redfin_metro_market_tracker.tsv000.gz"
        )
        df = pd.read_csv(url, sep="\t", compression="gzip", low_memory=False)
    # normalize names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    # filter metros
    df = df[df["region_type"] == "metro"]
    df["period_begin"] = pd.to_datetime(df["period_begin"], errors="coerce")
    df["median_sale_price"] = pd.to_numeric(df["median_sale_price"], errors="coerce")
    df.dropna(subset=["region", "period_begin", "median_sale_price"], inplace=True)
    return df

df = load_data()

# Sidebar search + select
st.sidebar.title("🏙️ Metro Selection")
search = st.sidebar.text_input("Search metros", "")
metros = sorted(df["region"].unique())
options = [m for m in metros if search.lower() in m.lower()] if search else metros
selected = st.sidebar.selectbox("Choose a metro", options)

# Filter and plot history
metro_df = df[df["region"] == selected].sort_values("period_begin")
st.title(f"📊 Median Sale Price — {selected}")
st.line_chart(metro_df.set_index("period_begin")["median_sale_price"])

# Feature engineering
metro_df["rolling_avg"] = metro_df["median_sale_price"].rolling(3).mean()
metro_df["yoy_pct"]   = metro_df["median_sale_price"].pct_change(12) * 100
metro_df["lag_1"]     = metro_df["median_sale_price"].shift(1)

features = metro_df.dropna(subset=["rolling_avg", "yoy_pct", "lag_1"])
if features.empty:
    st.warning("🚫 Not enough data to compute prediction features.")
    st.stop()

latest = features.iloc[-1]

@st.cache_resource
def load_model(region_name):
    # build filename like "Atlanta_GA_metro_area.pkl"
    base = region_name.replace(", ", "_").replace(" ", "_")
    fname = f"{base}_metro_area.pkl"
    url = (
        "https://raw.githubusercontent.com/tylermaire/"
        "housing-price-streamlit/main/metro_models/"
        + fname
    )
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None
    return joblib.load(io.BytesIO(r.content))

model = load_model(selected)
if model is None:
    st.error(f"🚫 No model found for {selected}.")
    st.stop()

# Prediction inputs
st.subheader("🔮 Predict Next Month’s Median Price")
f1 = st.number_input("Latest Median Price", value=float(latest["median_sale_price"]))
f2 = st.number_input("3-Month Rolling Avg", value=float(latest["rolling_avg"]))
f3 = st.number_input("Year-over-Year % Change", value=float(latest["yoy_pct"]))
f4 = st.number_input("Previous Month's Price", value=float(latest["lag_1"]))

X = np.array([[f1, f2, f3, f4]])
try:
    prediction = model.predict(X)[0]
    st.metric("💰 Predicted Next Month Price", f"${prediction:,.0f}")
except Exception as e:
    st.error(f"❌ Prediction failed: {e}")
