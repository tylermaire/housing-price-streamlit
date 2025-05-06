import streamlit as st
import pandas as pd
import numpy as np
import requests, io, joblib

st.set_page_config(page_title="🏠 Metro Price Predictor", layout="wide")

# 1. Load and cache the metro dataset
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
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df[df["region_type"] == "metro"]
    df["period_begin"] = pd.to_datetime(df["period_begin"], errors="coerce")
    df["median_sale_price"] = pd.to_numeric(df["median_sale_price"], errors="coerce")
    df = df.dropna(subset=["region", "period_begin", "median_sale_price"])
    return df

df = load_data()

# 2. Sidebar: search & select
st.sidebar.title("🏙️ Metro Selection")
search = st.sidebar.text_input("Search metros", "")
all_metros = sorted(df["region"].unique())
choices = [m for m in all_metros if search.lower() in m.lower()] if search else all_metros
selected = st.sidebar.selectbox("Choose a metro", choices)

# 3. Filter & plot history
metro_df = df[df["region"] == selected].sort_values("period_begin")
if metro_df.empty:
    st.error("No data for this metro.")
    st.stop()

st.title(f"📊 Median Sale Price — {selected}")
st.line_chart(metro_df.set_index("period_begin")["median_sale_price"])

# 4. Compute features
metro_df["rolling_avg"]    = metro_df["median_sale_price"].rolling(3).mean()
metro_df["yoy_pct"]        = metro_df["median_sale_price"].pct_change(12) * 100
metro_df["lag_1"]          = metro_df["median_sale_price"].shift(1)

# 5. Drop rows missing any feature
feature_cols = ["median_sale_price", "rolling_avg", "yoy_pct", "lag_1"]
features_df = metro_df.dropna(subset=feature_cols)
if features_df.shape[0] < 1:
    st.warning("🚫 Not enough complete data to generate prediction features.")
    st.stop()

latest = features_df.iloc[-1]

# 6. Load and cache the model from GitHub
@st.cache_resource
def load_model(metro_name):
    base = metro_name.replace(", ", "_").replace(" ", "_")
    fname = f"{base}_metro_area.pkl"
    url = (
        "https://raw.githubusercontent.com/tylermaire/"
        "housing-price-streamlit/main/metro_models/"
        + fname
    )
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        return None
    return joblib.load(io.BytesIO(resp.content))

model = load_model(selected)
if model is None:
    st.error(f"🚫 No model found for {selected}.")
    st.stop()

# 7. Prediction inputs
st.subheader("🔮 Predict Next Month’s Median Price")
f1 = st.number_input("Latest Median Price",         value=float(latest["median_sale_price"]))
f2 = st.number_input("3-Month Rolling Average",     value=float(latest["rolling_avg"]))
f3 = st.number_input("Year-over-Year Change (%)",    value=float(latest["yoy_pct"]))
f4 = st.number_input("Previous Month's Median Price", value=float(latest["lag_1"]))

# 8. Predict & display
X = np.array([[f1, f2, f3, f4]])
try:
    pred = model.predict(X)[0]
    st.metric("💰 Predicted Next Month Price", f"${pred:,.0f}")
except Exception as e:
    st.error(f"Prediction failed: {e}")
