import streamlit as st
import pandas as pd
import numpy as np
import requests, io, joblib

st.set_page_config(page_title="🏠 Metro Price Predictor", layout="wide")

# ----------------------
# 1. Load and cache the Redfin metro data
# ----------------------
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
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    # Keep only metro-level records
    df = df[df["region_type"] == "metro"]
    # Parse dates and prices
    df["period_begin"] = pd.to_datetime(df["period_begin"], errors="coerce")
    df["median_sale_price"] = pd.to_numeric(df["median_sale_price"], errors="coerce")
    # Drop rows missing critical fields
    df = df.dropna(subset=["region", "period_begin", "median_sale_price"])
    return df

df = load_data()

# ----------------------
# 2. Sidebar: search & select metro
# ----------------------
st.sidebar.title("🏙️ Metro Selector")
search = st.sidebar.text_input("🔍 Search metros", "")
all_metros = sorted(df["region"].unique())
options = [m for m in all_metros if search.lower() in m.lower()] if search else all_metros
selected = st.sidebar.selectbox("Choose a metro", options)

# ----------------------
# 3. Plot historical median price
# ----------------------
sub = df[df["region"] == selected].sort_values("period_begin")
st.title(f"📊 Historical Median Sale Price — {selected}")
st.line_chart(sub.set_index("period_begin")["median_sale_price"])

# ----------------------
# 4. Feature engineering for prediction
# ----------------------
sub["rolling_avg"] = sub["median_sale_price"].rolling(3).mean()
sub["yoy_pct"]   = sub["median_sale_price"].pct_change(12) * 100
sub["lag_1"]     = sub["median_sale_price"].shift(1)

# Keep only rows where all features are present
feature_cols = ["median_sale_price", "rolling_avg", "yoy_pct", "lag_1"]
features_df = sub.dropna(subset=feature_cols)
if features_df.empty:
    st.warning("🚫 Not enough complete data to compute prediction features.")
    st.stop()
latest = features_df.iloc[-1]

# ----------------------
# 5. Load the metro-specific model from GitHub
# ----------------------
@st.cache_resource
def load_model(region_name: str):
    # Remove trailing " metro area" if present
    name = region_name
    if name.lower().endswith(" metro area"):
        name = name[: -len(" metro area")]
    # Create safe filename
    safe = name.replace(",","").replace(" ", "_")
    fname = f"{safe}_metro_area.pkl"
    url = (
        "https://raw.githubusercontent.com/tylermaire/"
        "housing-price-streamlit/main/metro_models/"
        + fname
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return joblib.load(io.BytesIO(r.content))
    except Exception:
        return None

model = load_model(selected)
if model is None:
    st.error(f"🚫 No model found for {selected}.")
    st.stop()

# ----------------------
# 6. Prediction inputs & run
# ----------------------
st.subheader("🔮 Predict Next Month’s Median Price")
f1 = st.number_input("Latest Median Price",        value=float(latest["median_sale_price"]))
f2 = st.number_input("3-Month Rolling Average",    value=float(latest["rolling_avg"]))
f3 = st.number_input("Year-over-Year Change (%)",  value=float(latest["yoy_pct"]))
f4 = st.number_input("Previous Month’s Median Price", value=float(latest["lag_1"]))

X = np.array([[f1, f2, f3, f4]])
try:
    pred = model.predict(X)[0]
    st.metric("💰 Predicted Median Price Next Month", f"${pred:,.0f}")
except Exception as e:
    st.error(f"❌ Prediction error: {e}")
