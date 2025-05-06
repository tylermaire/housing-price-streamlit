import streamlit as st
import pandas as pd

st.set_page_config(page_title="📈 Metro Price Explorer", layout="wide")

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

# Load & cache data
df = load_data()

# App UI
st.title("📊 Metro Housing Market Explorer")
metros = sorted(df["region"].unique())
selected = st.selectbox("Select a metro area:", metros)

# Filter and plot
sub = df[df["region"] == selected].sort_values("period_begin")
if sub.empty:
    st.error("No data available for this metro.")
else:
    st.line_chart(sub.set_index("period_begin")["median_sale_price"])

## Part 2 ##

import requests

# Construct the expected model filename & URL
safe = selected.replace(",","").replace(" ","_")
model_fname = f"{safe}_metro_area.pkl"
model_url = (
    "https://raw.githubusercontent.com/tylermaire/"
    "housing-price-streamlit/main/metro_models/"
    + model_fname
)

st.write("Looking for model file:", model_fname)
st.write("Raw URL:", model_url)

# Check if it’s reachable
try:
    resp = requests.head(model_url, timeout=5)
    st.write("HTTP status code:", resp.status_code)
    if resp.status_code == 200:
        st.success("Model file is present on GitHub.")
    else:
        st.error("Model file *not* found (status ≠ 200).")
except Exception as e:
    st.error(f"Error checking model URL: {e}")
