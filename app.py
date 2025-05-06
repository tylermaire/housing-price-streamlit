import streamlit as st
import pandas as pd
import numpy as np
import os, zipfile, io
import requests, joblib
import gdown  # ensure gdown is in requirements

st.set_page_config(page_title="🏠 Metro Price Predictor", layout="wide")

# -----------------------
# 0) Ensure metro_models directory exists with real .pkl files
# -----------------------
DRIVE_ID = "1gVGV1XUzwoy1xlA1CcHCgAJgioGC-RHz"
ZIP_URL  = f"https://drive.google.com/uc?export=download&id={DRIVE_ID}"
ZIP_PATH = "metro_models.zip"
MODELS_DIR = "metro_models"

def ensure_models():
    if not os.path.isdir(MODELS_DIR) or len(os.listdir(MODELS_DIR)) == 0:
        st.info("🚚 Downloading model archive from Google Drive...")
        # Download via gdown
        gdown.download(ZIP_URL, ZIP_PATH, quiet=False)
        # Unzip
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(MODELS_DIR)
        os.remove(ZIP_PATH)
        st.success("✅ Models ready.")

ensure_models()

# -----------------------
# 1) Load & cache Redfin metro data
# -----------------------
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
    # Standardize and filter
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df[df["region_type"] == "metro"]
    df["period_begin"] = pd.to_datetime(df["period_begin"], errors="coerce")
    df["median_sale_price"] = pd.to_numeric(df["median_sale_price"], errors="coerce")
    df = df.dropna(subset=["region", "period_begin", "median_sale_price"])
    return df

df = load_data()

# -----------------------
# 2) Sidebar: search & select metro
# -----------------------
st.sidebar.title("🏙️ Metro Selection")
search = st.sidebar.text_input("Search metros", "")
all_metros = sorted(df["region"].unique())
choices = [m for m in all_metros if search.lower() in m.lower()] if search else all_metros
selected = st.sidebar.selectbox("Choose a metro", choices)

# -----------------------
# 3) Plot historical median price
# -----------------------
sub = df[df["region"] == selected].sort_values("period_begin")
st.title(f"📊 Median Sale Price — {selected}")
st.line_chart(sub.set_index("period_begin")["median_sale_price"])

# -----------------------
# 4) Compute prediction features
# -----------------------
sub["rolling_avg"] = sub["median_sale_price"].rolling(3).mean()
sub["yoy_pct"]   = sub["median_sale_price"].pct_change(12) * 100
sub["lag_1"]     = sub["median_sale_price"].shift(1)

feature_cols = ["median_sale_price", "rolling_avg", "yoy_pct", "lag_1"]
features_df = sub.dropna(subset=feature_cols)
if features_df.empty:
    st.warning("🚫 Not enough complete data for prediction features.")
    st.stop()
latest = features_df.iloc[-1]

# -----------------------
# 5) Load local model from metro_models folder
# -----------------------
def load_local_model(region_name):
    # strip trailing ' metro area'
    name = region_name
    if name.lower().endswith(" metro area"):
        name = name[: -len(" metro area")]
    safe = name.replace(",", "").replace(" ", "_").replace("/", "_")
    fname = f"{safe}_metro_area.pkl"
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

model = load_local_model(selected)
if model is None:
    st.error(f"🚫 No local model found for {selected}. Check metro_models folder.")
    st.stop()

# -----------------------
# 6) Prediction inputs & display
# -----------------------
st.subheader("🔮 Predict Next Month’s Median Price")
f1 = st.number_input("Latest Median Price",        value=float(latest["median_sale_price"]))
f2 = st.number_input("3-Month Rolling Average",    value=float(latest["rolling_avg"]))
f3 = st.number_input("Year-over-Year % Change",    value=float(latest["yoy_pct"]))
f4 = st.number_input("Previous Month’s Price",     value=float(latest["lag_1"]))

X = np.array([[f1, f2, f3, f4]])
try:
    pred = model.predict(X)[0]
    st.metric("💰 Predicted Price Next Month", f"${pred:,.0f}")
except Exception as e:
    st.error(f"❌ Prediction error: {e}")
