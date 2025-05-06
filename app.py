import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
import io
import gdown

# -------------------- File URLs --------------------
DATA_FILE_ID = "1ThLQ_PEE5uceKdPry5erfRtyrGT2Xy9C"  # metro.tsv.gz
GITHUB_BASE_URL = "https://raw.githubusercontent.com/tylermaire/housing-price-streamlit/main/metro_model"

# -------------------- Download metro.tsv.gz --------------------
@st.cache_resource
def download_data_file():
    if not os.path.exists("metro.tsv.gz"):
        st.info("📊 Downloading Redfin metro dataset...")
        gdown.download(f"https://drive.google.com/uc?id={DATA_FILE_ID}", "metro.tsv.gz", quiet=False)
        st.success("✅ Redfin data downloaded.")

download_data_file()

# -------------------- Load and clean data --------------------
@st.cache_data
def load_data():
    df = pd.read_csv('metro.tsv.gz', sep='\t', compression='gzip')
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df['period_begin'] = pd.to_datetime(df['period_begin'])
    df['median_sale_price'] = pd.to_numeric(df['median_sale_price'], errors='coerce')
    df = df[df['region_type'] == 'metro']
    df = df.dropna(subset=['region', 'period_begin', 'median_sale_price'])
    return df

df = load_data()

# -------------------- Metro selection --------------------
st.sidebar.title("🏙️ Metro Selection")
metros = sorted(df['region'].unique().tolist())
selected = st.sidebar.selectbox("Choose a metro area", metros)

# -------------------- Feature engineering --------------------
sub = df[df['region'] == selected].copy().sort_values('period_begin')
sub['rolling_avg_price'] = sub['median_sale_price'].rolling(3).mean()
sub['yoy_price_change'] = sub['median_sale_price'].pct_change(12)
sub['lag_1'] = sub['median_sale_price'].shift(1)

# -------------------- Chart --------------------
st.title(f"📊 Median Sale Price in {selected}")
st.line_chart(sub.set_index('period_begin')['median_sale_price'])

# -------------------- Load model from GitHub --------------------
@st.cache_resource
def load_model_from_github(model_name):
    url = f"{GITHUB_BASE_URL}/{model_name}"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return joblib.load(io.BytesIO(response.content))

safe_name = selected.replace(",", "").replace(" ", "_").replace("/", "_")
model_name = f"{safe_name}.pkl"
model = load_model_from_github(model_name)

# -------------------- Handle missing model --------------------
if model is None:
    st.warning(f"🚫 Model not found for {selected}.\nExpected file: `{model_name}` on GitHub.")
    st.stop()

# -------------------- Prediction input --------------------
latest = sub.dropna().iloc[-1]
f1 = st.sidebar.number_input("Current Median Price", value=float(latest['median_sale_price']), step=1000.0)
f2 = st.sidebar.number_input("3-Month Avg Price", value=float(latest['rolling_avg_price']), step=1000.0)
f3 = st.sidebar.number_input("YoY % Change", value=float(latest['yoy_price_change']), step=0.01, format="%.3f")
f4 = st.sidebar.number_input("Last Month's Price", value=float(latest['lag_1']), step=1000.0)

X_pred = np.array([[f1, f2, f3, f4]])
pred = model.predict(X_pred)[0]

# -------------------- Prediction output --------------------
st.header("💰 Predicted Median Price Next Month")
st.success(f"${pred:,.0f}")

if st.sidebar.checkbox("Show prediction inputs"):
    st.write(pd.DataFrame(X_pred, columns=['median_sale_price', 'rolling_avg_price', 'yoy_price_change', 'lag_1']))
