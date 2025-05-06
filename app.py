import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import zipfile
import subprocess
import gdown

# -------------------- Google Drive File IDs --------------------
MODELS_FOLDER_ID = "1RtVur5iM46wZfM5F86t15wpdsE1xdf7i"  # Drive folder with unzipped .pkl files
DATA_FILE_ID     = "1ThLQ_PEE5uceKdPry5erfRtyrGT2Xy9C"  # metro.tsv.gz

# -------------------- Step 1: Download model folder from shared Drive --------------------
@st.cache_resource
def download_model_folder_from_drive():
    if not os.path.exists("metro_models"):
        os.makedirs("metro_models", exist_ok=True)
        st.info("📥 Downloading model files from Drive folder...")
        subprocess.run([
            "gdown",
            "--folder",
            "--id", MODELS_FOLDER_ID,
            "-O", "metro_models"
        ], check=True)
        st.success("✅ metro_models folder downloaded.")

download_model_folder_from_drive()

# -------------------- Step 2: Download metro data --------------------
@st.cache_resource
def download_data_file():
    if not os.path.exists("metro.tsv.gz"):
        st.info("📊 Downloading Redfin metro dataset...")
        gdown.download(f"https://drive.google.com/uc?id={DATA_FILE_ID}", "metro.tsv.gz", quiet=False)
        st.success("✅ Redfin data downloaded.")

download_data_file()

# -------------------- Step 3: Load and clean data --------------------
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

# -------------------- Step 4: Metro selection --------------------
st.sidebar.title("🏙️ Metro Selection")
metros = sorted(df['region'].unique().tolist())
selected = st.sidebar.selectbox("Choose a metro area", metros)

# -------------------- Step 5: Feature engineering --------------------
sub = df[df['region'] == selected].copy().sort_values('period_begin')
sub['rolling_avg_price'] = sub['median_sale_price'].rolling(3).mean()
sub['yoy_price_change'] = sub['median_sale_price'].pct_change(12)
sub['lag_1'] = sub['median_sale_price'].shift(1)

# -------------------- Step 6: Display chart --------------------
st.title(f"📊 Median Sale Price in {selected}")
st.line_chart(sub.set_index('period_begin')['median_sale_price'])

# -------------------- Step 7: Load model --------------------
safe_name = selected.replace(",", "").replace(" ", "_").replace("/", "_")
model_path = os.path.join("metro_models", f"{safe_name}.pkl")

# Debug sidebar info
st.sidebar.markdown("---")
st.sidebar.write("🔍 Looking for model file:")
st.sidebar.code(model_path)

# Show available model files
try:
    model_files = os.listdir("metro_models")
    st.sidebar.write("📂 Available models (sample):")
    st.sidebar.code(model_files[:10])
except Exception as e:
    st.sidebar.error(f"❌ Could not list metro_models/: {e}")
    st.stop()

# Handle missing models
if not os.path.exists(model_path):
    st.warning(f"🚫 Model not found for {selected}.\nExpected file: `{model_path}`")
    st.stop()

model = joblib.load(model_path)

# -------------------- Step 8: Prediction inputs --------------------
latest = sub.dropna().iloc[-1]
f1 = st.sidebar.number_input("Current Median Price", value=float(latest['median_sale_price']), step=1000.0)
f2 = st.sidebar.number_input("3-Month Avg Price", value=float(latest['rolling_avg_price']), step=1000.0)
f3 = st.sidebar.number_input("YoY % Change", value=float(latest['yoy_price_change']), step=0.01, format="%.3f")
f4 = st.sidebar.number_input("Last Month's Price", value=float(latest['lag_1']), step=1000.0)

X_pred = np.array([[f1, f2, f3, f4]])
pred = model.predict(X_pred)[0]

# -------------------- Step 9: Show prediction --------------------
st.header("💰 Predicted Median Price Next Month")
st.success(f"${pred:,.0f}")

if st.sidebar.checkbox("Show prediction inputs"):
    st.write(pd.DataFrame(X_pred, columns=['median_sale_price', 'rolling_avg_price', 'yoy_price_change', 'lag_1']))

