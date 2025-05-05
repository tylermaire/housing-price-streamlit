import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import zipfile
import gdown

# -------------------- Google Drive File IDs --------------------
MODELS_FILE_ID = "1gVGV1XUzwoy1xlA1CcHCgAJgioGC-RHz"   # metro_models.zip
DATA_FILE_ID   = "1ThLQ_PEE5uceKdPry5erfRtyrGT2Xy9C"   # metro.tsv.gz

# -------------------- Step 1: Download files if not present --------------------
@st.cache_resource
def download_and_extract():
    if not os.path.exists("metro_models"):
        st.info("📦 Downloading model files...")
        gdown.download(f"https://drive.google.com/uc?id={MODELS_FILE_ID}", "metro_models.zip", quiet=False)
        with zipfile.ZipFile("metro_models.zip", "r") as zip_ref:
            zip_ref.extractall("metro_models")
        st.success("✅ Models downloaded and extracted.")

    if not os.path.exists("metro.tsv.gz"):
        st.info("📊 Downloading Redfin metro dataset...")
        gdown.download(f"https://drive.google.com/uc?id={DATA_FILE_ID}", "metro.tsv.gz", quiet=False)
        st.success("✅ Data file ready.")

download_and_extract()

# -------------------- Step 2: Load and clean metro-level data --------------------
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

# -------------------- Step 3: Sidebar metro selection --------------------
st.sidebar.title("🏙️ Metro Selection")
metros = sorted(df['region'].unique().tolist())
selected = st.sidebar.selectbox("Choose a metro area", metros)

# -------------------- Step 4: Feature engineering --------------------
sub = df[df['region'] == selected].copy().sort_values('period_begin')
sub['rolling_avg_price'] = sub['median_sale_price'].rolling(3).mean()
sub['yoy_price_change'] = sub['median_sale_price'].pct_change(12)
sub['lag_1'] = sub['median_sale_price'].shift(1)

# -------------------- Step 5: Display chart --------------------
st.title(f"📊 Median Sale Price in {selected}")
st.line_chart(sub.set_index('period_begin')['median_sale_price'])

# -------------------- Step 6: Load trained model --------------------
safe_name = selected.replace(",", "").replace(" ", "_").replace("/", "_")
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "metro_models", f"{safe_name}.pkl")

# 🔍 Debug sidebar info
st.sidebar.markdown("---")
st.sidebar.write("🔍 Looking for model file:")
st.sidebar.code(model_path)

# List a few available models
try:
    model_files = os.listdir(os.path.join(base_dir, "metro_models"))
    st.sidebar.write("📂 Models available (sample):")
    st.sidebar.code(model_files[:10])
except Exception as e:
    st.sidebar.error(f"❌ Error accessing metro_models/: {e}")
    st.stop()

# Load model or show warning
if not os.path.exists(model_path):
    st.warning(f"🚫 Model not found for {selected}\nExpected file: `{model_path}`")
    st.stop()

model = joblib.load(model_path)

# -------------------- Step 7: Sidebar prediction inputs --------------------
latest = sub.dropna().iloc[-1]
f1 = st.sidebar.number_input("Current Median Price", value=float(latest['median_sale_price']), step=1000.0)
f2 = st.sidebar.number_input("3-Month Avg Price", value=float(latest['rolling_avg_price']), step=1000.0)
f3 = st.sidebar.number_input("YoY % Change", value=float(latest['yoy_price_change']), step=0.01, format="%.3f")
f4 = st.sidebar.number_input("Last Month's Price", value=float(latest['lag_1']), step=1000.0)

X_pred = np.array([[f1, f2, f3, f4]])
pred = model.predict(X_pred)[0]

# -------------------- Step 8: Display prediction --------------------
st.header("💰 Predicted Median Price Next Month")
st.success(f"${pred:,.0f}")

if st.sidebar.checkbox("Show prediction inputs"):
    st.write(pd.DataFrame(X_pred, columns=['median_sale_price', 'rolling_avg_price', 'yoy_price_change', 'lag_1']))
