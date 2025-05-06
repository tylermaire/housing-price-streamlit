import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
import io
import gdown
import matplotlib.pyplot as plt
import base64

# -------------------- Google Drive Data File --------------------
DATA_FILE_ID = "1ThLQ_PEE5uceKdPry5erfRtyrGT2Xy9C"  # metro.tsv.gz
GITHUB_MODEL_BASE_URL = "https://raw.githubusercontent.com/tylermaire/housing-price-streamlit/main/metro_model"

# -------------------- Step 1: Download metro.tsv.gz --------------------
@st.cache_resource
def download_data_file():
    if not os.path.exists("metro.tsv.gz"):
        st.info("📊 Downloading Redfin metro dataset...")
        gdown.download(f"https://drive.google.com/uc?id={DATA_FILE_ID}", "metro.tsv.gz", quiet=False)
        st.success("✅ Redfin data downloaded.")

download_data_file()

# -------------------- Step 2: Load and clean metro dataset --------------------
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

# -------------------- Step 3: Metro Search Input --------------------
st.sidebar.title("\U0001F3D9\ufe0f Metro Search")
metros = sorted(df['region'].unique().tolist())
search = st.sidebar.text_input("Type a metro area (e.g. Atlanta, GA)", "")
matches = [m for m in metros if search.lower() in m.lower()]

if not matches:
    st.sidebar.warning("No matching metro areas. Try a different name.")
    st.stop()
elif len(matches) == 1:
    selected = matches[0]
else:
    selected = st.sidebar.selectbox("Select from matching areas", matches)

# -------------------- Step 4: Feature engineering --------------------
sub = df[df['region'] == selected].copy().sort_values('period_begin')
sub['rolling_avg_price'] = sub['median_sale_price'].rolling(3).mean()
sub['yoy_price_change'] = sub['median_sale_price'].pct_change(12)
sub['lag_1'] = sub['median_sale_price'].shift(1)

# -------------------- Step 5: Display chart --------------------
st.title(f"\U0001F4C8 Median Sale Price in {selected}")
st.line_chart(sub.set_index('period_begin')['median_sale_price'])

# -------------------- Step 6: Load model from GitHub --------------------
@st.cache_resource
def load_model_from_github(model_name):
    url = f"{GITHUB_MODEL_BASE_URL}/{model_name}"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return joblib.load(io.BytesIO(response.content))

safe_name = selected.replace(",", "").replace(" ", "_").replace("/", "_")
model_name = f"{safe_name}.pkl"
model = load_model_from_github(model_name)

if model is None:
    st.error(f"\u274c Model not found for {selected}. Expected GitHub file: `{model_name}`")
    st.stop()

sub_clean = sub.dropna(subset=['median_sale_price', 'rolling_avg_price', 'yoy_price_change', 'lag_1'])
if sub_clean.empty:
    st.warning(f"\u26a0\ufe0f Not enough data to make a prediction for {selected}. Try another metro area.")
    st.stop()

# -------------------- Step 7: Inputs and Prediction --------------------
latest = sub_clean.iloc[-1]
f1 = st.sidebar.number_input("Current Median Price", value=float(latest['median_sale_price']), step=1000.0)
f2 = st.sidebar.number_input("3-Month Avg Price", value=float(latest['rolling_avg_price']), step=1000.0)
f3 = st.sidebar.number_input("YoY % Change", value=float(latest['yoy_price_change']), step=0.01, format="%.3f")
f4 = st.sidebar.number_input("Last Month's Price", value=float(latest['lag_1']), step=1000.0)

X_pred = np.array([[f1, f2, f3, f4]])
pred = model.predict(X_pred)[0]

# -------------------- Step 8: Show Prediction --------------------
st.header("\U0001F4B0 Predicted Median Price Next Month")
st.success(f"${pred:,.0f}")

# Forecast plot
st.subheader("\U0001F4C9 Forecast Visualization")
forecast_df = sub_clean[['period_begin', 'median_sale_price']].copy()
next_month = forecast_df['period_begin'].max() + pd.DateOffset(months=1)
forecast_df = forecast_df.set_index('period_begin')
forecast_df.loc[next_month] = pred
st.line_chart(forecast_df['median_sale_price'])

# -------------------- Step 9: Download Button --------------------
result_df = pd.DataFrame(X_pred, columns=['median_sale_price', 'rolling_avg_price', 'yoy_price_change', 'lag_1'])
result_df['predicted_price'] = pred
result_df['metro'] = selected
result_df['date'] = pd.to_datetime("today").date()

csv = result_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Prediction as CSV",
    data=csv,
    file_name=f'{safe_name}_prediction.csv',
    mime='text/csv'
)

# -------------------- Step 10: Show Prediction Inputs --------------------
if st.sidebar.checkbox("Show prediction inputs"):
    st.write(result_df)
