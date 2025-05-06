import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# ------------------------------
# Download and Load Metro Data
# ------------------------------
@st.cache_data
def load_metro_data():
    st.info("📥 Downloading Redfin metro dataset...")
    url = "https://drive.google.com/uc?id=1ThLQ_PEE5uceKdPry5erfRtyrGT2Xy9C"
    response = requests.get(url)
    with open("metro.tsv.gz", "wb") as f:
        f.write(response.content)

    df = pd.read_csv("metro.tsv.gz", sep='\t', compression='gzip')
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df['period_begin'] = pd.to_datetime(df['period_begin'])
    df['median_sale_price'] = pd.to_numeric(df['median_sale_price'], errors='coerce')
    df = df[df['region_type'] == 'metro']
    df = df.dropna(subset=['region', 'period_begin', 'median_sale_price'])
    st.success("✅ Redfin data downloaded.")
    return df

# ------------------------------
# Load Metro Data
# ------------------------------
df = load_metro_data()

# ------------------------------
# Metro Search + Selection
# ------------------------------
st.sidebar.title("🏙️ Metro Selection")
search = st.sidebar.text_input("🔍 Search metro area")

metros = sorted(df['region'].unique().tolist())
filtered_metros = [m for m in metros if search.lower() in m.lower()] if search else metros
selected = st.sidebar.selectbox("Choose a metro area", filtered_metros)

# ------------------------------
# Data Prep for Selected Metro
# ------------------------------
sub = df[df['region'] == selected].copy().sort_values('period_begin')
sub['rolling_avg_price'] = sub['median_sale_price'].rolling(3).mean()
sub['yoy_price_change'] = sub['median_sale_price'].pct_change(12)
sub['lag_1'] = sub['median_sale_price'].shift(1)

if sub.dropna().empty:
    st.warning("⚠️ Not enough data for this metro area.")
    st.stop()

# ------------------------------
# 📊 Median Sale Price Chart
# ------------------------------
st.title(f"📉 Median Sale Price in {selected}")
st.line_chart(sub.set_index('period_begin')['median_sale_price'])

# ------------------------------
# 🔍 Model Loading
# ------------------------------
safe_name = selected.replace(",", "").replace(" ", "_").replace("/", "_")
model_url = f"https://raw.githubusercontent.com/tylermaire/housing-price-streamlit/main/metro_models/{safe_name}.pkl"
model_path = f"metro_models/{safe_name}.pkl"

if not os.path.exists(model_path):
    os.makedirs("metro_models", exist_ok=True)
    r = requests.get(model_url)
    if r.status_code == 200:
        with open(model_path, 'wb') as f:
            f.write(r.content)
    else:
        st.error(f"🚫 Model not found for {selected} Expected GitHub file: {safe_name}.pkl")
        st.stop()

model = joblib.load(model_path)

# ------------------------------
# 📈 Prediction UI
# ------------------------------
latest = sub.dropna().iloc[-1]
f1 = st.sidebar.number_input("Current Median Price", value=float(latest['median_sale_price']), step=1000.0)
f2 = st.sidebar.number_input("3-Month Avg Price", value=float(latest['rolling_avg_price']), step=1000.0)
f3 = st.sidebar.number_input("YoY % Change", value=float(latest['yoy_price_change']), step=0.01, format="%.3f")
f4 = st.sidebar.number_input("Last Month's Price", value=float(latest['lag_1']), step=1000.0)

X_pred = np.array([[f1, f2, f3, f4]])
pred = model.predict(X_pred)[0]

# ------------------------------
# 💰 Prediction Display
# ------------------------------
st.header("💰 Predicted Median Price Next Month")
st.success(f"${pred:,.0f}")

# ------------------------------
# 🔮 Forecast Visualization
# ------------------------------
st.subheader("🔮 Forecast Visualization")
forecast_df = sub[['period_begin', 'median_sale_price']].dropna().copy()
last_date = forecast_df['period_begin'].max()
next_month = last_date + pd.DateOffset(months=1)
forecast_df = forecast_df.append({
    'period_begin': next_month,
    'median_sale_price': pred
}, ignore_index=True)

fig, ax = plt.subplots()
forecast_df.set_index('period_begin')['median_sale_price'].plot(ax=ax, label='Actual + Forecast', color='deepskyblue')
ax.axvline(x=last_date, color='gray', linestyle='--', label='Prediction Start')
ax.set_ylabel("Price ($)")
ax.set_title("Forecast: Historical + Next Month")
ax.legend()
st.pyplot(fig)

# ------------------------------
# 📥 Download CSV
# ------------------------------
st.download_button(
    label="Download Prediction as CSV",
    data=forecast_df.to_csv(index=False),
    file_name=f'{safe_name}_forecast.csv',
    mime='text/csv'
)
