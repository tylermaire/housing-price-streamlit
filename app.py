import streamlit as st
import pandas as pd
import requests
import io
import pickle
import joblib

st.set_page_config(page_title="Metro Housing Predictor", layout="wide")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('metro.tsv.gz', sep='\t', compression='gzip')
    except FileNotFoundError:
        url = "https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/redfin_metro_market_tracker.tsv000.gz"
        df = pd.read_csv(url, sep='\t', compression='gzip')

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Date handling
    if 'period_begin' in df.columns:
        df['period_begin'] = pd.to_datetime(df['period_begin'], errors='coerce')
        df.dropna(subset=['period_begin'], inplace=True)
        df.sort_values(by=['region', 'period_begin'], inplace=True)
    else:
        st.error("❌ Could not find a usable date column.")
        st.stop()

    return df

# Load dataset
df = load_data()

# Sidebar metro filter with search
st.sidebar.title("🏙️ Metro Area Selector")
metro_list = sorted(df['region'].dropna().unique().tolist())
search_term = st.sidebar.text_input("Search metros...")
filtered_metros = [m for m in metro_list if search_term.lower() in m.lower()]
selected_metro = st.sidebar.selectbox("Choose a metro:", filtered_metros if filtered_metros else metro_list)

# Filter for selected metro
sub_df = df[df['region'] == selected_metro].copy()
if sub_df.empty:
    st.error("❌ No data found for selected metro.")
    st.stop()

# Visualization
st.title(f"📊 Median Sale Price in {selected_metro}")
sub_df = sub_df.sort_values('period_begin')
st.line_chart(sub_df.set_index('period_begin')['median_sale_price'])

# Prepare features
sub_df['rolling_avg'] = sub_df['median_sale_price'].rolling(3).mean()
sub_df['yoy'] = sub_df['median_sale_price'].pct_change(12)
sub_df['lag_1'] = sub_df['median_sale_price'].shift(1)
latest = sub_df.dropna().iloc[-1] if not sub_df.dropna().empty else None

# Load model from GitHub
@st.cache_resource
def load_model(name):
    safe_name = name.replace(",", "").replace(" ", "_")
    url = f"https://raw.githubusercontent.com/tylermaire/housing-price-streamlit/main/metro_models/{safe_name}.pkl"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return joblib.load(io.BytesIO(resp.content))
    except Exception as e:
        st.warning(f"⚠️ Model error: {e}")
    return None

model = load_model(selected_metro)

# Predict
st.header("💰 Predicted Price Next Month")
if latest is None or model is None:
    st.warning("🚫 Not enough data or model not found for this metro.")
else:
    f1 = st.number_input("Latest Median Price", value=float(latest['median_sale_price']))
    f2 = st.number_input("3-Month Avg Price", value=float(latest['rolling_avg']))
    f3 = st.number_input("YoY % Change", value=float(latest['yoy']))
    f4 = st.number_input("Last Month's Price", value=float(latest['lag_1']))

    try:
        X = [[f1, f2, f3, f4]]
        pred = model.predict(X)[0]
        st.success(f"📈 Predicted Median Price: **${pred:,.0f}**")
    except Exception as e:
        st.error(f"Prediction error: {e}")


