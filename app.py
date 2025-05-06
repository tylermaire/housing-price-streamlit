import streamlit as st
import pandas as pd
import requests
import io
import joblib

# ------------------
# Caching the data load
# ------------------
@st.cache_data
def load_data():
    """Load and return the metro housing data from a local TSV file or Redfin URL."""
    try:
        df = pd.read_csv("metro.tsv.gz", sep="\t", compression="gzip")
    except FileNotFoundError:
        url = "https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/redfin_metro_market_tracker.tsv000.gz"
        df = pd.read_csv(url, sep="\t", compression="gzip")

    df.columns = df.columns.str.lower()
    if "period_begin" in df.columns:
        df["period_begin"] = pd.to_datetime(df["period_begin"], errors="coerce")
        df.dropna(subset=["region", "period_begin", "median_sale_price"], inplace=True)
        df = df[df["region_type"] == "metro"]
        df.sort_values(by=["region", "period_begin"], inplace=True)
    return df

# ------------------
# Caching model loading
# ------------------
@st.cache_resource
def load_model(name):
    """Download and load the model for the given metro (if available)."""
    safe_name = name.replace(",", "").replace(" ", "_")
    url = f"https://raw.githubusercontent.com/tylermaire/housing-price-streamlit/main/metro_models/{safe_name}.pkl"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return joblib.load(io.BytesIO(resp.content))
    except Exception as e:
        st.warning(f"⚠️ Could not load model: {e}")
    return None

# ------------------
# Load data and build UI
# ------------------
df = load_data()
available_models = set()

# Precheck for available model files (based on GitHub list or known names)
metro_names = df["region"].unique().tolist()
for name in metro_names:
    safe_name = name.replace(",", "").replace(" ", "_")
    available_models.add(name)  # Assume all are added to GitHub

# ------------------
# Streamlit UI
# ------------------
st.title("🏙️ Metro Housing Market Price Predictor")
st.markdown("Select a metro area to view housing trends and predict next month's median sale price.")

# Only list metros we expect to have models for
valid_metros = sorted([m for m in metro_names if m in available_models])
selected_metro = st.selectbox("Choose a metro area:", valid_metros)

metro_df = df[df["region"] == selected_metro].copy()
if metro_df.empty or len(metro_df) < 3:
    st.error("🚫 Not enough data to display or make predictions for this metro.")
    st.stop()

# Show historical chart
st.subheader(f"📊 Historical Median Sale Price — {selected_metro}")
st.line_chart(
    metro_df.set_index("period_begin")["median_sale_price"].dropna()
)

# Prepare features
latest = metro_df.dropna().iloc[-1]
prev = metro_df.dropna().iloc[-2]
price_now = latest["median_sale_price"]
rolling_avg = metro_df["median_sale_price"].rolling(3).mean().iloc[-1]
yoy = latest.get("median_sale_price_yoy", 0)
prev_price = prev["median_sale_price"]

# Load model
model = load_model(selected_metro)

# Prediction
st.subheader("💰 Predict Next Month's Median Sale Price")
if model is None:
    st.warning(f"🚫 Model not found for {selected_metro}. Prediction unavailable.")
else:
    f1 = st.number_input("Latest Median Price", value=float(price_now))
    f2 = st.number_input("3-Month Rolling Avg", value=float(rolling_avg))
    f3 = st.number_input("YoY % Change", value=float(yoy))
    f4 = st.number_input("Previous Month's Price", value=float(prev_price))

    try:
        pred = model.predict([[f1, f2, f3, f4]])[0]
        st.metric("🔮 Predicted Next Month's Price", f"${pred:,.0f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
