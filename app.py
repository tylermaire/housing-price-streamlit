import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

st.set_page_config(page_title="🏠 Metro Price Predictor", layout="wide")

# ----------------------
# 1. Load & cache Redfin metro data
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

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df[df["region_type"] == "metro"]
    df["period_begin"]      = pd.to_datetime(df["period_begin"], errors="coerce")
    df["median_sale_price"] = pd.to_numeric(df["median_sale_price"], errors="coerce")
    return df.dropna(subset=["region", "period_begin", "median_sale_price"])

df = load_data()

# ----------------------
# 2. Sidebar: metro search & selection
# ----------------------
st.sidebar.title("🏙️ Metro Selection")
search     = st.sidebar.text_input("🔍 Search metros")
all_metros = sorted(df["region"].unique())
choices    = [m for m in all_metros if search.lower() in m.lower()] if search else all_metros
selected   = st.sidebar.selectbox("Choose a metro", choices)

# ----------------------
# 3. Plot historical median price
# ----------------------
sub = df[df["region"] == selected].sort_values("period_begin")
st.title(f"📊 Historical Median Sale Price — {selected}")
st.line_chart(sub.set_index("period_begin")["median_sale_price"])

# ----------------------
# 4. Feature engineering
# ----------------------
sub["rolling_avg"] = sub["median_sale_price"].rolling(3).mean()
sub["yoy_pct"]     = sub["median_sale_price"].pct_change(12) * 100
sub["lag_1"]       = sub["median_sale_price"].shift(1)

feature_cols = ["median_sale_price", "rolling_avg", "yoy_pct", "lag_1"]
features_df  = sub.dropna(subset=feature_cols)
if features_df.empty:
    st.warning("🚫 Not enough complete data for prediction features.")
    st.stop()

latest = features_df.iloc[-1]

# ----------------------
# 5. Load the HistGradientBoostingRegressor model
# ----------------------
@st.cache_resource
def load_model():
    path = os.path.join("metro_model_2", "hgb_log_price_model.pkl")
    if not os.path.exists(path):
        st.error(f"🚫 Model file not found: {path}")
        st.stop()
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"🚫 Failed to load model: {e}")
        st.stop()

model = load_model()

# ----------------------
# 6. (Optional) Show model feature importances if available
# ----------------------
if hasattr(model, "feature_importances_"):
    fi = pd.Series(model.feature_importances_, index=feature_cols)
    st.subheader("📈 Feature Importances")
    st.bar_chart(fi)

# ----------------------
# 7. Predict next‐month median price
# ----------------------
st.subheader("🔮 Predict Next Month’s Median Price")
f1 = st.number_input("Latest Median Price",     value=float(latest["median_sale_price"]))
f2 = st.number_input("3-Month Rolling Average", value=float(latest["rolling_avg"]))
f3 = st.number_input("Year-over-Year % Change", value=float(latest["yoy_pct"]))
f4 = st.number_input("Previous Month’s Price",  value=float(latest["lag_1"]))

X = np.array([[f1, f2, f3, f4]])
try:
    pred = model.predict(X)[0]
    st.metric("💰 Predicted Median Price Next Month", f"${pred:,.0f}")
except Exception as e:
    st.error(f"❌ Prediction failed: {e}")
    pred = None

# ----------------------
# 8. CSV & PDF downloads
# ----------------------
hist = sub[["period_begin", "median_sale_price"]].dropna().copy()
if pred is not None:
    nxt = hist["period_begin"].max() + pd.DateOffset(months=1)
    forecast = pd.DataFrame({"period_begin":[nxt],"median_sale_price":[pred]})
    hist = pd.concat([hist, forecast], ignore_index=True)

csv = hist.to_csv(index=False)
st.download_button("📥 Download Historical + Forecast CSV", csv, "forecast.csv", "text/csv")

def create_pdf():
    path = "summary.pdf"
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots()
        ax.plot(hist["period_begin"], hist["median_sale_price"], marker="o")
        ax.set_title(f"{selected} Price History + Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Median Sale Price")
        fig.autofmt_xdate()
        pdf.savefig(fig)
        plt.close(fig)
        if hasattr(model, "feature_importances_"):
            fig2, ax2 = plt.subplots()
            fi.plot(kind="bar", ax=ax2)
            ax2.set_title("Feature Importances")
            pdf.savefig(fig2)
            plt.close(fig2)
    return path

pdf_path = create_pdf()
st.download_button("📥 Download PDF Summary", open(pdf_path,"rb"), "summary.pdf","application/pdf")

# ----------------------
# 9. Custom‐value quick prediction
# ----------------------
st.subheader("🔧 Quick Predict From Custom House Value")
custom_price = st.number_input(
    "Enter a hypothetical current median price",
    value=float(latest["median_sale_price"])
)
custom_X = np.array([[
    custom_price,
    float(latest["rolling_avg"]),
    float(latest["yoy_pct"]),
    float(latest["lag_1"])
]])
try:
    custom_pred = model.predict(custom_X)[0]
    st.metric("📈 Predicted Next-Month Price", f"${custom_pred:,.0f}")
except Exception as e:
    st.error(f"❌ Custom prediction failed: {e}")

