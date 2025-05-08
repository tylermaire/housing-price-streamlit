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
        df = pd.read_csv(
            "metro.tsv.gz", sep="\t",
            compression="gzip", low_memory=False
        )
    except FileNotFoundError:
        url = (
            "https://redfin-public-data.s3.us-west-2.amazonaws.com/"
            "redfin_market_tracker/"
            "redfin_metro_market_tracker.tsv000.gz"
        )
        df = pd.read_csv(url, sep="\t", compression="gzip", low_memory=False)

    # Normalize
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(" ", "_")
    )
    # Keep only metros
    df = df[df["region_type"] == "metro"]
    df["period_begin"] = pd.to_datetime(
        df["period_begin"], errors="coerce"
    )
    df["median_sale_price"] = pd.to_numeric(
        df["median_sale_price"], errors="coerce"
    )
    return df.dropna(
        subset=["region", "period_begin", "median_sale_price"]
    )

df = load_data()

# ----------------------
# 2. Sidebar: search & select metro
# ----------------------
st.sidebar.title("🏙️ Metro Selection")
search     = st.sidebar.text_input("🔍 Search metros")
all_metros = sorted(df["region"].unique())
choices    = (
    [m for m in all_metros if search.lower() in m.lower()]
    if search else all_metros
)
selected   = st.sidebar.selectbox("Choose a metro", choices)

# ----------------------
# 3. Filter & plot history
# ----------------------
sub = df[df["region"] == selected].sort_values("period_begin")
st.title(f"📊 Historical Median Sale Price — {selected}")
st.line_chart(sub.set_index("period_begin")["median_sale_price"])

# ----------------------
# 4. Compute our 3 model features on log scale
# ----------------------
sub = sub.copy()
sub["log_price"] = np.log(sub["median_sale_price"])
sub["lag1"]  = sub["log_price"].shift(1)
sub["lag12"] = sub["log_price"].shift(12)
sub["roll3"] = sub["log_price"].rolling(3).mean()

features_df = sub.dropna(subset=["lag1", "lag12", "roll3"])
if features_df.empty:
    st.warning("🚫 Not enough data to compute features")
    st.stop()

latest = features_df.iloc[-1]

# ----------------------
# 5. Load the 4-feature HistGradientBoosting model
# ----------------------
@st.cache_resource
def load_model():
    path = os.path.join("metro_model_2", "hgb_4feat_model.pkl")
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
# 6. (Optional) Show feature importances
# ----------------------
if hasattr(model, "feature_importances_"):
    imp = pd.Series(
        model.feature_importances_,
        index=["lag1", "lag12", "roll3"]
    )
    st.subheader("📈 Feature Importances")
    st.bar_chart(imp)

# ----------------------
# 7. Predict next month’s price
# ----------------------
st.subheader("🔮 Predict Next Month’s Median Price")

# Prepare the 3 inputs
X = np.array([[
    latest["lag1"],
    latest["lag12"],
    latest["roll3"],
]])
pred_log = model.predict(X)[0]
pred_price = np.exp(pred_log)

st.metric("💰 Predicted Median Sale Price Next Month",
          f"${pred_price:,.0f}")

# ----------------------
# 8. Downloadable CSV & PDF
# ----------------------
hist = sub[["period_begin", "median_sale_price"]].dropna().copy()
next_month = hist["period_begin"].max() + pd.DateOffset(months=1)
forecast = pd.DataFrame({
    "period_begin": [next_month],
    "median_sale_price": [pred_price]
})
hist = pd.concat([hist, forecast], ignore_index=True)

# CSV
csv = hist.to_csv(index=False)
st.download_button(
    "📥 Download Historical + Forecast CSV",
    csv, "forecast.csv", "text/csv"
)

# PDF
def make_pdf():
    path = "summary.pdf"
    with PdfPages(path) as pdf:
        # Price chart
        fig, ax = plt.subplots()
        ax.plot(
            hist["period_begin"], hist["median_sale_price"], marker="o"
        )
        ax.set_title(f"{selected} Price History + Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Median Sale Price")
        fig.autofmt_xdate()
        pdf.savefig(fig)
        plt.close(fig)
        # Importances
        if hasattr(model, "feature_importances_"):
            fig2, ax2 = plt.subplots()
            imp.plot.bar(ax=ax2)
            ax2.set_title("Feature Importances")
            pdf.savefig(fig2)
            plt.close(fig2)
    return path

pdf_path = make_pdf()
st.download_button(
    "📥 Download PDF Summary",
    open(pdf_path, "rb"),
    "summary.pdf", "application/pdf"
)

# ----------------------
# 9. Quick‐value scenario
# ----------------------
st.subheader("🔧 Quick Predict from Custom Current Price")
custom_price = st.number_input(
    "Enter a hypothetical current median price",
    value=float(latest["median_sale_price"])
)
# Build custom features:
custom_log   = np.log(custom_price)
custom_X     = np.array([[
    custom_log,
    latest["lag12"],
    latest["roll3"]
]])
custom_log_pred = model.predict(custom_X)[0]
custom_price_pred = np.exp(custom_log_pred)
st.metric("📈 Forecast for Custom Price", f"${custom_price_pred:,.0f}")


# ----------------------
# 10. Legal disclaimer (displayed at the bottom of the sidebar)
# ----------------------
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Legal Disclaimer**  \n"
    "This application is provided **for research and testing purposes only**.  "
    "The housing‑price predictions and visualizations shown here are "
    "_not_ intended as real‑estate or investment advice, nor should they be "
    "used as the basis for any financial decision. Housing markets can be "
    "influenced by numerous unpredictable factors that are not captured in "
    "this simplified model. The authors assume **no liability** for decisions "
    "made using this tool."
