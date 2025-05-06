import streamlit as st
import pandas as pd
import numpy as np
import requests, io, os, zipfile, joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import gdown

st.set_page_config(page_title="🏠 Metro Price Predictor", layout="wide")

# -----------------------
# 0. Ensure real models exist locally (download & unzip from Google Drive)
# -----------------------
DRIVE_ID = "1gVGV1XUzwoy1xlA1CcHCgAJgioGC-RHz"  # Your Google Drive ZIP file ID
ZIP_URL  = f"https://drive.google.com/uc?export=download&id={DRIVE_ID}"
MODELS_DIR = "metro_models"
ZIP_PATH = "metro_models.zip"

def ensure_models():
    if not os.path.isdir(MODELS_DIR) or len(os.listdir(MODELS_DIR)) == 0:
        st.info("🚚 Downloading model archive from Google Drive...")
        gdown.download(ZIP_URL, ZIP_PATH, quiet=False)
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(MODELS_DIR)
        os.remove(ZIP_PATH)
        st.success("✅ Models ready.")

ensure_models()

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
    df["period_begin"] = pd.to_datetime(df["period_begin"], errors="coerce")
    df["median_sale_price"] = pd.to_numeric(df["median_sale_price"], errors="coerce")
    df = df.dropna(subset=["region", "period_begin", "median_sale_price"])
    return df

df = load_data()

# ----------------------
# 2. Sidebar: metro search & selection
# ----------------------
st.sidebar.title("🏙️ Metro Selection")
search = st.sidebar.text_input("🔍 Search metros", "")
all_metros = sorted(df["region"].unique())
choices = [m for m in all_metros if search.lower() in m.lower()] if search else all_metros
selected = st.sidebar.selectbox("Choose a metro", choices)

# ----------------------
# 3. Plot historical median price
# ----------------------
sub = df[df["region"] == selected].sort_values("period_begin")
st.title(f"📊 Historical Median Sale Price — {selected}")
st.line_chart(sub.set_index("period_begin")["median_sale_price"])

# ----------------------
# 4. Compute features
# ----------------------
sub["rolling_avg"] = sub["median_sale_price"].rolling(3).mean()
sub["yoy_pct"]   = sub["median_sale_price"].pct_change(12) * 100
sub["lag_1"]     = sub["median_sale_price"].shift(1)
feature_cols = ["median_sale_price", "rolling_avg", "yoy_pct", "lag_1"]
features_df = sub.dropna(subset=feature_cols)
if features_df.empty:
    st.warning("🚫 Not enough complete data for prediction features.")
    st.stop()
latest = features_df.iloc[-1]

# ----------------------
# 5. Display model coefficients
# ----------------------
def load_local_model(name):
    # strip trailing ' metro area'
    if name.lower().endswith(" metro area"):
        name = name[: -len(" metro area")]
    safe = name.replace(",", "").replace(" ", "_").replace("/", "_")
    path = os.path.join(MODELS_DIR, f"{safe}_metro_area.pkl")
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

model = load_local_model(selected)

if model is None:
    st.error(f"🚫 No model found for {selected}.")
    st.stop()

# If model has coefficients, show them
if hasattr(model, 'coef_'):
    coeffs = model.coef_
    feats = ['median_sale_price','rolling_avg','yoy_pct','lag_1']
    fi = pd.DataFrame({'feature':feats,'coef':coeffs}).set_index('feature')
    st.subheader("📈 Model Coefficients")
    st.bar_chart(fi['coef'])

# ----------------------
# 6. Prediction inputs & CSV download
# ----------------------
st.subheader("🔮 Predict Next Month’s Median Price")
f1 = st.number_input("Latest Median Price",        value=float(latest["median_sale_price"]), help="The median sale price in the most recent period.")
f2 = st.number_input("3-Month Rolling Average",    value=float(latest["rolling_avg"]),       help="Average of the last 3 months' median sale prices.")
f3 = st.number_input("Year-over-Year % Change",    value=float(latest["yoy_pct"]),           help="% change compared to same month last year.")
f4 = st.number_input("Previous Month’s Price",     value=float(latest["lag_1"]),             help="Median sale price one month ago.")

X = np.array([[f1, f2, f3, f4]])
try:
    pred = model.predict(X)[0]
    st.metric("💰 Predicted Median Price Next Month", f"${pred:,.0f}")
except Exception as e:
    st.error(f"❌ Prediction failed: {e}")

# CSV download of historical + forecast
hist = sub[['period_begin','median_sale_price']].dropna().copy()
next_month = hist['period_begin'].max() + pd.DateOffset(months=1)
# Add forecast row for next month
new_row = pd.DataFrame({'period_begin':[next_month],'median_sale_price':[pred]})
hist = pd.concat([hist, new_row], ignore_index=True)
csv = hist.to_csv(index=False)
st.download_button("📥 Download Historical + Forecast CSV", csv, file_name="forecast.csv", mime="text/csv")

# ----------------------
# 7. PDF summary download
# ----------------------
pdf_path = 'summary.pdf'
with PdfPages(pdf_path) as pdf:
    # Chart page
    fig, ax = plt.subplots()
    ax.plot(hist['period_begin'], hist['median_sale_price'], marker='o')
    ax.set_title(f'{selected} Price History + Forecast')
    ax.set_ylabel('Price ($)')
    pdf.savefig(fig)
    plt.close(fig)
    # Coefficients page
    if hasattr(model, 'coef_'):
        fig2, ax2 = plt.subplots()
        fi['coef'].plot(kind='bar', ax=ax2)
        ax2.set_title('Model Coefficients')
        pdf.savefig(fig2)
        plt.close(fig2)

with open(pdf_path, 'rb') as f:
    pdf_bytes = f.read()
st.download_button("📥 Download PDF Summary", pdf_bytes, file_name="summary.pdf", mime="application/pdf")
