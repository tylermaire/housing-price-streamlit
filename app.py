import streamlit as st
import pandas as pd
import requests
import io
import pickle
try:
    import joblib
except ImportError:
    joblib = None

# Caching the data load
@st.cache_data
def load_data():
    """Load and return the metro housing data from a local TSV file or Redfin URL."""
    try:
        df = pd.read_csv('metro.tsv.gz', sep='\t', compression='gzip')
    except FileNotFoundError:
        data_url = "https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/redfin_metro_market_tracker.tsv000.gz"
        df = pd.read_csv(data_url, sep='\t', compression='gzip')
    # Parse dates and sort by date for each region
    if 'period_begin' in df.columns:
        df['period_begin'] = pd.to_datetime(df['period_begin'])
        df.sort_values(['region', 'period_begin'], inplace=True)
    elif 'month' in df.columns and 'year' in df.columns:
        df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
        df.sort_values(['region', 'date'], inplace=True)
    else:
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'period' in col.lower()]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            df.sort_values(['region', date_cols[0]], inplace=True)
    return df

# Load data
data = load_data()

# Sidebar or main selection for metro area
st.title("📈 Metro Housing Market Explorer")
st.write("Select a metropolitan area to view historical housing market trends and predict the next month's median sale price.")
metros = sorted(data['region'].unique())
selected_metro = st.selectbox("Select Metro Area:", options=metros)

# Filter dataframe for the selected metro
metro_df = data[data['region'] == selected_metro]
if metro_df.empty:
    st.error("No data available for the selected metro area.")
    st.stop()

# Historical median sale price chart
st.subheader(f"Historical Median Sale Price – {selected_metro}")
if 'median_sale_price' in metro_df.columns:
    # Use period/date column as index for plotting
    if 'period_begin' in metro_df.columns:
        chart_df = metro_df[['period_begin', 'median_sale_price']].copy()
        chart_df.set_index('period_begin', inplace=True)
    elif 'date' in metro_df.columns:
        chart_df = metro_df[['date', 'median_sale_price']].copy()
        chart_df.set_index('date', inplace=True)
    else:
        chart_df = metro_df[['median_sale_price']].copy()
    st.line_chart(chart_df['median_sale_price'])
else:
    st.error("Median sale price data not found for this metro.")

# Caching model loading
@st.cache_resource
def load_model_for_metro(metro_name):
    """Download and load the model for the given metro (if available)."""
    # Construct filename and URL
    model_filename = metro_name.replace(", ", "_").replace(" ", "_") + ".pkl"
    model_url = f"https://raw.githubusercontent.com/tylermaire/housing-price-streamlit/main/metro_models/{model_filename}"
    try:
        resp = requests.get(model_url, timeout=5)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    model_bytes = resp.content
    # Try loading with joblib first, then pickle
    try:
        if joblib:
            model = joblib.load(io.BytesIO(model_bytes))
        else:
            model = pickle.loads(model_bytes)
    except Exception:
        try:
            model = pickle.loads(model_bytes)
        except Exception:
            model = None
    return model

# Load the model for the selected metro
model = load_model_for_metro(selected_metro)

# Prepare latest data features for prediction
latest_row = metro_df.iloc[-1]
# Latest median price
latest_price = float(latest_row['median_sale_price']) if 'median_sale_price' in latest_row else None
# 3-month rolling average of median price
if len(metro_df) >= 3:
    ma3 = float(metro_df['median_sale_price'].tail(3).mean())
else:
    ma3 = latest_price if latest_price is not None else 0.0
# Year-over-year change (%)
yoy_change = 0.0
if 'year' in metro_df.columns and 'month' in metro_df.columns:
    if len(metro_df) > 12:
        prev_year_mask = (
            (metro_df['year'] == latest_row['year'] - 1) &
            (metro_df['month'] == latest_row['month'])
        )
        prev_year_data = metro_df[prev_year_mask]
        if not prev_year_data.empty:
            prev_year_price = float(prev_year_data.iloc[-1]['median_sale_price'])
            if prev_year_price != 0:
                yoy_change = ((latest_price - prev_year_price) / prev_year_price) * 100
elif 'median_sale_price_yoy' in latest_row:
    yoy_change = float(latest_row['median_sale_price_yoy'])
# Previous month price
prev_price = float(metro_df.iloc[-2]['median_sale_price']) if len(metro_df) >= 2 else latest_price

# Prediction section
st.subheader("Predict Next Month's Median Sale Price")
if model is None:
    st.warning("No prediction model available for **{}**. Unable to forecast next month's price.".format(selected_metro))
else:
    st.markdown("Adjust the inputs if necessary and see the predicted next month’s median price:")
    price_input = st.number_input("Latest median sale price", value=latest_price or 0.0, format="%f")
    ma3_input = st.number_input("3-month rolling average", value=ma3 or 0.0, format="%f")
    yoy_input = st.number_input("Year-over-year change (%)", value=yoy_change or 0.0, format="%f")
    lag_input = st.number_input("Previous month median price", value=prev_price or 0.0, format="%f")
    # Predict using the model
    try:
        pred = model.predict([[price_input, ma3_input, yoy_input, lag_input]])[0]
        st.metric(label="🔮 Predicted Next Month Price", value=f"${pred:,.0f}")
    except Exception as e:
        st.error("An error occurred during prediction. Please check the input values or try a different metro.")
