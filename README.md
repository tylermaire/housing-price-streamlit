# Metro Price Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://housing-price-app-9v5qmn6qtrdfd3updowqht.streamlit.app/)

An interactive Streamlit app for exploring historical median home sale prices across U.S. metropolitan areas and forecasting next-month prices using a lightweight, high-performance scikit-learn model.

---

## Features

- 🔍 **Searchable Metro Selector**  
  Quickly find and select any U.S. metro area.
- 📈 **Historical Trends**  
  Interactive line chart of median sale prices by month.
- 🔮 **Next-Month Forecast**  
  Predicts the next month’s median sale price using a 4-feature `HistGradientBoostingRegressor` model (lag-1, lag-12, 3-month rolling average of log-price).
- 📥 **Downloadable Reports**  
  - **CSV**: Full historical data plus your forecast.  
  - **PDF**: Summary report with charts and feature importances.
- 🔧 **Custom-Price Scenario**  
  Enter a hypothetical current median price and get an instant next-month forecast.

---

## Repo Structure

