# U.S. Housing Price Prediction App

## Project Overview

This project is a Streamlit-based web application for exploring and forecasting U.S. housing prices. Users can select a metropolitan area to visualize its historical median housing prices and predict the next month's median price using a pre-trained machine learning model. The goal is to provide an interactive tool for understanding housing market trends in different regions. Recent market data shows how dynamic these trends can be (almost 90% of metro areas saw home price increases in Q4 2024):contentReference[oaicite:0]{index=0}, underscoring the importance of tools that help analyze and anticipate price movements. **Note:** This app is a proof-of-concept for research and educational purposes, not a product for real estate advice (see the Disclaimer below).

## Features

- **Metro Area Selection:** Choose from a list of U.S. metropolitan areas to view that region’s housing data. The app displays a time series of historical median home prices for the selected metro.
- **Interactive Trend Visualization:** View an interactive line chart of the historical median housing prices over time. Users can observe long-term trends, seasonal patterns, and volatility in the chosen metro’s housing market.
- **Next-Month Price Prediction:** Generate a forecast for the next month’s median house price. The app uses a pre-trained machine learning model to predict the upcoming month based on the latest available data and displays the result alongside the historical trend.
- **Data Export (CSV/PDF):** Download the displayed data and results for offline analysis. Users can export the historical prices and the model’s prediction as a CSV file. Additionally, the app provides an option to download a PDF report of the chart and prediction, enabling easy sharing or printing of the analysis.

## Methodology

**Data and Preprocessing:** The historical dataset consists of monthly median housing prices for various U.S. metro areas. These values represent typical home prices in each region (for example, Zillow’s Home Value Index reflects the median home value in the mid-tier segment of a market):contentReference[oaicite:1]{index=1}. To make the modeling more effective, we applied a log transformation to the price values. The log transform helps stabilize variance and normalize the distribution of housing prices:contentReference[oaicite:2]{index=2}, since raw home prices are often highly skewed. Before modeling, the data was sorted chronologically and split into training and testing sets based on time; the model is always trained on past data and evaluated on future data to avoid any look-ahead bias:contentReference[oaicite:3]{index=3}.

**Feature Engineering:** We engineered several time-series features to capture patterns and structures in the housing data. Key features included:
- *Lag features:* These are previous values of the time series (e.g., the price in the prior month, prior 2 months, etc.). Lag features provide the model with information on recent momentum or changes, under the assumption that recent history can help predict the near future:contentReference[oaicite:4]{index=4}. For example, the model considers the last month’s price and other recent prices as input features.
- *Rolling window statistics:* We calculated rolling averages (moving averages) over previous months, which smooth out short-term fluctuations and highlight longer-term trends. Rolling statistics help the model grasp the underlying trend amid seasonal swings or irregular variations:contentReference[oaicite:5]{index=5}.
- *Time indicators:* We included categorical time indicators such as the month of the year to account for seasonality. Housing markets often have seasonal patterns (for instance, prices might increase during certain times of the year), so we one-hot encoded the month to allow the model to learn season-specific effects.
- *Metro identifier:* If the model was trained across multiple metro areas combined, a categorical feature for the metro area was one-hot encoded so that the model could learn region-specific price levels and dynamics. (In some cases, separate models may be trained per metro; if so, a metro identifier is not needed for that model.)

All categorical features (like month or metro) were converted using one-hot encoding so that the HistGradientBoostingRegressor model could use them. The resulting feature set included the transformed price and its lags, rolling mean values over recent months, and dummy variables for categorical time components.

**Model Training:** The machine learning model behind the app is a **Histogram Gradient Boosting Regressor** (`HistGradientBoostingRegressor`) from scikit-learn. This is a decision-tree-based ensemble method that builds an additive model in a forward stage-wise fashion (a form of gradient boosting). The "histogram" variant is optimized for efficiency: scikit-learn’s histogram-based gradient boosting implementation (inspired by LightGBM) bins continuous features into discrete buckets to speed up training and handle large datasets:contentReference[oaicite:6]{index=6}. We chose this model for its strong performance on tabular data and ability to capture non-linear relationships. The model was trained on a time-based split of the data (e.g., training on data up to a certain year and testing on the period after that). Using a chronological split ensures that the model is always tested on data that comes **after** the training period, mimicking real-world forecasting and preventing information from the future leaking into training:contentReference[oaicite:7]{index=7}.

During training, the model learned to predict the **log-transformed** median price of the next month given the engineered features of previous months. Predictions from the model are then exponentiated (inverse of the log transform) to return to the original price scale for interpretability. The one-month-ahead forecasting approach means the model always predicts just the next point in the sequence. In the app, this prediction is updated as new data becomes available: when a user selects a metro, the app loads that metro’s latest data, computes the features for the most recent month, and then applies the trained model to predict the following month’s price.

## Technologies Used

This project leverages several technologies and libraries in the Python data science ecosystem:

- **Python 3.9+:** The core programming language used for all data processing, model training, and application logic.
- **Streamlit:** The web framework used to create the interactive dashboard. Streamlit simplifies turning Python scripts into sharable web apps, handling the UI for dropdowns, charts, and downloads.
- **pandas:** Used for data manipulation and analysis. The time series data for housing prices is loaded and preprocessed using pandas (e.g., computing log transforms, creating lagged features with DataFrame shifts, calculating rolling means).
- **scikit-learn:** Used for machine learning modeling. We utilize scikit-learn’s `HistGradientBoostingRegressor` for training the prediction model, along with tools for data splitting and evaluation.
- **Matplotlib/Plotly:** Used for creating visualizations. The historical price trends and prediction are plotted using a plotting library (for example, matplotlib for static plots or Plotly for interactive charts within Streamlit).
- **NumPy:** Used for numerical computations, such as applying the exponential function to reverse the log transform of predictions.
- **Joblib/Pickle:** (If applicable) Used for saving the trained model pipeline to disk so that the Streamlit app can load the pre-trained model without retraining each time.

These technologies work in tandem: pandas and NumPy handle data preparation, scikit-learn provides the modeling capabilities, and Streamlit with plotting libraries delivers an interactive user experience.

## Legal Disclaimer

**Disclaimer:** This application is intended for research and testing purposes only. The housing price predictions and visualizations in this app are **not** intended for real estate investment advice or financial planning. The model’s forecasts are based on historical trends and a simplified machine learning approach, which come with no guarantee of accuracy. Real-world housing markets can be influenced by many unpredictable factors (economic conditions, policy changes, etc.) that are not captured in this project. **Therefore, users should not rely on this app for any financial or property-related decisions.** Always consult professional real estate experts or financial advisors for advice related to housing investments. The creators of this project assume no liability for any decisions made based on the information provided by the app.

## References

National Association of REALTORS®. (2025, February 6). *Nearly 90% of Metro Areas Registered Home Price Gains in Fourth Quarter of 2024* [Press release]. Retrieved from https://www.nar.realtor/newsroom/nearly-90-of-metro-areas-registered-home-price-gains-in-fourth-quarter-of-2024

Scikit-learn. (2023). *Histogram-Based Gradient Boosting Regression.* In *scikit-learn User Guide* (Version 1.6.1). Retrieved from https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting

Yang, H., Kang, D., Hwang, K., Yang, Z., & Jiang, Y. (2018, March 25). *House Price Prediction with Creative Feature Engineering and Advanced Regression Techniques.* NYC Data Science Academy Blog. Retrieved from https://nycdatascience.com/blog/student-works/house-price-prediction-with-creative-feature-engineering-and-advanced-regression-techniques/

GeeksforGeeks. (2024, July 22). *Feature Engineering for Time-Series Data: Methods and Applications.* Retrieved from https://www.geeksforgeeks.org/feature-engineering-for-time-series-data-methods-and-applications/

Zillow Research. (2023). *Zillow Home Value Index (ZHVI) – Methodology and Data Definitions.* Retrieved from https://www.zillow.com/research/data/ (Housing Data documentation page)

En-nasiry, M. (2024, June 18). *Time Series Splitting Techniques: Ensuring Accurate Model Validation.* Medium. Retrieved from https://medium.com/@mouadenna/time-series-splitting-techniques-ensuring-accurate-model-validation-5a3146db3088
