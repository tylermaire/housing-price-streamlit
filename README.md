Metro Price Predictor



An interactive Streamlit app for exploring historical median home sale prices across U.S. metropolitan areas and forecasting next-month prices using a lightweight, high-performance scikit-learn model.

Features

🔍 Searchable Metro Selector: Quickly find and select any U.S. metro area.

📈 Historical Trends: Interactive line chart of median sale prices by month.

🔮 Next-Month Forecast: Predicts the next month’s median sale price using a 4-feature HistGradientBoostingRegressor model (lag-1, lag-12, 3‑month rolling average).

📥 Downloadable Reports:

CSV: Full historical data plus your forecast.

PDF: Summary report with charts and feature importances.

🔧 Custom‑Price Scenario: Enter a hypothetical current median price and get an instant next‑month forecast.

Repo Structure

housing-price-streamlit/
├── app.py                  # Main Streamlit app
├── metro.tsv.gz            # Redfin metro data (or auto-download)
├── metro_model_2/          # Folder with trained model files
│   └── hgb_4feat_model.pkl # 4‑feature HGBRegressor pickle
├── requirements.txt        # Python dependencies
└── README.md               # This file

Installation & Local Deployment

Clone this repo:

git clone https://github.com/tylermaire/housing-price-streamlit.git
cd housing-price-streamlit

Install dependencies (preferably in a virtual environment):

pip install -r requirements.txt
git lfs install               # ensure LFS is configured
git lfs pull                  # fetch the large model file

Run the Streamlit app:

streamlit run app.py

Your browser will open at http://localhost:8501/.

Data Source

We use the free public Redfin metro tracker data (2012–present), automatically downloaded and cached by the app if metro.tsv.gz is missing.

Model Training (Colab)

To retrain or improve the forecasting model, run the included Colab snippet:

# In a fresh Colab notebook
!pip install gdown scikit-learn --quiet

import gdown, pandas as pd, numpy as np, joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1) Download data
DRIVE_ID = "1tASwRQrSNs9689v-6PY0KcA2qOPiyg03"
gdown.download(
    f"https://drive.google.com/uc?export=download&id={DRIVE_ID}",
    "metro_data.tsv", quiet=False
)
df = pd.read_csv("metro_data.tsv", sep="\t", encoding="utf-16")

# 2) Clean & parse price, date, log transforms, lags, rolling
# ... (see training cell above) ...

# 3) Split, train HGBRegressor, evaluate
model = HistGradientBoostingRegressor(...)
model.fit(X_train, y_train)

# 4) Save model
joblib.dump(model, 'hgb_4feat_model.pkl')

Upload the resulting hgb_4feat_model.pkl to metro_model_2/ (Git LFS) to update forecasts in production.

Deployment on Streamlit Cloud

Your app is already live at:

https://housing-price-app-9v5qmn6qtrdfd3updowqht.streamlit.app/

Any push to main triggers an auto-deploy. Ensure requirements.txt includes:

streamlit
pandas
numpy
scikit-learn
python-dateutil
joblib

and that metro_model_2/hgb_4feat_model.pkl is LFS-managed.

Contributing

Contributions welcome! Feel free to open issues or PRs for:

🔧 Bug fixes or environment tweaks

✨ Feature requests (e.g. ZIP/county level forecasts)

📈 Model improvements (e.g. neural nets, feature expansion)

License

This project is licensed under the MIT License. See LICENSE for details.

Built with ❤️ using Streamlit
