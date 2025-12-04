# DeepForecasting App

**DeepForecasting** is an interactive Streamlit app for time series forecasting using modern machine learning models, including Nixtla’s MLForecast framework. It allows users to train, evaluate, and visualize forecasts from multiple ML models, including Random Forest, LightGBM, XGBoost, and CatBoost.

---

## 📐 Architecture

The app is structured to separate concerns between data handling, model logic, and visualization:

```
deep_forecasting_app/
│
├─ stream_lit2.py         # Main Streamlit app
├─ modules/               # Helper modules
│   ├─ __init__.py
│   ├─ ml_models.py       # MLModels class wrapping MLForecast models
│   ├─ data_config.py     # Data cleaning and wrangling
│   ├─ graph_utils.py     # Plotting functions (residuals, forecasts)
├─ data/                  # Optional: sample datasets
├─ requirements.txt       # Python dependencies
└─ README.md
```

**Flow:**

1. `stream_lit2.py` handles the Streamlit interface (file uploads, UI, buttons, progress bars).
2. `ml_models.py` wraps MLForecast models, providing:

   * One-step and multi-step forecasts
   * Cross-validation
   * Metrics calculation
3. `data_config.py` prepares time series data for MLForecast:

   * Missing data handling (imputation)
   * Long-format conversion
4. `graph_utils.py` handles all plots:

   * Forecast vs actual
   * Residual analysis
   * Metric summaries

---

## 📦 How Nixtla Packages Fit Together

This app leverages **Nixtla’s MLForecast** for ML-based forecasting:

* **MLForecast**: Core package for time series ML, supports multiple regressors and lags.
* **Lag and Target Transforms**:

  * `ExpandingMean`, `RollingStd`, etc., transform past values into features.
  * `Differences` can make target series stationary.
* **Workflow**:

  1. Data is cleaned and wrangled via `data_config.py`.
  2. Features (lags, date info) are generated.
  3. `MLModels` fits multiple MLForecast models in parallel.
  4. Predictions and metrics are returned for evaluation.

Optional dependencies:

* `xgboost`, `lightgbm`, `catboost` for additional regressors.
* Sklearn regressors like `RandomForestRegressor` and `LinearRegression`.

---

## 🚀 Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/username/deepforecasting-app.git
cd deepforecasting_app
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run stream_lit2.py
```

Open `http://localhost:8501` in your browser.

---

## ☁️ Deployment

### Deploy to Streamlit Cloud

1. Push your repository to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io/), sign in, and link your repo.
3. Configure the Python environment (requirements.txt).
4. Click **Deploy**.

### Deploy on Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "stream_lit2.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t deepforecasting-app .
docker run -p 8501:8501 deepforecasting-app
```

---

## 🛠 Features

* Train multiple ML models (RF, XGBoost, LightGBM, CatBoost)
* One-step and multi-step forecasting
* Rolling-window cross-validation
* Interactive evaluation of metrics (MAE, RMSE, MAPE)
* Residual analysis and forecast visualizations

---

## ⚡ Requirements

* Python ≥ 3.10
* Streamlit
* MLForecast
* scikit-learn
* NumPy, pandas, matplotlib
* Optional: xgboost, lightgbm, catboost

---

## 📖 References

* [MLForecast Documentation](https://nixtla.github.io/mlforecast/)
* [Streamlit Documentation](https://docs.streamlit.io/)
* [Scikit-learn Imputation](https://scikit-learn.org/stable/modules/impute.html)


