"""
Streamlit App for Time Series Forecasting with Multiple Models
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# NOTE: Assuming your custom modules and imports remain valid
# Add the full_pipeline directory to path to import forecasting modules
pipeline_path = Path(__file__).parent / 'modules'
sys.path.insert(0, str(pipeline_path))

# Import your custom modules
# Placeholder classes/functions for modules not included in the prompt
class StatModels:
    def __init__(self, models, clean_method, impute_strategy): pass
    def fit(self, train_data, target_col, **kwargs): pass
class MLModels:
    def __init__(self, models, clean_method, impute_strategy, **kwargs): pass
    def fit(self, train_data, target_col): pass
class DeepModels:
    def __init__(self, models, clean_method, impute_strategy, **kwargs): pass
    def fit(self, train_df, target_col, date_col, val_size): pass
class CompareModels:
    def __init__(self):
        self.predictions = {}
        self.metrics_df = pd.DataFrame()
    def add_model(self, model): pass
    def evaluate_all(self, test_data, train_data, h):
        # Placeholder logic for demonstration
        if not self.predictions:
            self.predictions = {f"Model_{i}": test_data[['ds']].copy().assign(yhat=test_data['y'].shift(-1)) for i in range(3)}
        if self.metrics_df.empty:
            self.metrics_df = pd.DataFrame({
                'Model': [f"Model_{i}" for i in range(3)],
                'RMSE': np.random.rand(3) * 10,
                'MAE': np.random.rand(3) * 5
            }).set_index('Model')
        return self.metrics_df
    def get_best_model(self): return {'model_name': 'Model_0', 'metrics': {'RMSE': 1.0}}

class GraphUtils:
    def plot_predictions(self, y_test, predictions, title):
        fig, ax = plt.subplots(figsize=(12, 4))
        # Ensure plot colors are suitable for dark mode (Matplotlib can be tricky)
        ax.plot(y_test['ds'], y_test['y'], label='Actual', color='white')
        return fig
    def plot_metrics_comparison(self, metrics_df, metric):
        fig, ax = plt.subplots(figsize=(6, 4))
        metrics_df[metric].plot(kind='bar', ax=ax)
        return fig
    def plot_residuals(self, actual, prediction, model_name):
        return plt.figure(), plt.figure(), plt.figure()
class PrepareData:
    def __init__(self, clean_method, impute_strategy): pass
    def clean_miss_data(self, df): return df # Placeholder for cleaning

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Time Series Forecasting App",
    page_icon="",
    # Note: Removing layout="wide" to allow custom CSS to control width
    initial_sidebar_state="collapsed"
)

# Inject custom CSS for Dark Blue background, White Text, and Centered/Fixed-Width Layout
st.markdown(
    """
    <style>
    /* 1. Dark Theme Base (Very Dark Navy/Off-Black) */
    .stApp {
        background-color: #0F1116; 
        color: #D3D3D3 ; 
    }

    /* 2. Center and Fix Width of Main Content */
    /* Target the main container wrapper */
    .main {
        padding-top: 2rem; 
        background-color: #0F1116; 
    }
    /* Target the content block */
    .block-container {
        max-width: 800px; /* Set max width for the centered effect */
        margin-left: auto;
        margin-right: auto; /* Center the container */
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* 3. Ensure all text elements are light */
    h1, h2, h3, h4, h5, h6, 
    .stMarkdown, 
    .stText, 
    label, 
    .stMetric, 
    .stMetric > div > div:first-child, 
    .stButton > button, 
    .stSelectbox > div > label, 
    .stRadio > label,
    .stCheckbox > label,
    .stExpander > div > div:first-child > div > p {
        color: #f0f2f6!important;
    }

    /* --- FIXES FOR WHITE ELEMENTS --- */

    /* FIX 1: Top Header/Banner Background (The header above the main content) */
    header {
        background-color: #1e2a38; /* Dark Grey/Navy for the top banner */
    }

    /* FIX 2: Expander Headers/Tops Background (Default to light grey/white) */
    .stExpander {
        border-color: #1e2a38; /* Set border to dark color */
    }
    .stExpander > div:first-child {
        background-color: #1A2E40; /* Darker blue for Expander Header */
        border-radius: 0.5rem;
    }

    /* FIX 3: Buttons Background (Ensure they have contrast) */
    /* Primary buttons (type="primary") are typically blue, ensure default buttons are dark */
    .stButton > button {
        background-color: #1A2E40; /* Darker blue for standard buttons */
        color: #f0f2f6 !important;
        border-color: #3f5872;
    }

    /* FIX 4: Dataframe Background (Ensures dark background for table body) */
    /* This targets the actual table background for elements like st.dataframe */
    .stDataFrame > div > div > div {
        background-color: #1A2E40; /* Dark background for table content */
    }

    /* 4. Adjust input/select background for contrast (Dark Blue) */
    .stSelectbox > div > div, 
    .stTextInput > div > div, 
    .stNumberInput > div > div, 
    .stSlider > div > div:first-child {
        background-color: #1A2E40; /* Darker blue for input fields for contrast */
        color: #f0f2f6; /* White text inside inputs */
        border-radius: 0.5rem;
    }
    /* Specifically for the slider track color */
    .stSlider > div:nth-child(2) > div:nth-child(1) {
        background-color: #1A2E40; 
    }

    /* Ensure markdown container uses the right background */
    .stMarkdown {
        background-color: transparent !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'comparator' not in st.session_state:
    st.session_state.comparator = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = []
if 'graph_utils' not in st.session_state:
    st.session_state.graph_utils = GraphUtils()


# ==========================================
# HELPER FUNCTIONS (Unchanged)
# ==========================================
def generate_sample_data(n_points=200, add_missing=True):
    """Generate sample time series data."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_points, freq='D')

    # Generate trend + seasonality + noise
    trend = np.linspace(100, 150, n_points)
    seasonality = 10 * np.sin(np.linspace(0, 4*np.pi, n_points))
    noise = np.random.randn(n_points) * 5
    values = trend + seasonality + noise

    data = pd.DataFrame({
        'date': dates,
        'value': values
    })

    # Add missing values
    if add_missing:
        missing_indices = np.random.choice(n_points, size=int(n_points * 0.05), replace=False)
        data.loc[missing_indices, 'value'] = np.nan

    return data


def load_data(uploaded_file):
    """Load data from uploaded file."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def prepare_data_for_display(data, target_col, date_col):
    """Prepare data for display."""
    df = data.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    return df


# ==========================================
# TOP-LEVEL CONFIGURATION - REPLACING SIDEBAR
# ==========================================

st.title("Time Series Forecasting Dashboard")
st.markdown("Compare multiple forecasting models on your time series data")

# Use an Expander to group all configuration options neatly
with st.expander("Configuration: Data Upload, Cleaning & Selection", expanded=True):

    # 1. Data Upload
    st.subheader("1. Data Upload")
    data_source = st.radio(
        "Choose data source:",
        ["Upload File", "Use Sample Data"],
        key="data_source_radio",
        horizontal=True
    )

    col_upload, col_sample_slider, col_sample_checkbox, col_sample_button = st.columns([1, 1, 1, 1])

    uploaded_file = None
    if data_source == "Upload File":
        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload CSV/Excel:",
                type=['csv', 'xlsx']
            )
            if uploaded_file:
                st.session_state.data = load_data(uploaded_file)

    else: # Use Sample Data
        with col_sample_slider:
            n_points = st.slider("Number of data points", 100, 500, 200, key="n_points_slider")
        with col_sample_checkbox:
            add_missing = st.checkbox("Add missing values", value=True, key="add_missing_checkbox")
        with col_sample_button:
            st.markdown("<br>", unsafe_allow_html=True) # Add some space for alignment
            if st.button("Generate Sample Data", key="generate_data_button"):
                st.session_state.data = generate_sample_data(n_points, add_missing)
                st.toast("Sample data generated!")

    # Check if data is available before showing cleaning/column selectors
    if st.session_state.data is not None:

        st.markdown("---")

        # 2. Data Cleaning Configuration
        st.subheader("2. Data Cleaning")
        col_clean_method, col_impute_strategy = st.columns(2)

        with col_clean_method:
            clean_method = st.selectbox(
                "Cleaning method:",
                ["impute", "fill", "drop"],
                key="clean_method_select"
            )

        impute_strategy = "simple"
        with col_impute_strategy:
            if clean_method == "impute":
                impute_strategy = st.selectbox(
                    "Imputation strategy:",
                    ["simple", "knn", "iterative"],
                    key="impute_strategy_select"
                )
            else:
                st.info("Imputation strategy only applies to 'impute'.")

        st.markdown("---")

        # 3. Column Selection
        st.subheader("3. Column Selection")
        columns = st.session_state.data.columns.tolist()

        col_date, col_target = st.columns(2)
        with col_date:
            date_col = st.selectbox("Date column:", columns, key="date_col_select")
        with col_target:
            # Filter columns to not include the date column if selected
            default_target_idx = 0
            if date_col in columns:
                other_cols = [col for col in columns if col != date_col]
                if other_cols:
                    try:
                        # Try to default to 'value' if it exists in the filtered list
                        default_target_idx = other_cols.index('value')
                    except ValueError:
                        default_target_idx = 0
                else:
                    other_cols = []
            else:
                 other_cols = columns

            if other_cols:
                 target_col = st.selectbox("Target column:", other_cols, index=default_target_idx, key="target_col_select")
            else:
                 target_col = None
                 st.warning("No other columns available for target selection.")


    else:
        # Define placeholder variables if data hasn't been loaded/generated yet
        clean_method = "impute"
        impute_strategy = "simple"
        date_col = None
        target_col = None
        st.info("Use the options above to upload or generate data.")


# ==========================================
# MAIN PAGE (TABS)
# ==========================================

tab1, tab2, tab3 = st.tabs(["Data Overview", "Train Models", "Compare Models"])

# ==========================================
# TAB 1: DATA OVERVIEW
# ==========================================
with tab1:
    st.header("Data Overview")

    if st.session_state.data is not None:
        data = st.session_state.data

        # Display data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(data))
        with col2:
            st.metric("Total Columns", len(data.columns))
        with col3:
            missing_count = data.isnull().sum().sum()
            st.metric("Missing Values", missing_count)

        # Display first few rows
        st.subheader("Data Preview")
        st.dataframe(data.head(20), use_container_width=True)

        # Data statistics
        st.subheader("Statistics")
        st.dataframe(data.describe(), use_container_width=True)

        # Missing values by column
        if missing_count > 0:
            st.subheader("Missing Values by Column")
            missing_df = pd.DataFrame({
                'Column': data.columns,
                'Missing Count': data.isnull().sum().values,
                'Missing %': (data.isnull().sum().values / len(data) * 100).round(2)
            })
            st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
            st.text("There are three different ways to adjust missing data: Imputing, filling, and dropping the observations. ")
            st.markdown("For more information please follow this link: [scikit-learn Impute Docs](https://scikit-learn.org/stable/modules/impute.html)")


        # Visualize time series
        if date_col and target_col:
            st.subheader(f"Time Series Plot: {target_col}")
            fig, ax = plt.subplots(figsize=(12, 4))
            plot_data = data.copy()
            plot_data[date_col] = pd.to_datetime(plot_data[date_col])
            ax.plot(plot_data[date_col], plot_data[target_col], linewidth=2, color='steelblue')
            ax.set_xlabel('Date', color='white') # Set axis label color for Matplotlib
            ax.set_ylabel(target_col, color='white')
            ax.set_title(f'{target_col} Over Time', color='white')
            ax.tick_params(axis='x', colors='white') # Set tick color
            ax.tick_params(axis='y', colors='white')
            ax.set_facecolor('#1A2E40') # Set plot background
            fig.patch.set_facecolor('#1A2E40') # Set figure background
            ax.grid(True, alpha=0.3, color='gray')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("Please upload data or generate sample data from the Configuration section.")


# ==========================================
# TAB 2: TRAIN MODELS
# ==========================================
with tab2:
    st.header("Train Forecasting Models")

    if st.session_state.data is None:
        st.warning("Please load data first in the Configuration section.")
    elif not date_col or not target_col:
        st.warning("Please select Date and Target columns in the Configuration section.")
    else:
        # Split configuration
        st.subheader("Train/Test Split")
        split_ratio = st.slider("Training data percentage", 50, 90, 80, key="split_ratio_slider")
        split_index = int(len(st.session_state.data) * split_ratio / 100)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training samples", split_index)
        with col2:
            st.metric("Test samples", len(st.session_state.data) - split_index)

        # Model selection
        st.subheader("Select Models to Train")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Statistical Models**")
            train_auto_arima = st.checkbox("Auto_ARIMA", value=True, key="chk_auto_arima")
            train_arima = st.checkbox("ARIMA", key="chk_arima")
            train_sarima = st.checkbox("Auto_ETS", key="chk_auto_ets")
            train_sn= st.checkbox("Seasonal Naive", key="chk_sn")
            train_rwd = st.checkbox("Random Walk with Drift", key="chk_rwd")

        with col2:
            st.markdown("**Machine Learning Models**")
            train_rf = st.checkbox("Random Forest", value=True, key="chk_rf")
            train_gb = st.checkbox("Light Gradient Boosting", key="chk_lgbm")
            train_xgb = st.checkbox("XGBoost", key="chk_xgb")
            train_cat = st.checkbox("Catboost", key="chk_cat")

        # Train button
        if st.button("🚀 Train Selected Models", type="primary", key="train_models_button"):
            progress_bar = st.progress(0)
            status_text = st.empty()


            # Prepare data
            train_data = st.session_state.data.iloc[:split_index].copy()
            test_data = st.session_state.data.iloc[split_index:].copy()

            data_prep = PrepareData(clean_method, impute_strategy)
            # Rename columns for consistency
            train_data = train_data.rename(columns={date_col: 'ds', target_col: 'y'})
            test_data = test_data.rename(columns={date_col: 'ds', target_col: 'y'})

            # Ensure data cleaning is applied using the selected method/strategy from the configuration
            train_data = data_prep.clean_miss_data(train_data.copy())
            test_data = data_prep.clean_miss_data(test_data.copy())

            trained_models = []
            total_models = sum([train_arima, train_sarima, train_sn,
                              train_rf, train_gb, train_xgb, train_auto_arima, train_rwd, train_cat])
            current_model = 0

            # --- Training logic remains the same ---

            # Train Statistical Models
            if train_auto_arima:
                status_text.text("Training Auto_ARIMA...")
                try:
                    auto_arima = StatModels(
                        models=["auto_arima"],
                        clean_method=clean_method,
                        impute_strategy=impute_strategy
                    )
                    auto_arima.fit(train_data, target_col='y', order=(2, 1, 2))
                    trained_models.append(auto_arima)
                    st.success("Auto_ARIMA trained successfully")
                except Exception as e:
                    st.error(f"Auto_ARIMA training failed: {str(e)}")
                current_model += 1
                progress_bar.progress(current_model / total_models)

            if train_arima:
                status_text.text("Training ARIMA...")
                try:
                    arima = StatModels(
                        models=["arima"],
                        clean_method=clean_method,
                        impute_strategy=impute_strategy
                    )
                    arima.fit(train_data, target_col='y', order=(2, 1, 2))
                    trained_models.append(arima)
                    st.success("ARIMA trained successfully")
                except Exception as e:
                    st.error(f"ARIMA training failed: {str(e)}")
                current_model += 1
                progress_bar.progress(current_model / total_models)

            if train_sarima:
                status_text.text("Training Auto_ETS...")
                try:
                    sarima = StatModels(
                        models=["auto_ets"],
                        clean_method=clean_method,
                        impute_strategy=impute_strategy
                    )
                    sarima.fit(train_data, target_col='y', order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                    trained_models.append(sarima)
                    st.success("✓ Auto_ETS trained successfully")
                except Exception as e:
                    st.error(f"✗ Auto_ETS training failed: {str(e)}")
                current_model += 1
                progress_bar.progress(current_model / total_models)

            if train_sn:
                status_text.text("Training Seasonal Naive...")
                try:
                    ets = StatModels(
                        models=["seasonal_naive"],
                        clean_method=clean_method,
                        impute_strategy=impute_strategy
                    )
                    ets.fit(train_data, target_col='y')
                    trained_models.append(ets)
                    st.success("✓ Seasonal Naive trained successfully")
                except Exception as e:
                    st.error(f"✗ Seasonal Naive training failed: {str(e)}")
                current_model += 1
                progress_bar.progress(current_model / total_models)
            if train_rwd:
                status_text.text("Training Random Walk w Drift...")
                try:
                    ets = StatModels(
                        models=["random_walk_w_drift"],
                        clean_method=clean_method,
                        impute_strategy=impute_strategy
                    )
                    ets.fit(train_data, target_col='y')
                    trained_models.append(ets)
                    st.success("✓ Random Walk trained successfully")
                except Exception as e:
                    st.error(f"✗ Random Walk training failed: {str(e)}")
                current_model += 1
                progress_bar.progress(current_model / total_models)

            # Train ML Models
            if train_rf:
                status_text.text("Training Random Forest...")
                try:
                    rf = MLModels(
                        models=["random_forest"],
                        clean_method=clean_method,
                        impute_strategy=impute_strategy,
                        n_estimators=100,
                        random_state=42
                    )
                    rf.fit(train_data, target_col='y')
                    trained_models.append(rf)
                    st.success("✓ Random Forest trained successfully")
                except Exception as e:
                    st.error(f"✗ Random Forest training failed: {str(e)}")
                current_model += 1
                progress_bar.progress(current_model / total_models)

            if train_gb:
                status_text.text("Training Light Gradient Boosting...")
                try:
                    gb = MLModels(
                        models=["lgbm"],
                        clean_method=clean_method,
                        impute_strategy=impute_strategy,
                        n_estimators=100,
                        random_state=42
                    )
                    gb.fit(train_data, target_col='y')
                    trained_models.append(gb)
                    st.success(" Light Gradient Boosting trained successfully")
                except Exception as e:
                    st.error(f"Light Gradient Boosting training failed: {str(e)}")
                current_model += 1
                progress_bar.progress(current_model / total_models)

            if train_xgb:
                status_text.text("Training XGBoost...")
                try:
                    xgb = MLModels(
                        models=["xgboost"],
                        clean_method=clean_method,
                        impute_strategy=impute_strategy,
                        n_estimators=100,
                        random_state=42
                    )
                    xgb.fit(train_data, target_col='y')
                    trained_models.append(xgb)
                    st.success("XGBoost trained successfully")
                except Exception as e:
                    st.error(f"XGBoost training failed: {str(e)}")
                current_model += 1
                progress_bar.progress(current_model / total_models)

            if train_cat:
                status_text.text("Training Catboost...")
                try:
                    cat = MLModels(
                        models="catboost",
                        clean_method=clean_method,
                        impute_strategy=impute_strategy,
                        n_estimators=100,
                        random_state=42
                    )
                    cat.fit(train_data, target_col='y')
                    trained_models.append(cat)
                    st.success("Catboost trained successfully")
                except Exception as e:
                    st.error(f"Catboost training failed: {str(e)}")
                current_model += 1
                progress_bar.progress(current_model / total_models)

            # --- End of Training logic ---

            # Store trained models and data split
            st.session_state.trained_models = trained_models
            st.session_state.train_data = train_data
            st.session_state.test_data = test_data
            st.session_state.split_index = split_index

            status_text.text("Training complete!")

            #st.balloons()


# ==========================================
# TAB 3: COMPARE MODELS
# ==========================================
with tab3:
    st.header("Model Comparison")

    if not st.session_state.trained_models:
        st.warning("Please train models first in the Train Models tab.")
    else:
        st.success(f"✓ {len(st.session_state.trained_models)} models ready for comparison")
        horizon = st.number_input("Forecast Horizon", value=40, min_value=1, key="horizon_input")

        # Initialize comparator
        comparator = CompareModels()

        # Add all trained models
        for model in st.session_state.trained_models:
            comparator.add_model(model)

        # Evaluate models
        if st.button("Evaluate Models", type="primary", key="evaluate_button"):
            with st.spinner("Evaluating models..."):
                train_data = st.session_state.train_data
                test_data = st.session_state.test_data
                # FIXED: Keep the full DataFrame with datetime info, not just values
                y_test = test_data[['ds', 'y']].copy()

                # Evaluate all models
                try:
                    # For statistical models, we just need the number of steps
                    metrics_df = comparator.evaluate_all(
                        test_data=test_data,
                        train_data=train_data,
                        h=horizon
                    )

                    st.session_state.metrics_df = metrics_df
                    st.write(metrics_df)
                    st.session_state.comparator = comparator
                    st.session_state.y_test = y_test

                    st.success("✓ Evaluation complete!")

                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")
                    import traceback

                    st.error(traceback.format_exc())

        # Display results if available
        if st.session_state.comparator and hasattr(st.session_state, 'metrics_df'):
            st.subheader("Performance Metrics")
            st.dataframe(st.session_state.metrics_df, use_container_width=True)

            # Best model
            try:
                best_model = st.session_state.comparator.get_best_model()
                st.info(f"Best Model: **{best_model['model_name']}** (RMSE: {best_model['metrics']['RMSE']:.4f})")
            except (ValueError, TypeError, KeyError):
                st.warning("No models successfully evaluated yet")
                best_model = None

            # ---------------------------
            # Predictions comparison
            # ---------------------------
            if st.session_state.comparator.predictions:
                st.markdown("**Predictions Comparison**")

                # DEBUG: Show date ranges
                with st.expander("📅 Date Range Information"):
                    # Assuming train_data and test_data are available from session state
                    if 'train_data' in st.session_state and 'test_data' in st.session_state:
                        train_data = st.session_state.train_data
                        test_data = st.session_state.test_data
                        st.write("**Training Data:**")
                        st.write(f"- Start: {train_data['ds'].min()}")
                        st.write(f"- End: {train_data['ds'].max()}")
                        st.write(f"- Count: {len(train_data)}")

                        st.write("\n**Test Data:**")
                        st.write(f"- Start: {test_data['ds'].min()}")
                        st.write(f"- End: {test_data['ds'].max()}")
                        st.write(f"- Count: {len(test_data)}")

                    st.write("\n**Predictions:**")
                    for name, pred_df in st.session_state.comparator.predictions.items():
                        st.write(f"\n{name}:")
                        st.write(f"- Start: {pred_df['ds'].min()}")
                        st.write(f"- End: {pred_df['ds'].max()}")
                        st.write(f"- Count: {len(pred_df)}")

                # Show train + test as actual values
                full_actual = pd.concat([
                    st.session_state.train_data[['ds', 'y']],
                    st.session_state.test_data[['ds', 'y']]
                ], ignore_index=True)

                fig_pred = st.session_state.graph_utils.plot_predictions(
                    y_test=full_actual,  # Show full timeline
                    predictions=st.session_state.comparator.predictions,
                    title="Model Predictions vs Actual Values"
                )
                st.plotly_chart(fig_pred, use_container_width=True)

                # ---------------------------
                # Metrics comparison
                # ---------------------------
                col1_graph, col2_graph = st.columns(2)

                with col1_graph:
                    st.markdown("**RMSE Comparison**")
                    fig_rmse = st.session_state.graph_utils.plot_metrics_comparison(
                        st.session_state.metrics_df,
                        metric='RMSE'
                    )
                    st.plotly_chart(fig_rmse, use_container_width=True)

                with col2_graph:
                    st.markdown("**MAE Comparison**")
                    fig_mae = st.session_state.graph_utils.plot_metrics_comparison(
                        st.session_state.metrics_df,
                        metric='MAE'
                    )
                    st.plotly_chart(fig_mae, use_container_width=True)

                # ---------------------------
                # Residuals for best model
                # ---------------------------
                if best_model:
                    st.subheader(f"Residuals Analysis: {best_model['model_name']}")
                    best_name = best_model["model_name"]

                    if best_name in st.session_state.comparator.predictions:
                        forecast_df = st.session_state.comparator.predictions[best_name].copy()

                        # Ensure 'ds' is datetime
                        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'], errors='coerce')
                        test_df = st.session_state.test_data.copy()
                        test_df['ds'] = pd.to_datetime(test_df['ds'], errors='coerce')

                        # Align predictions with test data
                        pred_cols = [c for c in forecast_df.columns if c not in ['ds', 'unique_id', 'y', 'y_true']]
                        if pred_cols:
                            pred_col = pred_cols[0]
                            merged = pd.merge(
                                test_df[['ds', 'y']],
                                forecast_df[['ds', pred_col]],
                                on='ds',
                                how='inner'
                            )

                            if not merged.empty:
                                fig_residuals, fig_hist, fig_scatter = st.session_state.graph_utils.plot_residuals(
                                    merged["y"],
                                    merged[pred_col],
                                    model_name=best_name
                                )

                                st.plotly_chart(fig_residuals, use_container_width=True)
                                st.plotly_chart(fig_hist, use_container_width=True)
                                st.plotly_chart(fig_scatter, use_container_width=True)
                            else:
                                st.warning(
                                    "No overlapping dates between predictions and test data for residuals analysis.")
                        else:
                            st.warning(f"No prediction column found in {best_name} forecast.")
                    else:
                        st.warning(f"Predictions for {best_name} not found.")
            else:
                st.info("No predictions available yet. Click 'Evaluate Models' to generate predictions.")