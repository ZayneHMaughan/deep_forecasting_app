"""
Streamlit App for Time Series Forecasting with Multiple Models
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import io

# Import your custom modules
from data_config import prepare_data
from Stats_models import StatModels
from ML_Models import MLModels
from Deep_models import DeepModels
from modeling import CompareModels
from graph_utils import GraphUtils
import pickle


# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Time Series Forecasting App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
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
#if 'graph_utils' not in st.session_state:
#    st.session_state.graph_utils = GraphUtils()


# ==========================================
# HELPER FUNCTIONS
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
# SIDEBAR - DATA UPLOAD & CONFIGURATION
# ==========================================
st.sidebar.title("⚙️ Configuration")

# Data Upload Section
st.sidebar.header("1. Data Upload")
data_source = st.sidebar.radio(
    "Choose data source:",
    ["Upload File", "Use Sample Data"]
)

if data_source == "Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV",
        type=['csv']
    )
    if uploaded_file:
        st.session_state.data = load_data(uploaded_file)
else:
    n_points = st.sidebar.slider("Number of data points", 100, 500, 200)
    add_missing = st.sidebar.checkbox("Add missing values", value=True)
    if st.sidebar.button("Generate Sample Data"):
        st.session_state.data = generate_sample_data(n_points, add_missing)

# Data Cleaning Configuration
st.sidebar.header("2. Data Cleaning")
clean_method = st.sidebar.selectbox(
    "Cleaning method:",
    ["impute", "fill", "drop"]
)
if clean_method == "impute":
    impute_strategy = st.sidebar.selectbox(
        "Imputation strategy:",
        ["simple", "knn", "iterative"]
    )
else:
    impute_strategy = "simple"

# Column Selection
if st.session_state.data is not None:
    st.sidebar.header("3. Column Selection")
    columns = st.session_state.data.columns.tolist()
    date_col = st.sidebar.selectbox("Date column:", columns)
    target_col = st.sidebar.selectbox("Target column:", [col for col in columns if col != date_col])


# ==========================================
# MAIN PAGE
# ==========================================
st.title("📈 Time Series Forecasting Dashboard")
st.markdown("Compare multiple forecasting models on your time series data")

# ==========================================
# TAB 1: DATA OVERVIEW
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🤖 Train Models", "📉 Compare Models"])

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
        
        # Visualize time series
        if date_col and target_col:
            st.subheader(f"Time Series Plot: {target_col}")
            fig, ax = plt.subplots(figsize=(12, 4))
            plot_data = data.copy()
            plot_data[date_col] = pd.to_datetime(plot_data[date_col])
            ax.plot(plot_data[date_col], plot_data[target_col], linewidth=2, color='steelblue')
            ax.set_xlabel('Date')
            ax.set_ylabel(target_col)
            ax.set_title(f'{target_col} Over Time')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("👈 Please upload data or generate sample data from the sidebar.")


# ==========================================
# TAB 2: TRAIN MODELS
# ==========================================
with tab2:
    st.header("Train Forecasting Models")
    
    if st.session_state.data is None:
        st.warning("Please load data first in the Data Overview tab.")
    else:
        # Split configuration
        st.subheader("Train/Test Split")
        split_ratio = st.slider("Training data percentage", 50, 90, 80)
        split_index = int(len(st.session_state.data) * split_ratio / 100)
        #max_steps = st.number_input("Max Training Steps", value=100)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training samples", split_index)
        with col2:
            st.metric("Test samples", len(st.session_state.data) - split_index)
        
        # Model selection
        st.subheader("Select Models to Train")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Statistical Models**")
            train_arima = st.checkbox("ARIMA", value=True)
            train_sarima = st.checkbox("SARIMA")
            train_ets = st.checkbox("Exponential Smoothing")
        
        with col2:
            st.markdown("**Machine Learning Models**")
            train_rf = st.checkbox("Random Forest", value=True)
            train_gb = st.checkbox("Light Gradient Boosting")
            train_xgb = st.checkbox("XGBoost")
        
        with col3:
            st.markdown("**Deep Learning Models**")
            train_lstm = st.checkbox("RNN", value=True)
            #epochs = st.number_input("Epochs (Deep Learning)", 10, 200, 50)
        
        # Train button
        if st.button("🚀 Train Selected Models", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Prepare data
            train_data = st.session_state.data.iloc[:split_index].copy()
            test_data = st.session_state.data.iloc[split_index:].copy()
            
            # Rename columns for consistency
            train_data = train_data.rename(columns={date_col: 'ds', target_col: 'y'})
            test_data = test_data.rename(columns={date_col: 'ds', target_col: 'y'})
            
            trained_models = []
            total_models = sum([train_arima, train_sarima, train_ets, 
                              train_rf, train_gb, train_xgb, train_lstm])
            current_model = 0
            
            # Train Statistical Models
            if train_arima:
                status_text.text("Training ARIMA...")
                try:
                    arima = StatModels(
                        models="arima",
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
                status_text.text("Training SARIMA...")
                try:
                    sarima = StatModels(
                        models="sarima",
                        clean_method=clean_method,
                        impute_strategy=impute_strategy
                    )
                    sarima.fit(train_data, target_col='y', order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                    trained_models.append(sarima)
                    st.success("✓ SARIMA trained successfully")
                except Exception as e:
                    st.error(f"✗ SARIMA training failed: {str(e)}")
                current_model += 1
                progress_bar.progress(current_model / total_models)
            
            if train_ets:
                status_text.text("Training Exponential Smoothing...")
                try:
                    ets = StatModels(
                        models="ets",
                        clean_method=clean_method,
                        impute_strategy=impute_strategy
                    )
                    ets.fit(train_data, target_col='y')
                    trained_models.append(ets)
                    st.success("✓ ETS trained successfully")
                except Exception as e:
                    st.error(f"✗ ETS training failed: {str(e)}")
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
                        models="lgbm",
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
                        models="xgboost",
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
            
            if train_lstm:
                st.info("Training deep learning LSTM model...")
                # Take last 50 points or fewer
                subset_size = min(50, len(train_data))
                train_data_subset = train_data.iloc[-subset_size:].copy()

                # Safe initialization and training with cache
                @st.cache_resource
                def get_rnn_model(clean_method, impute_strategy):
                    return DeepModels(
                        models=["rnn"],
                        clean_method=clean_method,
                        impute_strategy=impute_strategy,
                        h=3,          # small horizon
                        max_steps=5  # small steps
                    )

                @st.cache_resource
                def train_rnn_model(_dl_model, train_df):
                    _dl_model.fit(train_df, target_col='y', date_col='ds', val_size=5)
                    return _dl_model

                
                try:
                    lstm_model = get_rnn_model(clean_method, impute_strategy)
                    lstm_model = train_rnn_model(lstm_model, train_data_subset)
                    trained_models.append(lstm_model)
                   
                    st.success("Deep learning model trained!")


                except Exception as e:
                        st.error(f"Training failed: {e}")
                        

                current_model += 1
                progress_bar.progress(current_model / total_models)

            
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
        horizon = st.number_input("Forecast Horizon", value=10, min_value = 1)
         # Initialize GraphUtils if not already
        if 'graph_utils' not in st.session_state:
            st.session_state.graph_utils = GraphUtils()
        
        # Initialize comparator
        comparator = CompareModels()
        
        # Add all trained models
        for model in st.session_state.trained_models:
            comparator.add_model(model)
        
        # Evaluate models
        if st.button("📊 Evaluate Models", type="primary"):
            with st.spinner("Evaluating models..."):
                train_data = st.session_state.train_data
                test_data = st.session_state.test_data
                y_test = test_data['y'].values
                
                # Evaluate all models
                try:
                    # For statistical models, we just need the number of steps
                    metrics_df = comparator.evaluate_all(
                        test_data=test_data,
                        train_data=train_data, 
                        h =horizon
                    )
                    
                    st.session_state.metrics_df = metrics_df
                    st.write(metrics_df)
                    st.session_state.comparator = comparator
                    st.session_state.y_test = y_test
                    
                    st.success("✓ Evaluation complete!")
                    
                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")

        
        # Display results if available
        if st.session_state.comparator and hasattr(st.session_state, 'metrics_df'):
            st.subheader("Performance Metrics")
            st.dataframe(st.session_state.metrics_df, use_container_width=True)
            
            # Best model
             # Best model
            try:
                best_model = st.session_state.comparator.get_best_model()
                st.info(f"🏆 Best Model: **{best_model['model_name']}** (RMSE: {best_model['metrics']['RMSE']:.4f})")
            except ValueError:
                st.warning("No models successfully evaluated yet")
                best_model = None
            
            
            # Visualization
            st.subheader("Visualizations")
            
            # Predictions comparison
            st.markdown("**Predictions Comparison**")
            fig, ax = st.session_state.graph_utils.plot_predictions(
                st.session_state.y_test,
                st.session_state.comparator.predictions,
                title="Model Predictions vs Actual Values"
            )
            st.pyplot(fig)
            plt.close()
            
            # Metrics comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**RMSE Comparison**")
                fig, ax = st.session_state.graph_utils.plot_metrics_comparison(
                    st.session_state.metrics_df,
                    metric='RMSE'
                )
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.markdown("**MAE Comparison**")
                fig, ax = st.session_state.graph_utils.plot_metrics_comparison(
                    st.session_state.metrics_df,
                    metric='MAE'
                )
                st.pyplot(fig)
                plt.close()
            
            # # Residuals for best model
            # st.subheader(f"Residuals Analysis: {best_model['model_name']}")
            # best_pred = st.session_state.comparator.predictions[best_model['model_name']]
            # # Make sure it's a numpy array
            # if isinstance(best_pred, pd.Series):
            #     best_pred = best_pred.values
            # elif isinstance(best_pred, pd.DataFrame):
            #     best_pred = best_pred.iloc[:, 0].values  # take first column

            # y_true = st.session_state.y_test
            # if isinstance(y_true, pd.Series):
            #     y_true = y_true.values
            # elif isinstance(y_true, pd.DataFrame):
            #     y_true = y_true.iloc[:, 0].values

            # fig, axes = st.session_state.graph_utils.plot_residuals(
            #     y_true,
            #     best_pred,
            #     model_name=best_model['model_name']
            # )


# ==========================================
# TAB 4: RESULTS & EXPORT
# ==========================================
# with tab4:
#     st.header("Results & Export")
    
#     if st.session_state.comparator and hasattr(st.session_state, 'metrics_df'):
#         # Generate report
#         st.subheader("📋 Detailed Report")
#         report = st.session_state.comparator.generate_report()
#         st.text(report)
        
#         # Download report
#         st.download_button(
#             label="📥 Download Report (TXT)",
#             data=report,
#             file_name=f"model_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
#             mime="text/plain"
#         )
        
#         # Download metrics CSV
#         csv = st.session_state.metrics_df.to_csv(index=False)
#         st.download_button(
#             label="📥 Download Metrics (CSV)",
#             data=csv,
#             file_name=f"model_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#             mime="text/csv"
#         )
        
#         # Download predictions
#         predictions_df = pd.DataFrame(st.session_state.comparator.predictions)
#         predictions_df['Actual'] = st.session_state.y_test
#         predictions_csv = predictions_df.to_csv(index=False)
#         st.download_button(
#             label="📥 Download Predictions (CSV)",
#             data=predictions_csv,
#             file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#             mime="text/csv"
#         )
        
#     else:
#         st.info("Train and evaluate models to see results here.")


# ==========================================
# FOOTER
# ==========================================
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    **Time Series Forecasting App**
    
    This app allows you to:
    - Upload or generate time series data
    - Clean and prepare data
    - Train multiple forecasting models
    - Compare model performance
    - Export results
    
    Built with Streamlit, scikit-learn, and PyTorch
    """
)