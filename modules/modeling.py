import pandas as pd
import numpy as np
import streamlit as st

class CompareModels:
    def __init__(self, graph_utils=None):
        """Initialize model comparison class."""
        self.models = []
        self.metrics = []
        self.predictions = {}
        self.graph_utils = graph_utils  # optional GraphUtils instance

    def add_model(self, model, name: str = None):
        """Add a model to comparison."""
        model_name = name or getattr(model, 'model_name', f'Model_{len(self.models)}')
        model_name = str(model_name)

        self.models.append({'name': model_name, 'model': model})
        st.write(f"Added {model_name}")

    def evaluate_all(self, train_data: pd.DataFrame, test_data: pd.DataFrame, h: int = 10):
        """Evaluate all models and store predictions + metrics."""
        self.metrics = []
        self.predictions = {}
        self.results = {}

        if len(self.models) == 0:
            st.warning("No models to evaluate.")
            return pd.DataFrame()

        for model_info in self.models:
            name = model_info['name']
            model = model_info['model']

            try:
                # ---- CALL THE RIGHT FORECAST METHOD ------------------------------
                if hasattr(model, "one_step_forecast") and h == 1:
                    result = model.one_step_forecast(
                        train_df=train_data,
                        test_df=test_data,
                        target_col='y',
                        date_col='ds',
                        unique_id='series_1'
                    )
                elif hasattr(model, "multi_step_forecast"):
                    result = model.multi_step_forecast(
                        train_df=train_data,
                        horizon=h,
                        target_col='y',
                        date_col='ds',
                        unique_id='series_1',
                        test_df=test_data
                    )
                elif hasattr(model, "predict_multi_step"):
                    result = model.predict_multi_step(
                        h=h,
                        test_df=test_data,
                        target_col='y',
                        date_col='ds',
                        unique_id='series_1'
                    )
                elif hasattr(model, "predict"):
                    raw = model.predict(h=h)
                    if isinstance(raw, dict):
                        result = raw
                    else:
                        st.warning(f"{name} predict() did not return dict with 'forecasts', skipping.")
                        continue
                else:
                    st.warning(f"Model {name} has no predict() method, skipping.")
                    continue

                forecast_df = result.get('forecasts')
                if not isinstance(forecast_df, pd.DataFrame):
                    st.warning(f"{name} did not return a DataFrame for forecasts, skipping.")
                    continue

                if 'ds' in forecast_df.columns:
                    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

                # Store raw forecast with model name as key
                self.predictions[name] = forecast_df.copy()

                # ---- FIND PREDICTION COLUMN(S) -------------------------------------
                pred_cols = [
                    c for c in forecast_df.columns
                    if c not in ["unique_id", "ds", "y", "y_true"]
                ]

                if not pred_cols:
                    st.warning(f"{name}: no prediction column found in {forecast_df.columns.tolist()}")
                    continue

                pred_col = pred_cols[0]

                # ---- STANDARDIZE TIMESTAMPS ----------------------------------------
                if 'ds' in forecast_df.columns:
                    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'], errors='coerce')

                test_data_copy = test_data.copy()
                if test_data_copy['ds'].dtype != 'datetime64[ns]':
                    test_data_copy['ds'] = pd.to_datetime(test_data_copy['ds'], errors='coerce')

                # ---- ALIGN FORECASTS WITH TEST DATA ---------------------------------
                merged = pd.merge(
                    test_data_copy[['ds', 'y']],
                    forecast_df[['ds', pred_col]],
                    on='ds',
                    how='inner'
                )

                if merged.empty:
                    st.error(f"{name}: no overlapping timestamps between test and forecast")
                    continue

                y_true = merged["y"].values
                y_pred = merged[pred_col].values

                # ---- CHECK FOR NAN/INF VALUES --------------------------------------
                nan_in_true = np.isnan(y_true).sum()
                nan_in_pred = np.isnan(y_pred).sum()
                inf_in_true = np.isinf(y_true).sum()
                inf_in_pred = np.isinf(y_pred).sum()

                if nan_in_true > 0 or nan_in_pred > 0 or inf_in_true > 0 or inf_in_pred > 0:
                    st.warning(f"{name}: Data quality issues detected:")
                    st.write(f"  - NaN in actuals: {nan_in_true}")
                    st.write(f"  - NaN in predictions: {nan_in_pred}")
                    st.write(f"  - Inf in actuals: {inf_in_true}")
                    st.write(f"  - Inf in predictions: {inf_in_pred}")

                    # Show sample of problematic values
                    if nan_in_pred > 0:
                        st.write(f"  Sample predictions: {y_pred[:10]}")

                    # Option 1: Skip this model
                    # continue

                    # Option 2: Remove NaN/Inf values
                    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
                    y_true_clean = y_true[valid_mask]
                    y_pred_clean = y_pred[valid_mask]

                    if len(y_true_clean) == 0:
                        st.error(f"{name}: No valid data points after removing NaN/Inf, skipping")
                        continue

                    st.info(f"{name}: Using {len(y_true_clean)}/{len(y_true)} valid data points")
                    y_true = y_true_clean
                    y_pred = y_pred_clean

                # ---- COMPUTE METRICS ----------------------------------------------
                errors = y_true - y_pred

                # Additional safety checks
                if len(errors) == 0:
                    st.error(f"{name}: No errors to compute metrics from")
                    continue

                mae = float(np.mean(np.abs(errors)))
                rmse = float(np.sqrt(np.mean(errors ** 2)))

                # Safe MAPE calculation
                if np.all(y_true == 0):
                    mape = np.nan
                    st.warning(f"{name}: Cannot compute MAPE (all actual values are zero)")
                else:
                    # Avoid division by zero
                    mape_values = np.abs(errors / np.where(y_true == 0, 1e-10, y_true))
                    mape = float(np.mean(mape_values) * 100)

                metrics = {
                    "MODEL": name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape,
                    "N_POINTS": len(y_true)  # Track how many points were used
                }

                # Verify metrics are valid
                if np.isnan(rmse) or np.isinf(rmse):
                    st.error(f"{name}: RMSE is {rmse}, skipping this model")
                    st.write(f"  Debug - errors: {errors[:10]}")
                    st.write(f"  Debug - y_true: {y_true[:10]}")
                    st.write(f"  Debug - y_pred: {y_pred[:10]}")
                    continue

                self.metrics.append(metrics)

                # Store complete result
                self.results[name] = {
                    'forecast': forecast_df.copy(),
                    'metrics': metrics,
                    'predictions': y_pred,
                    'actuals': y_true,
                    'merged': merged.copy()
                }

                st.success(f"✓ {name} evaluated (RMSE = {metrics['RMSE']:.4f}, N = {metrics['N_POINTS']})")

            except Exception as e:
                import traceback
                st.error(f"✗ Error evaluating {name}: {e}")
                st.error(traceback.format_exc())

        # ---- RETURN SUMMARY FOR ALL MODELS (OUTSIDE LOOP) ------------------------
        if not self.metrics:
            st.warning("⚠️ No valid metrics were computed for any model!")
            return pd.DataFrame()

        metrics_df = pd.DataFrame(self.metrics)

        # Show summary of data quality
        st.write(f"📊 Successfully evaluated {len(metrics_df)} models")
        if 'N_POINTS' in metrics_df.columns:
            st.write(f"📈 Data points used: {metrics_df['N_POINTS'].min()} to {metrics_df['N_POINTS'].max()}")

        return metrics_df

    def get_best_model(self, metric="RMSE"):
        if not self.metrics:
            return None

        df = pd.DataFrame(self.metrics)
        if metric not in df.columns:
            metric = "RMSE"

        best_row = df.loc[df[metric].idxmin()]
        return {"model_name": best_row["MODEL"], "metrics": best_row.to_dict()}

    def _standardize_predictions(self, preds, train_index, horizon):
        # Convert DataFrame to Series if needed
        if isinstance(preds, pd.DataFrame):
            preds = preds.iloc[:, 0]

        # Ensure preds is numeric
        preds = pd.to_numeric(preds, errors="coerce")

        # Build correct forecast index
        forecast_index = pd.date_range(
            start=train_index[-1],
            periods=horizon + 1,
            freq=train_index.freq
        )[1:]  # skip the last training timestamp

        preds.index = forecast_index
        return preds



