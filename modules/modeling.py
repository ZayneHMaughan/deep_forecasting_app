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
        
        # CRITICAL: Handle all cases where model_name might not be a string
        if isinstance(model_name, list):
            model_name = '_'.join(str(m) for m in model_name)
        elif not isinstance(model_name, str):
            model_name = str(model_name)
        
        self.models.append({'name': model_name, 'model': model})
        st.write(f"✓ Added {model_name} to comparison")

    @staticmethod
    def _format_predictions(pred, model_name: str):
        """
        Ensure predictions are returned as a DataFrame with consistent columns
        for CompareModels.
        """
        if isinstance(pred, pd.DataFrame):
            df = pred.copy()
            if 'unique_id' not in df.columns:
                df['unique_id'] = model_name
            if 'ds' not in df.columns:
                df['ds'] = np.arange(len(df))
        else:
            # Assume 1D array or Series
            df = pd.DataFrame({
                'y_pred': np.array(pred),
                'unique_id': model_name,
                'ds': np.arange(len(pred))
            })
        return df

    @staticmethod
    def _compute_metrics(model, y_true, y_pred, model_name=None):
        """Compute standard metrics."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        min_len = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:min_len], y_pred[:min_len]

        # Try calling model.get_metrics if available
        if hasattr(model, 'get_metrics'):
            metrics = model.get_metrics(y_true, y_pred, model_name=model_name)
        else:
            # Compute manually
            mae = np.mean(np.abs(y_true - y_pred))
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
            metrics = {
                'MODEL': model_name or getattr(model, 'model_name', 'Unknown'),
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'MAPE': mape
            }
        return metrics

    def evaluate_all(self, train_data: pd.DataFrame, test_data: pd.DataFrame, h: int = 10) -> pd.DataFrame:
        """Evaluate all models and store predictions + metrics."""
        self.metrics = []
        self.predictions = {}

        if len(self.models) == 0:
            st.warning("No models to evaluate.")
            return pd.DataFrame()

        for model_info in self.models:
            name = model_info['name']
            model = model_info['model']
            
            # ENSURE name is a string
            if not isinstance(name, str):
                name = str(name)

            try:
                # Determine forecast type
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
                    preds = model.predict(h=h)
                    if isinstance(preds, dict):
                        result = preds
                    elif isinstance(preds, pd.DataFrame):
                        result = {'forecasts': preds, 'metrics': None}
                    else:
                        st.warning(f"{name} predict() did not return expected format, skipping.")
                        continue
                else:
                    st.warning(f"Model {name} has no predict() method, skipping.")
                    continue

                forecast_df = result.get('forecasts')
                if not isinstance(forecast_df, pd.DataFrame):
                    st.warning(f"{name} did not return a DataFrame for forecasts, skipping.")
                    continue

                pred_cols = [c for c in forecast_df.columns if c not in ['unique_id', 'ds', 'y_true', 'y']]
                if len(pred_cols) == 0:
                    st.warning(f"No valid prediction columns for {name}, skipping.")
                    continue

                model_metrics = result.get('metrics', {})

                for col in pred_cols:
                    y_pred = forecast_df[col].values
                    y_true = test_data['y'].values[:len(y_pred)]
                    model_name_full = f"{name}_{col}" if len(pred_cols) > 1 else name
                    
                    # ENSURE model_name_full is a string
                    if not isinstance(model_name_full, str):
                        model_name_full = str(model_name_full)

                    if isinstance(model_metrics, dict) and col in model_metrics:
                        metrics = model_metrics[col].copy()
                        metrics['MODEL'] = model_name_full
                    elif isinstance(model_metrics, dict) and 'MAE' in model_metrics:
                        metrics = model_metrics.copy()
                        metrics['MODEL'] = model_name_full
                    else:
                        errors = y_true - y_pred
                        metrics = {
                            'MODEL': model_name_full,
                            'MAE': float(np.mean(np.abs(errors))),
                            'RMSE': float(np.sqrt(np.mean(errors ** 2))),
                            'MAPE': float(np.mean(np.abs(errors / (y_true + 1e-10))) * 100)
                        }

                    self.metrics.append(metrics)
                    # Store as list (hashable)
                    self.predictions[model_name_full] = y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred)
                    st.write(f"✓ Evaluated {model_name_full}: RMSE={metrics['RMSE']:.4f}")

            except Exception as e:
                st.error(f"✗ Error evaluating {name}: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

        if len(self.metrics) == 0:
            st.warning("No metrics could be computed. Returning empty DataFrame.")
            return pd.DataFrame()
        else:
            metrics_df = pd.DataFrame(self.metrics)
            
            # Ensure MODEL column exists
            if 'MODEL' not in metrics_df.columns:
                if 'model' in metrics_df.columns:
                    metrics_df.rename(columns={'model': 'MODEL'}, inplace=True)
                else:
                    metrics_df.rename(columns={metrics_df.columns[0]: 'MODEL'}, inplace=True)
            
            # CRITICAL: Ensure 'model' column contains ONLY strings
            metrics_df['model'] = metrics_df['MODEL'].astype(str)
            
            # Verify no lists remain
            for col in metrics_df.columns:
                if metrics_df[col].apply(lambda x: isinstance(x, list)).any():
                    st.warning(f"Column {col} contains lists, converting to strings")
                    metrics_df[col] = metrics_df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
            
            return metrics_df




    def _format_predictions(self, pred: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
            Ensure predictions are a proper DataFrame for evaluation.
        Handles multi-index and multi-column predictions.
        """
        if isinstance(pred, pd.Series):
            pred = pred.to_frame(name='prediction')

        if isinstance(pred.index, pd.MultiIndex):
            pred = pred.reset_index()

        # If no explicit prediction column, use the last one
        cols = [c for c in pred.columns if c not in ['unique_id', 'ds']]
        if 'prediction' not in cols and len(cols) > 0:
            pred = pred.rename(columns={cols[-1]: 'prediction'})

        return pred


    def get_best_model(self, metric='RMSE'):
        """Return best model based on metric."""
        if not self.metrics:
            st.warning("No metrics available.")
            return None
        metrics_df = pd.DataFrame(self.metrics)
        if metric not in metrics_df.columns:
            st.warning(f"Metric '{metric}' not found. Using RMSE instead.")
            metric = 'RMSE'
        best_idx = metrics_df[metric].idxmin()
        best_model_name = metrics_df.loc[best_idx, 'MODEL']
        return {
            'model_name': best_model_name,
            'metrics': metrics_df.loc[best_idx].to_dict()
        }
