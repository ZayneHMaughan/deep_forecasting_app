# deep_models.py

from neuralforecast import NeuralForecast
from neuralforecast.models import RNN, GRU, LSTM, NBEATS
import pandas as pd
from typing import Optional, List, Literal, Dict, Union
import numpy as np
from data_config import PrepareData


class DeepModels:
    """
    Deep learning forecasting models wrapper (NeuralForecast)
    """

    def __init__(
            self,
            models: Optional[List[str]] = None,
            clean_method: str = "impute",
            impute_strategy: str = "knn",
            freq: str = 'D',
            h: int = 3,
            max_steps: int = 5,
            forecast_strategy: Literal["one_step", "multi_step", "multi_output"] = "multi_step",
    ):
        """
        Initialize deep learning models.
        Use lightweight RNN for in-app training.
        Pre-trained GRU/LSTM can be loaded for larger datasets.
        """
        self.data_prep = PrepareData(
            options=clean_method,
            imputed_options=impute_strategy
        )
        self.freq = freq
        self.h = h
        self.max_steps = max_steps
        self.forecast_strategy = forecast_strategy
        self.nf = None
        self.fitted_data = None

        if models is None:
            models = ["rnn"]  # default demo model

        # Store model names for later use
        self.model_list = models if isinstance(models, list) else [models]

        # Set model_name (for compatibility with comparator)
        if len(self.model_list) == 1:
            self.model_name = self.model_list[0].lower()
        else:
            self.model_name = '_'.join([m.lower() for m in self.model_list])

        self.models = self._initialize_models(self.model_list)

    def _initialize_models(self, model_names: List[str]):
        """Initialize NeuralForecast models with lightweight parameters for in-app training."""
        all_models = {}
        for name in model_names:
            name_lower = name.lower()
            if name_lower == "rnn":
                all_models["rnn"] = RNN(h=self.h, input_size=2 * self.h, max_steps=self.max_steps,
                                        early_stop_patience_steps=2)
            elif name_lower == "gru":
                all_models["gru"] = GRU(h=self.h, input_size=2 * self.h, max_steps=self.max_steps,
                                        early_stop_patience_steps=2)
            elif name_lower == "lstm":
                all_models["lstm"] = LSTM(h=self.h, input_size=2 * self.h, max_steps=self.max_steps,
                                          early_stop_patience_steps=2)
            elif name_lower == "nbeats":
                all_models["nbeats"] = NBEATS(h=self.h, input_size=2 * self.h, max_steps=self.max_steps,
                                              early_stop_patience_steps=2)
        return [all_models[name.lower()] for name in model_names if name.lower() in all_models]

    def fit(self, data: pd.DataFrame, target_col="y", date_col="ds", unique_id="series_1", val_size=2):
        """Fit deep learning model."""
        cleaned = self.data_prep.clean_miss_data(data)
        prepared = self.data_prep.wrangle_data(data=cleaned, target_col=target_col, date_col=date_col,
                                               unique_id=unique_id)

        self.nf = NeuralForecast(models=self.models, freq=self.freq)
        self.nf.fit(df=prepared, val_size=val_size)
        self.fitted_data = prepared
        return self

    def predict(self, h: Optional[int] = None):
        """
        Make predictions - returns dict format for compatibility with CompareModels.

        Parameters
        ----------
        h : int, optional
            Forecast horizon. If None, uses the horizon specified during initialization.

        Returns
        -------
        dict
            Dictionary with 'forecasts' (DataFrame) and 'metrics' (None or dict)
        """
        if self.nf is None:
            raise ValueError("Model must be fitted before prediction")

        try:
            # Get raw predictions from NeuralForecast
            if h is None:
                forecast_df = self.nf.predict()
            else:
                forecast_df = self.nf.predict(h=h)

            # Reset index to get ds column
            if isinstance(forecast_df.index, pd.MultiIndex) or 'ds' not in forecast_df.columns:
                forecast_df = forecast_df.reset_index()

            # Rename columns to match our naming convention
            column_mapping = {
                'RNN': 'rnn',
                'GRU': 'gru',
                'LSTM': 'lstm',
                'NBEATS': 'nbeats'
            }
            forecast_df = forecast_df.rename(columns=column_mapping)

            # Ensure ds is datetime
            if 'ds' in forecast_df.columns:
                forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

            # Return in the expected format
            return {
                'forecasts': forecast_df,
                'metrics': None  # No metrics without test data
            }

        except Exception as e:
            raise RuntimeError(f"Prediction failed for {self.model_name}: {str(e)}")

    def predict_multi_step(
            self,
            h: int = 10,
            test_df: Optional[pd.DataFrame] = None,
            target_col: str = 'y',
            date_col: str = 'ds',
            unique_id: str = 'series_1'
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
        """
        Multi-step forecast using NeuralForecast.
        Returns predictions and optional metrics in the same format as StatModels.
        """
        if self.nf is None:
            raise ValueError("Model must be fitted before prediction")

        # Generate forecasts
        forecast_df = self.nf.predict(h=h)
        forecast_df = forecast_df.reset_index()

        # Rename columns to match our naming convention
        # NeuralForecast returns columns like 'RNN', 'LSTM', etc.
        column_mapping = {
            'RNN': 'rnn',
            'GRU': 'gru',
            'LSTM': 'lstm',
            'NBEATS': 'nbeats'
        }
        forecast_df = forecast_df.rename(columns=column_mapping)

        # Ensure ds is datetime
        if 'ds' in forecast_df.columns:
            forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

        # Get prediction columns (exclude metadata)
        exclude_cols = ['unique_id', 'ds', 'cutoff', 'y']
        model_cols = [c for c in forecast_df.columns if c not in exclude_cols]

        # Compute metrics if test data is provided
        metrics = {}
        if test_df is not None:
            # Prepare test data
            test_long = self.data_prep.wrangle_data(test_df, target_col, date_col, unique_id)

            # Compute metrics for EACH model column
            for model_col in model_cols:
                merged = forecast_df[['ds', model_col]].merge(
                    test_long[['ds', target_col]],
                    on='ds',
                    how='left'
                )

                actuals = merged[target_col].values
                preds = merged[model_col].values
                mask = ~pd.isna(actuals) & ~pd.isna(preds)

                if mask.sum() > 0:
                    errors = actuals[mask] - preds[mask]
                    metrics[model_col] = {
                        'MAE': float(np.mean(np.abs(errors))),
                        'RMSE': float(np.sqrt(np.mean(errors ** 2))),
                        'MAPE': float(np.mean(np.abs(errors / (actuals[mask] + 1e-10))) * 100) if np.all(
                            actuals[mask] != 0) else np.nan
                    }

        return {'forecasts': forecast_df, 'metrics': metrics if metrics else None}

    def one_step_forecast(
            self,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame,
            target_col: str = 'y',
            date_col: str = 'ds',
            unique_id: str = 'series_1'
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
        """
        One-step ahead forecasting using rolling origin evaluation.
        Returns predictions and metrics in the same format as StatModels.
        """
        # Prepare data
        train_data = self.data_prep.wrangle_data(train_df, target_col, date_col, unique_id)
        test_data = self.data_prep.wrangle_data(test_df, target_col, date_col, unique_id)

        preds = []
        actuals = test_data[target_col].values
        dates = test_data['ds'].values

        # Rolling origin forecast
        for i in range(len(test_data)):
            # Expand training with previous actuals
            current_train = pd.concat([train_data, test_data.iloc[:i]], ignore_index=True)

            # Create temporary model for this step
            temp_nf = NeuralForecast(models=self.models, freq=self.freq)
            temp_nf.fit(df=current_train, val_size=min(2, len(current_train) // 5))

            # Forecast 1 step ahead
            forecast = temp_nf.predict(h=1)
            forecast = forecast.reset_index()

            # Extract prediction (use first model column)
            exclude_cols = ['unique_id', 'ds', 'cutoff', 'y']
            pred_cols = [c for c in forecast.columns if c not in exclude_cols]

            if pred_cols:
                preds.append(forecast[pred_cols[0]].values[0])
            else:
                preds.append(np.nan)

        # Build forecast DataFrame
        forecast_df = pd.DataFrame({
            'unique_id': [unique_id] * len(preds),
            'ds': dates,
            'y_true': actuals,
            self.model_name: preds  # Use model name as column
        })

        # Compute metrics
        valid_mask = ~np.isnan(actuals) & ~np.isnan(preds)
        if valid_mask.sum() > 0:
            errors = actuals[valid_mask] - np.array(preds)[valid_mask]
            metrics = {
                'MAE': float(np.mean(np.abs(errors))),
                'RMSE': float(np.sqrt(np.mean(errors ** 2))),
                'MAPE': float(np.mean(np.abs(errors / actuals[valid_mask])) * 100) if np.all(
                    actuals[valid_mask] != 0) else np.nan
            }
        else:
            metrics = {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

        return {'forecasts': forecast_df, 'metrics': metrics}

    def get_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = None) -> dict:
        """Compute standard metrics for DeepModels to match ML/Stat models interface."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Remove NaN values
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        if len(y_true) == 0:
            return {
                "MODEL": model_name or self.model_name,
                "MAE": np.nan,
                "MSE": np.nan,
                "RMSE": np.nan,
                "MAPE": np.nan
            }

        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

        return {
            "MODEL": model_name or self.model_name,
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAPE": float(mape)
        }