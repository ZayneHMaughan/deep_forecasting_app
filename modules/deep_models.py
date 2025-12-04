# deep_models.py

from neuralforecast import NeuralForecast
from neuralforecast.models import RNN, GRU, LSTM, NBEATS
import pandas as pd
from typing import Optional, List, Literal
import numpy as np
from data_config import prepare_data


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
        self.data_prep = prepare_data(
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

        self.models = self._initialize_models(models)

    def _initialize_models(self, model_names: List[str]):
        """Initialize NeuralForecast models with lightweight parameters for in-app training."""
        all_models = {}
        for name in model_names:
            if name.lower() == "rnn":
                all_models["rnn"] = RNN(h=self.h, input_size=2*self.h, max_steps=self.max_steps, early_stop_patience_steps=2)
            elif name.lower() == "gru":
                all_models["gru"] = GRU(h=self.h, input_size=2*self.h, max_steps=self.max_steps, early_stop_patience_steps=2)
            elif name.lower() == "lstm":
                all_models["lstm"] = LSTM(h=self.h, input_size=2*self.h, max_steps=self.max_steps, early_stop_patience_steps=2)
            elif name.lower() == "nbeats":
                all_models["nbeats"] = NBEATS(h=self.h, input_size=2*self.h, max_steps=self.max_steps, early_stop_patience_steps=2)
        return [all_models[name] for name in model_names if name in all_models]

    def fit(self, data: pd.DataFrame, target_col="y", date_col="ds", unique_id="series_1", val_size=2):
        """Fit deep learning model."""
        cleaned = self.data_prep.clean_miss_data(data)
        prepared = self.data_prep.wrangle_data(data=cleaned, target_col=target_col, date_col=date_col, unique_id=unique_id)

        self.nf = NeuralForecast(models=self.models, freq=self.freq)
        self.nf.fit(df=prepared, val_size=val_size)
        self.fitted_data = prepared
        return self

    def predict(self, h: Optional[int] = None):
        """Make predictions."""
        if self.nf is None:
            raise ValueError("Model must be fitted before prediction")
        return self.nf.predict() if h is None else self.nf.predict(h=h)
    
    def get_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = None) -> dict:
        """Compute standard metrics for DeepModels to match ML/Stat models interface."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred)**2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

        return {
            "MODEL": model_name or self.model_name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape
        }
