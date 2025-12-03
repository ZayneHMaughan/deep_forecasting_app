from neuralforecast import NeuralForecast
from neuralforecast.models import (
    NBEATS,
    NHITS,
    RNN,
    LSTM,
    GRU,
    TCN,
    DeepAR,
    TFT
)
import pandas as pd
from typing import Optional, List, Dict
import numpy as np
from data_config import prepare_data


class DeepModels:
    def __init__(
        self,
        models: Optional[List[str]] = None,
        clean_method: str = "impute",
        impute_strategy: str = "knn",
        freq: str = 'D',
        h: int = 10,
        max_steps: int = 100
    ):
        """
        Initialize deep learning models using NeuralForecast.
        
        Parameters
        ----------
        models : list, optional
            List of model names: 'nbeats', 'nhits', 'lstm', etc.
        clean_method : str
            Data cleaning method
        impute_strategy : str
            Imputation strategy
        freq : str
            Frequency of time series
        h : int
            Forecast horizon
        max_steps : int
            Maximum training steps
        """
        self.data_prep = prepare_data(
            options=clean_method,
            imputed_options=impute_strategy
        )
        self.freq = freq
        self.h = h
        self.max_steps = max_steps
        self.nf = None
        self.fitted_data = None
        self.model_name = "NeuralForecast"
        
        # Initialize models
        if models is None:
            models = ['nbeats']
        
        self.models = self._initialize_models(models)
        
    def _initialize_models(self, model_names: List[str]):
        """Initialize NeuralForecast models."""
        model_dict = {
            'nbeats': NBEATS(
                h=self.h,
                input_size=2*self.h,
                max_steps=self.max_steps,
                early_stop_patience_steps=5
            ),
            'nhits': NHITS(
                h=self.h,
                input_size=2*self.h,
                max_steps=self.max_steps,
                early_stop_patience_steps=5
            ),
            'lstm': LSTM(
                h=self.h,
                input_size=2*self.h,
                max_steps=self.max_steps,
                early_stop_patience_steps=5
            ),
            'gru': GRU(
                h=self.h,
                input_size=2*self.h,
                max_steps=self.max_steps,
                early_stop_patience_steps=5
            ),
            'rnn': RNN(
                h=self.h,
                input_size=2*self.h,
                max_steps=self.max_steps,
                early_stop_patience_steps=5
            ),
            'tcn': TCN(
                h=self.h,
                input_size=2*self.h,
                max_steps=self.max_steps,
                early_stop_patience_steps=5
            ),
            'deepar': DeepAR(
                h=self.h,
                input_size=2*self.h,
                max_steps=self.max_steps,
                early_stop_patience_steps=5
            ),
            'tft': TFT(
                h=self.h,
                input_size=2*self.h,
                max_steps=self.max_steps,
                early_stop_patience_steps=5
            )
        }
        
        return [model_dict[name] for name in model_names if name in model_dict]
    
    def fit(
        self,
        data: pd.DataFrame,
        target_col: str = 'y',
        date_col: str = 'ds',
        unique_id: str = 'series_1',
        val_size: int = 10
    ) -> 'DeepModels':
        """
        Fit deep learning models.
        
        Parameters
        ----------
        data : pd.DataFrame
            Training data
        target_col : str
            Target column name
        date_col : str
            Date column name
        unique_id : str
            Unique identifier for the series
        val_size : int
            Validation set size
        """
        # Clean data
        cleaned = self.data_prep.clean_miss_data(data)
        
        # Prepare data in Nixtla format
        prepared = self.data_prep.prepare_data(
            data=cleaned,
            target_col=target_col,
            date_col=date_col,
            unique_id=unique_id
        )
        
        # Initialize NeuralForecast
        self.nf = NeuralForecast(
            models=self.models,
            freq=self.freq
        )
        
        # Fit models
        self.nf.fit(df=prepared, val_size=val_size)
        self.fitted_data = prepared
        
        print(f"✓ {self.model_name} fitted successfully with {len(self.models)} models")
        return self
    
    def predict(self, h: Optional[int] = None) -> pd.DataFrame:
        """
        Make predictions.
        
        Parameters
        ----------
        h : int, optional
            Forecast horizon (uses self.h if not provided)
            
        Returns
        -------
        pd.DataFrame
            Predictions with columns for each model
        """
        if self.nf is None:
            raise ValueError("Model must be fitted before prediction")
        
        if h is None:
            h = self.h
        
        predictions = self.nf.predict()
        return predictions
    
    def get_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = None) -> Dict[str, float]:
        """Calculate model metrics."""
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        return {
            'model': model_name or self.model_name,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def cross_validation(
        self,
        h: int = 10,
        n_windows: int = 3,
        step_size: int = 1
    ) -> pd.DataFrame:
        """Perform time series cross-validation."""
        if self.nf is None:
            raise ValueError("Model must be fitted before cross-validation")
        
        cv_results = self.nf.cross_validation(
            df=self.fitted_data,
            h=h,
            n_windows=n_windows,
            step_size=step_size, )