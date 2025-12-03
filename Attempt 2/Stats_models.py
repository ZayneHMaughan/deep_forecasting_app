"""Statistical forecasting models using StatsForecast"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from data_config import prepare_data
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    ARIMA,
    SeasonalNaive,
    RandomWalkWithDrift

)


class StatModels:
    def __init__(
        self,
        models: Optional[List[str]] = None,
        clean_method: str = "impute",
        impute_strategy: str = "knn",
        freq: str = 'D',
        season_length: int = 7
    ):
        """
        Initialize statistical models using StatsForecast.
        
        Parameters
        ----------
        models : list, optional
            List of model names: 'auto_arima', 'auto_ets', 'seasonal_naive', etc.
        clean_method : str
            Data cleaning method
        impute_strategy : str
            Imputation strategy
        freq : str
            Frequency of time series ('D', 'W', 'M', 'H', etc.)
        season_length : int
            Season length for seasonal models
        """
        self.data_prep = prepare_data(
            options=clean_method,
            imputed_options=impute_strategy
        )
        self.freq = freq
        self.season_length = season_length
        self.sf = None
        self.fitted_data = None
        self.model_name = "StatsForecast"
        
        # Initialize models
        if models is None:
            models = ['auto_arima']
        
        self.models = self._initialize_models(models)
        
    def _initialize_models(self, model_names: List[str]):
        """Initialize StatsForecast models."""
        model_dict = {
            'auto_arima': AutoARIMA(season_length=self.season_length),
            'auto_ets': AutoETS(season_length=self.season_length),
            'arima': ARIMA(order=(1, 1, 1)),
            'seasonal_naive': SeasonalNaive(season_length=self.season_length),
            'random_walk_w_drift':RandomWalkWithDrift()

        }
        
        return [model_dict[name] for name in model_names if name in model_dict]
    
    def fit(
        self,
        data: pd.DataFrame,
        target_col: str = 'y',
        date_col: str = 'ds',
        unique_id: str = 'series_1'
    ) -> 'StatModels':
        """
        Fit statistical models.
        
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
        
        # Initialize and fit StatsForecast
        self.sf = StatsForecast(
            models=self.models,
            freq=self.freq,
            n_jobs=-1
        )
        
        self.sf.fit(prepared)
        self.fitted_data = prepared
        
        print(f"✓ {self.model_name} fitted successfully with {len(self.models)} models")
        return self
    
    def predict(self, h: int = 10) -> pd.DataFrame:
        """
        Make predictions.
        
        Parameters
        ----------
        h : int
            Forecast horizon
            
        Returns
        -------
        pd.DataFrame
            Predictions with columns for each model
        """
        if self.sf is None:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self.sf.predict(h=h)
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
        if self.sf is None:
            raise ValueError("Model must be fitted before cross-validation")
        
        cv_results = self.sf.cross_validation(
            df=self.fitted_data,
            h=h,
            n_windows=n_windows,
            step_size=step_size
        )
        
        return cv_results