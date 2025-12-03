from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from numba import njit
import window_ops.expanding as expanding
import window_ops.rolling as rolling
from data_config import prepare_data
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

class MLModels:
    def __init__(
        self,
        models: Optional[Dict[str, Any]] = None,
        clean_method: str = "impute",
        impute_strategy: str = "knn",
        freq: str = 'D',
        lags: Optional[List[int]] = None,
        lag_transforms: Optional[Dict] = None
    ):
        """
        Initialize ML models using MLForecast.
        
        Parameters
        ----------
        models : dict, optional
            Dictionary of {model_name: model_instance}
        clean_method : str
            Data cleaning method
        impute_strategy : str
            Imputation strategy
        freq : str
            Frequency of time series
        lags : list, optional
            List of lag features to create
        lag_transforms : dict, optional
            Transformations to apply to lags
        """
        self.data_prep = prepare_data(
            options=clean_method,
            imputed_options=impute_strategy
        )
        self.freq = freq
        self.lags = lags or [1, 7, 14]
        self.mlf = None
        self.fitted_data = None
        self.model_name = "MLForecast"
        
        # Initialize models
        if models is None:
            models = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'lgbm': LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)
            }
        
        self.models = models
        
        # Default lag transforms
        if lag_transforms is None:
            self.lag_transforms = {
                1: [expanding.mean, expanding.std],
                7: [rolling.mean, rolling.std, rolling.min, rolling.max]
            }
        else:
            self.lag_transforms = lag_transforms
    
    def fit(
        self,
        data: pd.DataFrame,
        target_col: str = 'y',
        date_col: str = 'ds',
        unique_id: str = 'series_1'
    ) -> 'MLModels':
        """
        Fit ML models.
        
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
        
        # Initialize MLForecast
        self.mlf = MLForecast(
            models=self.models,
            freq=self.freq,
            lags=self.lags,
            lag_transforms=self.lag_transforms,
            date_features=['dayofweek', 'month', 'year']
        )
        
        # Fit models
        self.mlf.fit(prepared)
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
        if self.mlf is None:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self.mlf.predict(h=h)
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
        if self.mlf is None:
            raise ValueError("Model must be fitted before cross-validation")
        
        cv_results = self.mlf.cross_validation(
            df=self.fitted_data,
            h=h,
            n_windows=n_windows,
            step_size=step_size
        )
        
        return cv_results