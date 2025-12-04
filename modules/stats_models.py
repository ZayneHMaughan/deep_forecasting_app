"""Statistical forecasting models using StatsForecast"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
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
        """Initialize statistical models using StatsForecast."""
        self.data_prep = prepare_data(
            options=clean_method,
            imputed_options=impute_strategy
        )
        self.freq = freq
        self.season_length = season_length
        self.sf = None
        self.fitted_data = None
        
        # Initialize models
        if models is None:
            models = ['auto_arima']
        
        # Store as list for internal use
        self.model_list = models if isinstance(models, list) else [models]
        
        # CRITICAL FIX: Convert to string for external use
        self.model_name = '_'.join(self.model_list)
        
        self.models = self._initialize_models(self.model_list)
        
    def _initialize_models(self, model_names: List[str]):
        """Initialize StatsForecast models."""
        model_dict = {
            'auto_arima': AutoARIMA(season_length=self.season_length),
            'auto_ets': AutoETS(season_length=self.season_length),
            'arima': ARIMA(order=(1, 1, 1)),
            'seasonal_naive': SeasonalNaive(season_length=self.season_length),
            'random_walk_w_drift':RandomWalkWithDrift()

        }
        
        return {name:model_dict[name] for name in model_names if name in model_dict}
    
    def fit(
        self,
        data: pd.DataFrame,
        target_col: str = 'y',
        date_col: str = 'ds',
        unique_id: str = 'series_1', 
        **model_params
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
        prepared = self.data_prep.wrangle_data(
            data=cleaned,
            target_col=target_col,
            date_col=date_col,
            unique_id=unique_id
        )
        
        # Initialize and fit StatsForecast
        self.sf = StatsForecast(
            models=list(self.models.values()),
            freq=self.freq,
            n_jobs=-1
        )
        
        self.sf.fit(prepared)
        self.fitted_data = prepared
        
        #print(f"✓ {self.model_name} fitted successfully with {len(self.models)} models")
        return self
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
        Returns predictions and metrics.
        """
        # Prepare data
        train_data = prepare_data.wrangle_data(train_df, target_col, date_col, unique_id)
        test_data = prepare_data.wrangle_data(test_df, target_col, date_col, unique_id)

        preds = []
        actuals = test_data['y'].values
        dates = test_data['ds'].values

        # Use a single StatsForecast model if possible
        for i in range(len(test_data)):
            # Rolling origin: expand training with previous actuals
            current_train = pd.concat([train_data, test_data.iloc[:i]], ignore_index=True)

            # Create model
            model = self._create_model(alias="Model")
            sf = StatsForecast(models=[model], freq=self.freq)
            
            # Forecast 1 step ahead
            forecast = sf.forecast(df=current_train, h=1)

            # Extract prediction column safely
            pred_col = [c for c in forecast.columns if c not in ['unique_id', 'ds']][0]
            preds.append(forecast[pred_col].values[0])

        # Build forecast DataFrame
        forecast_df = pd.DataFrame({
            'unique_id': [unique_id] * len(preds),
            'ds': dates,
            'y_true': actuals,
            'y_pred': preds
        })

        # Compute metrics
        errors = actuals - np.array(preds)
        metrics = {
            'MAE': float(np.mean(np.abs(errors))),
            'RMSE': float(np.sqrt(np.mean(errors ** 2))),
            'MAPE': float(np.mean(np.abs(errors / actuals)) * 100) if np.all(actuals != 0) else np.nan
        }

        return {'forecasts':forecast_df, 'metrics':metrics}

    def predict_multi_step(
    self,
    h: int = 10,
    test_df: Optional[pd.DataFrame] = None,
    target_col: str = 'y',
    date_col: str = 'ds',
    unique_id: str = 'series_1'
) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
        """
        Multi-step forecast using StatsForecast with recursive predictions.
        Returns both predictions and optional metrics.
        """
        if self.sf is None:
            raise ValueError("Model must be fitted before prediction")

        # Prepare training data
        train_long = self.fitted_data

        # Use StatsForecast recursive forecast
        forecast_df = self.sf.forecast(df=train_long, h=h)
        
        # DEBUG: Print what columns we actually have
        print(f"DEBUG: Forecast columns: {forecast_df.columns.tolist()}")
        print(f"DEBUG: Forecast shape: {forecast_df.shape}")
        print(f"DEBUG: First few rows:\n{forecast_df.head()}")

        # Get prediction columns - be more flexible with column names
        # StatsForecast might use different naming conventions
        exclude_cols = ['unique_id', 'ds', 'cutoff', 'y']
        model_cols = [c for c in forecast_df.columns if c not in exclude_cols]
        
        print(f"DEBUG: Model columns found: {model_cols}")
        
        if len(model_cols) == 0:
            print(f"ERROR: All columns: {forecast_df.columns.tolist()}")
            print(f"ERROR: Excluded: {exclude_cols}")
            raise ValueError(f"No prediction columns found in forecast. Available columns: {forecast_df.columns.tolist()}")
        
        # Keep original model names
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
                mask = ~pd.isna(actuals)
                
                if mask.sum() > 0:
                    errors = actuals[mask] - preds[mask]
                    metrics[model_col] = {
                        'MAE': float(np.mean(np.abs(errors))),
                        'RMSE': float(np.sqrt(np.mean(errors ** 2))),
                        'MAPE': float(np.mean(np.abs(errors / (actuals[mask] + 1e-10))) * 100) if np.all(actuals[mask] != 0) else np.nan
                    }

        return {'forecasts': forecast_df, 'metrics': metrics if metrics else None}

    
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
        if h == 1 :
            return self.one_step_forecast(h=h)
        elif h > 1 :
            return self.predict_multi_step(h=h)
        
    
    def get_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = None) -> Dict[str, float]:
        """Calculate model metrics."""
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        return {
            'MODEL': model_name or self.model_name,
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