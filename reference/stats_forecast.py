"""
Docstring for deep_forecasting_app.stats_forecast

This module provides forecasting strategies across statisical models

Models: ARIMA, AutoArima, AutoETS, Naive, SeasonalNaive, RandomWalkWithDrift
"""

from typing import Dict, List, Literal, Optional, Tuple, Union
import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import (
    ARIMA, 
    AutoARIMA,
    AutoETS,
    Naive, 
    SeasonalNaive, 
    RandomWalkWithDrift
)

from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mape, rmse
from functools import partial 

try:
    from data_config import prepare_data
    PREPARE_DATA = True
except ImportError as e: 
    PREPARE_DATA = False
    print(f"Data Configuration not available : {e}")

class StatForecaster:
    """
   Forecaster using StatsForecast models
    """
    def __init__(
            self, 
            model_type: Literal['arima', 'auto_arima', 'auto_ets', 'naive', 'seasonal_naive', 'rw_drift'],
            freq: str = 'MS',
            season_length: int = 12, 
            clean_method: Literal["impute", "fill", "drop"] = "fill", 
            imputed_options: Literal["simple", "knn", "iterative"] = "simple",
            **model_params
            ):
        self.model_type = model_type.lower()
        self.freq = freq
        self.season_length = season_length
        self.model_params = model_params
        self.sf_model = None
        self.data_prep = prepare_data(options=clean_method, imputed_options=imputed_options)
        self._validate_model_type()

    def _validate_model_type(self) -> None:
        """
        Ensure the model type is supported
        """
        valid_models = ['arima', 'auto_arima', 'auto_ets', 'naive', 'seasonal_naive', 'rw_drift']
        if self.model_type not in valid_models:
            raise ValueError(
                f"model_type must be one of {valid_models}, got '{self.model_type}'"
            )
        
    def _create_model(self, alias: str = 'Model'):
        if self.model_type == "arima":
            order = self.model_params.get('order', (1,1,1))
            seasonal_order = self.model_params('seasonal_order', (0,0,0))
            include_mean = self.model_params.get('include_mean', True)

            return ARIMA(
                order = order, 
                seasonal_length = self.season_length,
                seasonal_order = seasonal_order,
                include_mean = include_mean,
                alias = alias 
            )
        elif self.model_type == 'auto_arima':
            return AutoARIMA(
                season_length=self.season_length,
                seasonal=self.model_params.get('seasonal', True),
                d=self.model_params.get('d', None),
                D=self.model_params.get('D', None),
                max_p=self.model_params.get('max_p', 5),
                max_q=self.model_params.get('max_q', 5),
                max_P=self.model_params.get('max_P', 2),
                max_Q=self.model_params.get('max_Q', 2),
                stepwise=self.model_params.get('stepwise', True),
                alias=alias
            )

        elif self.model_type == 'auto_ets':
            # AutoETS for exponential smoothing
            model = self.model_params.get('model', 'ZZZ')
            return AutoETS(
                model=model,
                season_length=self.season_length,
                alias=alias
            )

        elif self.model_type == 'naive':
            return Naive(alias=alias)

        elif self.model_type == 'seasonal_naive':
            return SeasonalNaive(
                season_length=self.season_length,
                alias=alias
            )

        elif self.model_type == 'rw_drift':
            return RandomWalkWithDrift(alias=alias)
        

    # def fit(self, 
    #         data: pd.DataFrame, 
    #         target_col: str = 'y',
    #         date_col : str = 'ds',
    #         unique_id: str = 'series_1') -> 'StatForecaster':
    #     cleaned = self.data_prep.clean_miss_data(data)
        
    #     # Prepare data in Nixtla format
    #     prepared = self.data_prep.prepare_data(
    #         data=cleaned,
    #         target_col=target_col,
    #         date_col=date_col,
    #         unique_id=unique_id
    #     )
        
    #     # Initialize and fit StatsForecast
    #     self.sf = StatsForecast(
    #         models=self.models,
    #         freq=self.freq,
    #         n_jobs=-1
    #     )
        
    #     self.sf.fit(prepared)
    #     self.fitted_data = prepared
        
    #     return self
        

    def one_step_forecast(
            self, 
            train_df: pd.DataFrame,
            test_df: pd.DataFrame,
            target_col: str = 'y',
            date_col: str = 'ds',
            unique_id: str = 'series_1'
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
        """
       One Step ahead forecasting
        """
        train_data = prepare_data.wrangle_data(data =train_df, target_col=target_col,date_col=date_col, unique_id=unique_id)
        test_data = prepare_data.wrangle_data(data = test_df, target_col=target_col,date_col=date_col, unique_id=unique_id)

        preds = []
        actuals = test_data['y'].values
        dates = test_data['ds'].values

        for i in range(len(test_data)):
            if i ==0: 
                current_train = train_data.copy()
            else:
                current_train = pd.concat([train_data, test_data.iloc[:i]], ignore_index=True)
            
            model = self._create_model(alias="Model")
            sf = StatsForecast(models = [model], freq=self.freq)

            forecast = sf.forecast(df=current_train, h = 1)
            preds.append(forecast['Model'].values[0])
        
        forecast_df = pd.DataFrame({
            'unique_id' : [unique_id] * len(preds),
            'ds': dates,
            'y_true': actuals,
            'y_pred' :preds
        })

        errors = actuals - np.array(preds)
        metrics = {
            'mae': float(np.mean(np.abs(errors))),
            'rmse': float(np.sqrt(np.mean(errors ** 2))),
            'mape': float(np.mean(np.abs(errors / actuals)) * 100) if np.all(actuals != 0) else np.nan
        }

        return {
            'forecasts': forecast_df,
            'metrics': metrics
        }

    def multi_step_forecast(
        self,
        train_df: pd.DataFrame,
        horizon: int,
        target_col: str = 'y',
        date_col: str = 'ds',
        unique_id: str = 'series_1',
        test_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
        """
        Multi-step recursive forecasting.

        For each timestamp t:
        1. Predict using model
        2. Use previous predictions as input
        3. Iterate through entire horizon

        This simulates real deployment where future actuals are unknown.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data
        horizon : int
            Forecast horizon
        target_col : str
            Name of target column
        date_col : str
            Name of date column
        unique_id : str
            Series identifier
        test_df : pd.DataFrame, optional
            Test data for metric calculation

        Returns
        -------
        dict
            Dictionary with keys:
            - 'forecasts': DataFrame with predictions
            - 'metrics': Dictionary with MAE, MAPE, RMSE (if test_df provided)
        """
        # Prepare data
        train_long = prepare_data.wrangle_data(train_df, target_col, date_col, unique_id)

        # Create model and forecast
        model = self._create_model(alias='Model')
        sf = StatsForecast(models=[model], freq=self.freq)

        # This is the default StatsForecast behavior - recursive forecasting
        forecast = sf.forecast(df=train_long, h=horizon)

        forecast_df = forecast.copy()
        forecast_df = forecast_df.rename(columns={'Model': 'y_pred'})

        # Calculate metrics if test data provided
        metrics = None
        if test_df is not None:
            test_long = prepare_data.wrangle_data(test_df, target_col, date_col, unique_id)

            # Merge with actuals
            forecast_df = forecast_df.merge(
                test_long[['ds', 'y']],
                on='ds',
                how='left'
            )
            forecast_df = forecast_df.rename(columns={'y': 'y_true'})

            # Calculate metrics
            actuals = forecast_df['y_true'].values
            predictions = forecast_df['y_pred'].values

            # Only calculate where we have actuals
            mask = ~pd.isna(actuals)
            if mask.sum() > 0:
                errors = actuals[mask] - predictions[mask]
                metrics = {
                    'mae': float(np.mean(np.abs(errors))),
                    'rmse': float(np.sqrt(np.mean(errors ** 2))),
                    'mape': float(np.mean(np.abs(errors / actuals[mask])) * 100) if np.all(actuals[mask] != 0) else np.nan
                }

        return {
            'forecasts': forecast_df,
            'metrics': metrics
        }

    def multi_output_forecast(
        self,
        train_df: pd.DataFrame,
        horizon: int,
        target_col: str = 'y',
        date_col: str = 'ds',
        unique_id: str = 'series_1'
    ) -> None:
        """
        Multi-output forecasting - NOT SUPPORTED for statistical models.

        ARIMA/ETS models do NOT support multi-output forecasting.
        They can only do recursive forecasting.

        This method raises NotImplementedError to handle gracefully.

        Raises
        ------
        NotImplementedError
            Always raised - statistical models don't support multi-output
        """
        raise NotImplementedError(
            f"Multi-output forecasting is NOT supported for statistical models "
            f"like {self.model_type.upper()}. Statistical models (ARIMA/ETS) can only "
            f"perform recursive forecasting. Use multi_step_forecast() instead, or "
            f"switch to ML models (mlforecast) or neural models (neuralforecast) "
            f"which support true multi-output forecasting."
        )


