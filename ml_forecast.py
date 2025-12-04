"""
Similar to Stat Forecaster but this time with more focus on Machine Learning Methods

"""


from typing import Dict, List, Literal, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Try to import optional ML libraries
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
try:
    from data_config import prepare_data
    PREPARE_DATA = True
except ImportError as e: 
    PREPARE_DATA = False
    print(f"Data Configuration not available : {e}")

class MLForecaster: 
    def __init__(
        self, 
        model_type: Literal['xgboost', 'lightgbm', 'random_forest', 'catboost', 'linear'],
        freq: str = 'MS',
        lags: Optional[List[int]] = None,
        lag_transforms: Optional[Dict] = None,
        date_features: Optional[List[str]] = None,
        target_transforms: Optional[List] = None,
        **model_params
        ):
        self.model_type = model_type.lower()
        self.freq = freq
        self.lags = lags if lags is not None else [1,12]
        self.lag_transforms = lag_transforms
        self.date_features = date_features if date_features is not None else []
        self.target_transforms = target_transforms if target_transforms is not None else []
        self.model_params = model_params
        self.mlf = None
        self._validate_model_type()

    def _validate_model_type(self)-> None:
        if self.model_type == 'xgboost' and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
        elif self.model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")
        elif self.model_type == 'catboost' and not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed. Install with: pip install catboost")
        
    def _create_model(self):
        if self.model_type == 'xgboost':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(self.model_params)
            return XGBRegressor(**default_params)

        elif self.model_type == 'lightgbm':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            default_params.update(self.model_params)
            return LGBMRegressor(**default_params)

        elif self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(self.model_params)
            return RandomForestRegressor(**default_params)

        elif self.model_type == 'catboost':
            default_params = {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbose': 0
            }
            default_params.update(self.model_params)
            return CatBoostRegressor(**default_params)

        elif self.model_type == 'linear':
            return LinearRegression(**self.model_params)

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def one_step_forecast(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str = 'y',
        date_col: str = 'ds',
        unique_id: str = 'series_1'
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
        """
        One-step ahead forecasting (h=1 iterative).

        For each timestamp in test set:
        1. Train on actuals up to t-1
        2. Predict ŷ(t)
        3. Prediction does NOT feed into next step
        4. Proceed iteratively

        This is "optimistic backtesting" - most accurate but not real deployment.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data
        test_df : pd.DataFrame
            Test data
        target_col : str
            Name of target column
        date_col : str
            Name of date column
        unique_id : str
            Series identifier

        Returns
        -------
        dict
            Dictionary with keys:
            - 'forecasts': DataFrame with predictions
            - 'metrics': Dictionary with MAE, MAPE, RMSE
        """
        # Prepare data
        train_long = prepare_data.wrangle_data(train_df, target_col, date_col, unique_id)
        test_long = prepare_data.wrangle_data(test_df, target_col, date_col, unique_id)

        predictions = []
        actuals = test_long['y'].values
        dates = test_long['ds'].values

        # Iterative one-step forecasting
        for i in range(len(test_long)):
            # Use all actual data up to current point
            if i == 0:
                current_train = train_long.copy()
            else:
                current_train = pd.concat([
                    train_long,
                    test_long.iloc[:i]
                ], ignore_index=True)

            # Create MLForecast model
            model = self._create_model()
            mlf = MLForecast(
                models=[model],
                freq=self.freq,
                lags=self.lags,
                lag_transforms=self.lag_transforms,
                date_features=self.date_features,
                target_transforms=self.target_transforms
            )

            # Fit and predict h=1
            mlf.fit(df=current_train)
            forecast = mlf.predict(h=1)

            # Extract prediction (column name depends on model type)
            pred_col = forecast.columns[-1]  # Last column is the prediction
            predictions.append(forecast[pred_col].values[0])

        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'unique_id': [unique_id] * len(predictions),
            'ds': dates,
            'y_true': actuals,
            'y_pred': predictions
        })

        # Calculate metrics
        errors = actuals - np.array(predictions)
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
        This is the DEFAULT MLForecast behavior when you call predict(h).

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

        # Create MLForecast model
        model = self._create_model()
        mlf = MLForecast(
            models=[model],
            freq=self.freq,
            lags=self.lags,
            lag_transforms=self.lag_transforms,
            date_features=self.date_features,
            target_transforms=self.target_transforms
        )

        # Fit model
        mlf.fit(df=train_long)

        # Recursive forecasting (default behavior)
        forecast = mlf.predict(h=horizon)

        # Extract predictions
        pred_col = forecast.columns[-1]
        forecast_df = forecast.copy()
        forecast_df = forecast_df.rename(columns={pred_col: 'y_pred'})

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
        unique_id: str = 'series_1',
        test_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
        """
        Multi-output direct forecasting (one model per horizon).

        Trains H separate models, each specialized to predict t+k horizon directly.
        This is implemented via MLForecast's max_horizon parameter.

        Advantages:
        - No error accumulation (each step predicted independently)
        - Better accuracy for longer horizons

        Disadvantages:
        - More training time (H models vs 1)
        - Requires more data

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

        # Create MLForecast model
        model = self._create_model()
        mlf = MLForecast(
            models=[model],
            freq=self.freq,
            lags=self.lags,
            lag_transforms=self.lag_transforms,
            date_features=self.date_features,
            target_transforms=self.target_transforms
        )

        # Fit with max_horizon to train one model per step
        mlf.fit(df=train_long, max_horizon=horizon)

        # Predict using direct strategy
        forecast = mlf.predict(h=horizon)

        # Extract predictions
        pred_col = forecast.columns[-1]
        forecast_df = forecast.copy()
        forecast_df = forecast_df.rename(columns={pred_col: 'y_pred'})

        # Calculate metrics if test data provided
        metrics = None
        if test_df is not None:
            test_long = self._prepare_data(test_df, target_col, date_col, unique_id)

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


def train_test_split_ts(
    data: pd.DataFrame,
    test_size: Union[int, float] = 0.2,
    target_col: str = 'y'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train and test sets.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    test_size : int or float
        If int, number of observations for test set
        If float, proportion of data for test set
    target_col : str
        Name of target column

    Returns
    -------
    tuple
        (train_df, test_df)
    """
    if isinstance(test_size, float):
        test_size = int(len(data) * test_size)

    train_df = data.iloc[:-test_size].copy()
    test_df = data.iloc[-test_size:].copy()

    return train_df, test_df


def evaluate_forecasts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate forecast evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    dict
        Dictionary with MAE, MAPE, RMSE
    """
    errors = y_true - y_pred

    metrics = {
        'mae': float(np.mean(np.abs(errors))),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mape': float(np.mean(np.abs(errors / y_true)) * 100) if np.all(y_true != 0) else np.nan
    }

    return metrics
