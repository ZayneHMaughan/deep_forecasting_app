from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from mlforecast.lag_transforms import ExpandingMean, ExpandingStd, RollingMean, RollingStd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from numba import njit
#import window_ops.expanding as expanding
#import window_ops.rolling as rolling
from data_config import prepare_data
from typing import Optional, Dict, Any, List, Literal, Union
import numpy as np
import pandas as pd

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


class MLModels:
    def __init__(
        self,
        models: Optional[Union[str, List[str]]] = None,
        clean_method: str = "impute",
        impute_strategy: str = "knn",
        freq: str = 'D',
        lags: Optional[List[int]] = None,
        lag_transforms: Optional[Dict] = None,
        forecast_strategy: Literal["one_step", "multi_step", "multi-output"] = "multi_step",
        **model_params
    ):
        """Initialize ML models using MLForecast."""
        self.data_prep = prepare_data(
            options=clean_method,
            imputed_options=impute_strategy
        )
        self.freq = freq
        self.lags = lags or [1, 7, 14]
        self.mlf = None
        self.fitted_data = None
        self.forecast_strategy = forecast_strategy

        # Normalize models argument to a list of names
        if models is None:
            model_names = ['random_forest']
        elif isinstance(models, str):
            model_names = [models]
        else:
            model_names = list(models)

        # Store both versions
        self.model_names = model_names
        
        # CRITICAL FIX: Convert list to string for model_name attribute
        self.model_name = '_'.join(model_names)
        
        self.model_params = model_params or {}
        self.models = self._initialize_models(model_names)
        
        # wrap in MultiOutput if necessary
        if forecast_strategy == "multi_output":
            self.models = {
                name: MultiOutputRegressor(model)
                for name, model in self.models.items()
            }
        
        # Default lag transforms
        if lag_transforms is None:
            self.lag_transforms = None
        else:
            self.lag_transforms = lag_transforms
        
        # ADD MISSING ATTRIBUTES
        self.date_features = ['dayofweek', 'month', 'year']
        self.target_transforms = None
        
        self.is_train = None

    def _initialize_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Return dict {name: estimator} for requested model names."""
        # default base estimators
        base = {
            'random_forest': RandomForestRegressor,
            'lgbm': LGBMRegressor,
            'xgboost': XGBRegressor,
            'catboost': CatBoostRegressor
        }

        models_out = {}
        for name in model_names:
            if name not in base:
                raise ValueError(f"Unknown model name '{name}'. Supported: {list(base.keys())}")
            params = self.model_params.get(name, {})
            # instantiate
            models_out[name] = base[name](**params)
        return models_out
    
    
        # Wrap models for multi-output if needed
    
    def fit(
        self,
        data: pd.DataFrame,
        target_col: str = 'y',
        date_col: str = 'ds',
        unique_id: str = 'series_1',
        h: int= 10
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
        prepared = self.data_prep.wrangle_data(
            data=cleaned,
            target_col=target_col,
            date_col=date_col,
            unique_id=unique_id
        )
        
            # Initialize MLForecast for one_step or multi_step
        self.mlf = MLForecast(
                models=list(self.models.values()),
                freq=self.freq,
                lags=self.lags,
                lag_transforms=self.lag_transforms,
                date_features=['dayofweek', 'month', 'year']
            )
        # Fit models
        self.mlf.fit(prepared)
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
        One-step ahead forecasting for multiple ML models.

        Iterative one-step forecast: at each t, train on all actuals up to t-1.
        """
        train_long = prepare_data.wrangle_data(train_df, target_col, date_col, unique_id)
        test_long = prepare_data.wrangle_data(test_df, target_col, date_col, unique_id)

        forecasts_list = []
        metrics_dict = {}

        for model_name, model_instance in self.models.items():
            preds = []
            actuals = test_long['y'].values
            dates = test_long['ds'].values

            for i in range(len(test_long)):
                # Ensure 2D DataFrame slice to avoid Series issues
                if i == 0:
                    current_train = train_long.copy()
                else:
                    current_train = pd.concat([
                        train_long,
                        test_long.iloc[:i, :].copy()  # <- fix here
                    ], ignore_index=True)

                # Fit MLForecast model
                mlf = MLForecast(
                    models=[model_instance],
                    freq=self.freq,
                    lags=self.lags,
                    lag_transforms=self.lag_transforms,
                    date_features=self.date_features,
                    target_transforms=self.target_transforms
                )
                mlf.fit(current_train)

                # Predict one step
                forecast = mlf.predict(h=1)
                pred_col = forecast.columns[-1]
                preds.append(forecast[pred_col].values[0])

            # Forecast DataFrame for this model
            forecast_df = pd.DataFrame({
                'unique_id': [unique_id] * len(preds),
                'ds': dates,
                'y_true': actuals,
                model_name: preds
            })
            forecasts_list.append(forecast_df)

            # Compute metrics
            errors = actuals - np.array(preds)
            metrics_dict[model_name] = {
                'MAE': float(np.mean(np.abs(errors))),
                'RMSE': float(np.sqrt(np.mean(errors ** 2))),
                'MAPE': float(np.mean(np.abs(errors / (actuals + 1e-10))) * 100)
            }

        # Merge all model forecasts into single DataFrame
        merged_forecasts = forecasts_list[0]
        for df in forecasts_list[1:]:
            cols = [c for c in df.columns if c not in ['unique_id', 'ds', 'y_true']]
            merged_forecasts = pd.concat([merged_forecasts, df[cols]], axis=1)

        return {'forecasts':merged_forecasts, 'metrics':metrics_dict}


        # Fix the multi_step_forecast method
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
        Multi-step recursive forecasting for multiple ML models.
        Uses MLForecast default h-step prediction behavior.
        """
        train_long = self.data_prep.wrangle_data(train_df, target_col, date_col, unique_id)

        forecasts_list = []
        metrics_dict = {}

        for model_name, model_instance in self.models.items():
            mlf = MLForecast(
                models={model_name: model_instance},  # Use dict with name
                freq=self.freq,
                lags=self.lags,
                lag_transforms=self.lag_transforms,
                date_features=self.date_features
            )
            mlf.fit(train_long)

            # Predict h steps ahead
            forecast = mlf.predict(h=horizon)
            
            # The prediction column will be named after the model
            pred_col = model_name  # MLForecast uses the model name
            if pred_col not in forecast.columns:
                # Fallback: find the prediction column
                pred_cols = [c for c in forecast.columns if c not in ['unique_id', 'ds']]
                if pred_cols:
                    pred_col = pred_cols[0]
            
            forecast_df = forecast.rename(columns={pred_col: model_name})
            forecasts_list.append(forecast_df)

            # Compute metrics if test_df provided
            if test_df is not None:
                test_long = self.data_prep.wrangle_data(test_df, target_col, date_col, unique_id)
                merged = forecast_df.merge(test_long[['ds', 'y']], on='ds', how='left')
                y_true = merged['y'].values
                y_pred = merged[model_name].values
                mask = ~pd.isna(y_true)
                if mask.sum() > 0:
                    errors = y_true[mask] - y_pred[mask]
                    metrics_dict[model_name] = {
                        'MAE': float(np.mean(np.abs(errors))),
                        'RMSE': float(np.sqrt(np.mean(errors ** 2))),
                        'MAPE': float(np.mean(np.abs(errors / (y_true[mask] + 1e-10))) * 100)
                    }

        # Merge all forecasts into a single DataFrame
        merged_forecasts = forecasts_list[0]
        for df in forecasts_list[1:]:
            cols = [c for c in df.columns if c not in ['unique_id', 'ds']]
            merged_forecasts = merged_forecasts.merge(
                df[['ds'] + cols], 
                on='ds', 
                how='outer'
            )

        return {'forecasts': merged_forecasts, 'metrics': metrics_dict}




    def predict(self, h: int = 10, X_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
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
            raise ValueError("Model must be fitted first")
        
        preds = self.mlf.predict(h=h, X_df=X_df)  # DataFrame with columns ['unique_id', 'ds', 'MLForecast_Single']
        return preds
        
    
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
        if self.mlf is None:
            raise ValueError("Model must be fitted before cross-validation")
        
        cv_results = self.mlf.cross_validation(
            df=self.fitted_data,
            h=h,
            n_windows=n_windows,
            step_size=step_size
        )
        
        return cv_results