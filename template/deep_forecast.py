"""
Like Stats and ML but this time focused on Deep learning models
"""

from typing import Dict, List, Literal, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import (
    MLP,
    RNN,
    LSTM,
    GRU
)
from neuralforecast.losses.pytorch import MSE, MAE
import warnings
warnings.filterwarnings('ignore')

try:
    from data_config import prepare_data
    PREPARE_DATA = True
except ImportError as e: 
    PREPARE_DATA = False
    print(f"Data Configuration not available : {e}")

class DeepForecaster:
    """
    Production-ready forecaster using NeuralForecast neural network models.

    Supports MLP, RNN, LSTM, GRU, NBEATS, NHITS, TCN with three forecasting modes:
    - one_step: One-step ahead forecasting (h=1 iterative, refit for each step)
    - multi_step: Multi-step recursive forecasting (RNN/LSTM/GRU with recurrent=True)
    - multi_output: Multi-output direct forecasting (default for NBEATS, NHITS, MLP, TCN)

    Parameters
    ----------
    model_type : str
        Type of model to use. Options: 'mlp', 'rnn', 'lstm', 'gru',
    freq : str
        Frequency of the time series (e.g., 'MS', 'D', 'H')
    input_size : int
        Number of lagged observations to use as input window
    horizon : int
        Forecast horizon (h parameter)
    **model_params : dict
        Additional model-specific parameters (hidden_size, num_layers, etc.)
    """

    def __init__(
        self,
        model_type: Literal['mlp', 'rnn', 'lstm', 'gru'],
        freq: str = 'MS',
        input_size: int = 12,
        horizon: int = 1,
        **model_params
    ):
        self.model_type = model_type.lower()
        self.freq = freq
        self.input_size = input_size
        self.horizon = horizon
        self.model_params = model_params
        self.nf = None

    def _create_model(self, h: int, recurrent: bool = False):
        """
        Create the NeuralForecast model instance.

        Parameters
        ----------
        h : int
            Forecast horizon
        recurrent : bool
            Whether to use recursive forecasting (only for RNN/LSTM/GRU)

        Returns
        -------
        NeuralForecast model instance
        """
        # Common parameters
        common_params = {
            'h': h,
            'input_size': self.input_size,
            'scaler_type': self.model_params.get('scaler_type', 'robust'),
            'max_steps': self.model_params.get('max_steps', 100),
            'learning_rate': self.model_params.get('learning_rate', 0.001),
            'batch_size': self.model_params.get('batch_size', 32),
            'random_seed': self.model_params.get('random_seed', 42),
        }

        # Loss function
        loss_type = self.model_params.get('loss', 'mae')
        loss = MAE() if loss_type == 'mae' else MSE()
        common_params['loss'] = loss

        if self.model_type == 'mlp':
            # Multi-Layer Perceptron
            return MLP(
                **common_params,
                num_layers=self.model_params.get('num_layers', 2),
                hidden_size=self.model_params.get('hidden_size', 32),
            )

        elif self.model_type == 'rnn':
            # Simple RNN
            rnn_params = common_params.copy()
            if recurrent and h > 1:
                # Only RNN/LSTM/GRU support recurrent parameter
                rnn_params['recurrent'] = True

            return RNN(
                **rnn_params,
                encoder_n_layers=self.model_params.get('encoder_n_layers', 1),
                encoder_hidden_size=self.model_params.get('encoder_hidden_size', 16),
                decoder_hidden_size=self.model_params.get('decoder_hidden_size', 16),
            )

        elif self.model_type == 'lstm':
            # LSTM
            lstm_params = common_params.copy()
            if recurrent and h > 1:
                lstm_params['recurrent'] = True

            return LSTM(
                **lstm_params,
                encoder_n_layers=self.model_params.get('encoder_n_layers', 1),
                encoder_hidden_size=self.model_params.get('encoder_hidden_size', 16),
                decoder_hidden_size=self.model_params.get('decoder_hidden_size', 16),
            )

        elif self.model_type == 'gru':
            # GRU
            gru_params = common_params.copy()
            if recurrent and h > 1:
                gru_params['recurrent'] = True

            return GRU(
                **gru_params,
                encoder_n_layers=self.model_params.get('encoder_n_layers', 1),
                encoder_hidden_size=self.model_params.get('encoder_hidden_size', 16),
                decoder_hidden_size=self.model_params.get('decoder_hidden_size', 16),
            )


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
        One-step ahead forecasting (h=1 iterative, refit for each step).

        For each timestamp in test set:
        1. Train on actuals up to t-1
        2. Predict ŷ(t) with h=1
        3. Prediction does NOT feed into next step
        4. Proceed iteratively

        This is "optimistic backtesting" - most accurate but not real deployment.

        NOTE: This refits the model for each prediction, which is computationally
        expensive but provides the most accurate one-step forecasts.

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

            # Create model with h=1
            model = self._create_model(h=1, recurrent=False)
            nf = NeuralForecast(models=[model], freq=self.freq)

            # Fit and predict
            nf.fit(df=current_train)
            forecast = nf.predict(df=current_train)

            # Extract prediction
            pred_col = forecast.columns[-1]  # Last column is the model prediction
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
        test_df: Optional[pd.DataFrame] = None,
        use_recurrent: bool = False
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
        """
        Multi-step recursive forecasting.

        For RNN/LSTM/GRU models with recurrent=True:
        1. Predict step 1
        2. Use prediction as input
        3. Predict step 2
        4. Repeat until horizon h is reached

        For other models (NBEATS, NHITS, MLP, TCN):
        - These models don't support recurrent forecasting
        - They will use multi-output direct forecasting instead
        - Set use_recurrent=False for these models

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
        use_recurrent : bool
            Whether to use recursive forecasting (only for RNN/LSTM/GRU)

        Returns
        -------
        dict
            Dictionary with keys:
            - 'forecasts': DataFrame with predictions
            - 'metrics': Dictionary with MAE, MAPE, RMSE (if test_df provided)
        """
        # Validate recurrent parameter
        if use_recurrent and self.model_type not in ['rnn', 'lstm', 'gru']:
            warnings.warn(
                f"recurrent=True is only supported for RNN/LSTM/GRU models. "
                f"Model type '{self.model_type}' will use multi-output direct forecasting instead."
            )
            use_recurrent = False

        # Prepare data
        train_long = prepare_data.wrangle_data(train_df, target_col, date_col, unique_id)

        # Create model with specified horizon
        model = self._create_model(h=horizon, recurrent=use_recurrent)
        nf = NeuralForecast(models=[model], freq=self.freq)

        # Fit model
        nf.fit(df=train_long)

        # Predict
        forecast = nf.predict(df=train_long)

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
        Multi-output direct forecasting (default for most NeuralForecast models).

        Direct forecast models produce all steps in the forecast horizon at once.
        This is the DEFAULT approach for most NeuralForecast models.

        How it works:
        1. Takes an entire window of past values (input_size)
        2. Computes all forecast timepoint values in a single forward pass
        3. The neural network architecture outputs h values simultaneously

        Advantages:
        - Better accuracy than recursive (less error propagation)
        - All predictions made simultaneously
        - Default behavior for NBEATS, NHITS, MLP, TCN

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

        # Create model with specified horizon (recurrent=False for multi-output)
        model = self._create_model(h=horizon, recurrent=False)
        nf = NeuralForecast(models=[model], freq=self.freq)

        # Fit model
        nf.fit(df=train_long)

        # Multi-output prediction
        forecast = nf.predict(df=train_long)

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