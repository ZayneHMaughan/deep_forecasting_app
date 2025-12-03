"""
Create a One Stop Shop for everything related to modeling
"""
import streamlit as st
from typing import Literal, Optional, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    from data_config import prepare_data
    PREPARE_DATA = True
except ImportError as e: 
    PREPARE_DATA = False
    print(f"Data Configuration not available : {e}")
    
try: 
    from stats_forecast import StatForecaster
    STATFORECAST_AVAILABLE = True
except ImportError as e: 
    STATFORECAST_AVAILABLE = False
    st.error(f"StatsForecasting module not available : {e}")

try: 
    from ml_forecast import MLForecaster
    MLFORECAST_AVAILABLE = True
except ImportError as e: 
    MLFORECAST_AVAILABLE = False
    st.error(f"MLForecasting module not available : {e}")


try: 
    from deep_forecast import DeepForecaster
    DEEPFORECAST_AVAILABLE = True
except ImportError as e: 
    DEEPFORECAST_AVAILABLE = False
    st.error(f"deepForecasting module not available : {e}")

class ForecasterModel:
    def __init__(
            self
            ):
        self.models = []
        self.metrics = []
        self.preds = {}

    def add_models(self, model, name: Optional[str]):
        model_name = name or getattr(model, 'model_name')
        self.models.append({'name':model_name, 'model':model})
    
    def evaluate_all(
            self,
            X_test: pd.DataFrame, 
            y_test: np.ndarray,
            model_specific_params: Optional[Dict] = None
    ) -> pd.DataFrame:
        for model_info in self.models:
            name = model_info['name']
            model = model_info['model']

            try:
                if name in model_specific_params:
                    y_pred = model.predict(**model_specific_params[name])
                else: 
                    y_pred = model.predict(X_test)

                self.preds[name] = y_pred

                metrics = model.get_metrics(y_test, y_pred)
                self.metrics.append(metrics)  

            except Exception as e:
                print(f"Error Evaluating {name}: {e}")

    def get_best_model(self, metric: str = 'RMSE') -> Dict[str, Any]:
        """Get the best performing model."""
        if not self.metrics:
            raise ValueError("No models have been evaluated yet")
        
        metrics_df = pd.DataFrame(self.metrics)
        best_idx = metrics_df[metric].idxmin()
        best_model_name = metrics_df.loc[best_idx, 'model']
        
        return {
            'model_name': best_model_name,
            'metrics': metrics_df.loc[best_idx].to_dict()
        }
    
    def plot_comparison(
        self,
        y_test: np.ndarray,
        save_dir: Optional[str] = None
    ):
        """
        Create comprehensive comparison plots.
        
        Parameters
        ----------
        y_test : np.ndarray
            True test values
        save_dir : str, optional
            Directory to save plots
        """
        metrics_df = pd.DataFrame(self.metrics)
        
        # Plot 1: Predictions comparison
        self.graph_utils.plot_predictions(
            y_test,
            self.predictions,
            title="Model Predictions Comparison",
            save_path=f"{save_dir}/predictions_comparison.png" if save_dir else None
        )
        
        # Plot 2: Metrics comparison
        for metric in ['RMSE', 'MAE', 'MAPE']:
            if metric in metrics_df.columns:
                self.graph_utils.plot_metrics_comparison(
                    metrics_df,
                    metric=metric,
                    save_path=f"{save_dir}/{metric.lower()}_comparison.png" if save_dir else None
                )
        
        plt.show()
