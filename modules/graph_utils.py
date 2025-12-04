import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Optional
import pandas as pd


class GraphUtils:
    def __init__(self, style: str = 'seaborn-v0_8', figsize: tuple = (12, 6)):
        """Initialize graph utilities."""
        plt.style.use(style)
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 10)
        
    def plot_predictions(
        self,
        y_true: np.ndarray,
        predictions_dict: Dict[str, np.ndarray],
        title: str = "Model Predictions Comparison",
        save_path: Optional[str] = None
    ):
        """Plot predictions from multiple models."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot true values
        ax.plot(y_true, label='True Values', linewidth=2, color='black', marker='o')
        
        # Plot predictions
        for i, (model_name, preds) in enumerate(predictions_dict.items()):
            ax.plot(preds, label=model_name, linewidth=2, 
                   color=self.colors[i], marker='s', alpha=0.7)
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_strategy_comparison(
        self,
        metrics_df: pd.DataFrame,
        metric: str = 'RMSE',
        save_path: Optional[str] = None
    ):
        """Compare strategies across models."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Group by strategy
        strategies = metrics_df['strategy'].unique()
        x = np.arange(len(metrics_df))
        width = 0.25
        
        for i, strategy in enumerate(strategies):
            strategy_data = metrics_df[metrics_df['strategy'] == strategy]
            ax.bar(x[i::len(strategies)] + i*width, strategy_data[metric], 
                  width, label=strategy, color=self.colors[i])
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} by Forecasting Strategy', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics_df['model'].unique(), rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_metrics_comparison(
        self,
        metrics_df: pd.DataFrame,
        metric: str = 'RMSE',
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """Plot metric comparison across models."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Sort by metric
        sorted_df = metrics_df.sort_values(by=metric)
        
        # Create bar plot
        bars = ax.barh(sorted_df['model'], sorted_df[metric], color=self.colors[:len(sorted_df)])
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2f}', ha='left', va='center', fontsize=10)
        
        ax.set_xlabel(metric, fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        ax.set_title(title or f'{metric} Comparison Across Models', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ):
        """Plot residuals analysis."""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Residuals over time
        axes[0].plot(residuals, color=self.colors[0], marker='o')
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'{model_name} - Residuals Over Time')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[1].hist(residuals, bins=30, color=self.colors[1], edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residuals Distribution')
        axes[1].grid(True, alpha=0.3)
        
        # Predicted vs Actual
        axes[2].scatter(y_pred, y_true, color=self.colors[2], alpha=0.6)
        axes[2].plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 
                    'r--', linewidth=2)
        axes[2].set_xlabel('Predicted')
        axes[2].set_ylabel('Actual')
        axes[2].set_title('Predicted vs Actual')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, axes
