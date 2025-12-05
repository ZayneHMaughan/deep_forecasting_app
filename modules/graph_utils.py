import plotly.graph_objects as go
import pandas as pd
import numpy as np

class GraphUtils:
    def __init__(self, colors=None):
        self.colors = colors or [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf"
        ]

    def plot_predictions(self, y_test, predictions: dict, title="Predictions vs Actual"):
        # Handle y_test (actual values)
        if isinstance(y_test, pd.Series):
            if isinstance(y_test.index, pd.DatetimeIndex):
                y_df = pd.DataFrame({'ds': y_test.index, 'y': y_test.values})
            else:
                y_df = pd.DataFrame({'ds': range(len(y_test)), 'y': y_test.values})
        elif isinstance(y_test, pd.DataFrame):
            y_df = y_test.copy()
            if 'ds' not in y_df.columns:
                # Check if index is datetime
                if isinstance(y_df.index, pd.DatetimeIndex):
                    y_df['ds'] = y_df.index
                else:
                    y_df['ds'] = pd.RangeIndex(len(y_df))
            if 'y' not in y_df.columns:
                # Find the target column (usually 'y' or first numeric column)
                numeric_cols = y_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    y_df['y'] = y_df[numeric_cols[0]]
                else:
                    y_df['y'] = y_df.iloc[:, 0]
        else:
            # Handle numpy arrays or other array-like objects
            y_array = np.array(y_test)
            y_df = pd.DataFrame({
                'ds': range(len(y_array)),
                'y': y_array
            })

        # Ensure ds is datetime if it looks like datetime
        if 'ds' in y_df.columns and y_df['ds'].dtype == 'object':
            try:
                y_df['ds'] = pd.to_datetime(y_df['ds'])
            except:
                pass

        fig = go.Figure()

        # Plot actual values
        fig.add_trace(go.Scatter(
            x=y_df['ds'],
            y=y_df['y'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='black', width=2),
            marker=dict(size=4)
        ))

        # Plot predictions for each model
        for i, (model_name, preds) in enumerate(predictions.items()):
            if isinstance(preds, pd.DataFrame):
                # Ensure ds column exists and is datetime
                if 'ds' not in preds.columns:
                    continue

                if preds['ds'].dtype == 'object':
                    try:
                        preds['ds'] = pd.to_datetime(preds['ds'])
                    except:
                        pass

                # Find prediction columns (exclude metadata columns)
                pred_cols = [c for c in preds.columns if c not in ['ds', 'unique_id', 'y', 'y_true']]
                if not pred_cols:
                    continue

                pred_col = pred_cols[0]
                x_vals = preds['ds']
                y_vals = preds[pred_col]
            else:
                # If predictions don't have ds column, try to generate future dates
                pred_array = np.array(preds)
                last_ds = y_df['ds'].iloc[-1]

                if isinstance(last_ds, pd.Timestamp):
                    # Infer frequency and generate future dates
                    if len(y_df) > 1:
                        freq = pd.infer_freq(y_df['ds'])
                        if freq:
                            x_vals = pd.date_range(start=last_ds, periods=len(pred_array) + 1, freq=freq)[1:]
                        else:
                            # Fallback: calculate mean difference
                            time_diff = (y_df['ds'].iloc[-1] - y_df['ds'].iloc[-2])
                            x_vals = [last_ds + time_diff * (j + 1) for j in range(len(pred_array))]
                    else:
                        x_vals = pd.date_range(start=last_ds, periods=len(pred_array) + 1, freq='D')[1:]
                else:
                    # Integer indices - just continue the sequence
                    x_vals = range(int(last_ds) + 1, int(last_ds) + len(pred_array) + 1)

                y_vals = pred_array

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                name=model_name,
                line=dict(color=self.colors[i % len(self.colors)], width=2),
                marker=dict(size=4)
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_white",
            legend_title="Model",
            hovermode='x unified'
        )

        return fig

    def plot_metrics_comparison(self, metrics_df: pd.DataFrame, metric='RMSE', title=None):
        if metric not in metrics_df.columns:
            raise ValueError(f"Metric {metric} not found in DataFrame")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=metrics_df['MODEL'],
            y=metrics_df[metric],
            marker_color=self.colors[:len(metrics_df)]
        ))
        fig.update_layout(
            title=title or f"{metric} Comparison Across Models",
            xaxis_title="Model",
            yaxis_title=metric,
            template="plotly_white"
        )
        return fig

    def plot_residuals(self, y_true, y_pred, model_name="Model"):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        min_len = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:min_len], y_pred[:min_len]
        residuals = y_true - y_pred

        # Residuals over time
        fig_residuals = go.Figure()
        fig_residuals.add_trace(go.Scatter(
            x=np.arange(len(residuals)), y=residuals,
            mode='lines+markers',
            name='Residuals',
            line=dict(color=self.colors[0], width=2)
        ))
        fig_residuals.add_trace(go.Scatter(
            x=[0, len(residuals)-1], y=[0,0],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Zero'
        ))
        fig_residuals.update_layout(
            title=f"{model_name} Residuals Over Time",
            xaxis_title="Time Step",
            yaxis_title="Residual",
            template="plotly_white"
        )

        # Histogram of residuals
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=residuals,
            nbinsx=20,
            marker_color=self.colors[1]
        ))
        fig_hist.update_layout(
            title=f"{model_name} Residuals Histogram",
            xaxis_title="Residual",
            yaxis_title="Count",
            template="plotly_white"
        )

        # Predicted vs actual
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=y_pred, y=y_true,
            mode='markers',
            name='Predicted vs Actual',
            marker=dict(color=self.colors[2])
        ))
        fig_scatter.add_trace(go.Scatter(
            x=[y_true.min(), y_true.max()],
            y=[y_true.min(), y_true.max()],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        ))
        fig_scatter.update_layout(
            title=f"{model_name} Predicted vs Actual",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            template="plotly_white"
        )

        return fig_residuals, fig_hist, fig_scatter
