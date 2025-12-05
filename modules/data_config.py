"""
This Module will be used to help clean the data including how to handle missing data, and how to split the data
"""

from typing import Dict, List, Literal, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer


class PrepareData:
    def __init__(self,
                 options: Literal["impute", "fill", "drop"] = "fill", 
                 imputed_options: Literal["simple", "knn", "iterative"] = "simple",
                 ):
        self, 
        self.options = options, 
        self.imputed_options = imputed_options

    def clean_miss_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Only treat the target column as missing data
        target = 'y'

        if self.options == "impute":
            if self.imputed_options == "simple":
                imp = SimpleImputer(strategy="mean")
            elif self.imputed_options == "knn":
                imp = KNNImputer(n_neighbors=5)
            elif self.imputed_options == "iterative":
                imp = IterativeImputer(max_iter=10, random_state=24)
            else:
                imp = None

            if imp:
                df[[target]] = imp.fit_transform(df[[target]])

        elif self.options == "fill":
            df[target] = df[target].ffill()

        else:  # drop
            df = df.dropna(subset=[target])

        return df

    def wrangle_data(
        self,
        data: pd.DataFrame,
        target_col: str = 'y',
        date_col: str = 'ds',
        unique_id: str = 'series_1'
    ) -> pd.DataFrame:
        """
        Prepare data in MLForecast format.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe
        target_col : str
            Name of target column
        date_col : str
            Name of date column
        unique_id : str
            Unique identifier for the series

        Returns
        -------
        pd.DataFrame
            Data in MLForecast format with columns [unique_id, ds, y]
        """
        df = data.copy()

        # Handle index as date column if needed
        if date_col not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: 'ds'})
        elif date_col in df.columns:
            df = df.rename(columns={date_col: 'ds'})
# Create column mapping for renaming
        rename_dict = {}
        
        # Map date column to 'ds'
        if date_col in df.columns and date_col != 'ds':
            rename_dict[date_col] = 'ds'
        
        # Map target column to 'y'
        if target_col in df.columns and target_col != 'y':
            rename_dict[target_col] = 'y'
        
        # Apply renaming
        if rename_dict:
            df = df.rename(columns=rename_dict)
        
        # Ensure ds and y columns exist
        if 'ds' not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in dataframe. Available columns: {df.columns.tolist()}")
        
        if 'y' not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe. Available columns: {df.columns.tolist()}")

        # Add unique_id
        df['unique_id'] = unique_id

        # Ensure correct column order
        df = df[['unique_id', 'ds', 'y']]

        # Ensure ds is datetime
        df['ds'] = pd.to_datetime(df['ds'])

        return df
    
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





            

