from typing import Any, Dict, List, Optional, Union, Callable
import pandas as pd
import numpy as np
from .base import DataFrameProcessor, ProcessingResult

# Opt-in to future behavior for downcasting
pd.set_option('future.no_silent_downcasting', True)

class MissingValueProcessor(DataFrameProcessor):
    """Processor for handling missing values in a DataFrame."""
    
    def _validate_config(self) -> None:
        """Validate the processor configuration."""
        required_keys = ["strategy"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        if self.config["strategy"] not in ["drop", "fill", "interpolate"]:
            raise ValueError("Strategy must be one of: drop, fill, interpolate")
        
        if self.config["strategy"] == "fill" and "fill_value" not in self.config:
            raise ValueError("fill_value is required when strategy is 'fill'")

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process missing values in the DataFrame."""
        strategy = self.config["strategy"]
        
        # Handle empty DataFrame
        if len(df) == 0:
            return df.copy()
        
        result_df = df.copy()
        
        if strategy == "drop":
            # Drop rows where any value is NaN, unless subset is specified
            drop_params = self.config.get("drop_params", {})
            if "subset" not in drop_params:
                drop_params["how"] = "any"  # Drop if any column has NaN
            return df.dropna(**drop_params)
        
        elif strategy == "fill":
            fill_value = self.config["fill_value"]
            fill_params = self.config.get("fill_params", {})
            
            if isinstance(fill_value, dict):
                # Check if all columns with missing values have corresponding fill values
                missing_cols = result_df.columns[result_df.isna().any()].tolist()
                missing_fill_values = [col for col in missing_cols if col not in fill_value]
                if missing_fill_values:
                    raise ValueError(
                        f"Missing fill values for columns: {', '.join(missing_fill_values)}"
                    )
                for col, val in fill_value.items():
                    if col in result_df.columns:
                        # First infer objects to handle dtype properly
                        result_df[col] = result_df[col].infer_objects(copy=False)
                        result_df[col] = result_df[col].fillna(val)
            else:
                # First infer objects to handle dtype properly
                result_df = result_df.infer_objects(copy=False)
                result_df = result_df.fillna(fill_value)
            
            return result_df
        
        else:  # interpolate
            # Check if there are any non-numeric columns with missing values
            non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64']).columns
            non_numeric_missing = [col for col in non_numeric_cols if df[col].isna().any()]
            if non_numeric_missing:
                raise ValueError(f"Cannot interpolate non-numeric data in column: {non_numeric_missing[0]}")
            
            return df.interpolate(**self.config.get("interpolate_params", {}))

class OutlierProcessor(DataFrameProcessor):
    """Processor for handling outliers in a DataFrame."""
    
    def _validate_config(self) -> None:
        """Validate the processor configuration."""
        required_keys = ["method", "columns"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        if self.config["method"] not in ["zscore", "iqr", "percentile"]:
            raise ValueError("Method must be one of: zscore, iqr, percentile")
        
        if not isinstance(self.config["columns"], (list, str)):
            raise ValueError("columns must be a list or string")

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process outliers in the DataFrame."""
        method = self.config["method"]
        columns = self.config["columns"]
        if isinstance(columns, str):
            columns = [columns]
        
        result_df = df.copy()
        
        for column in columns:
            if column not in df.columns:
                continue
                
            if method == "zscore":
                # Calculate z-scores, handling NaN values
                data = df[column].dropna()
                if len(data) < 2:  # Need at least 2 values for std
                    continue
                    
                mean = data.mean()
                std = data.std()
                if std == 0:  # All values are the same
                    continue
                    
                z_scores = np.abs((df[column] - mean) / std)
                threshold = float(self.config.get("threshold", 3))
                # Mark as outliers where z-score exceeds threshold
                result_df.loc[z_scores > threshold, column] = np.nan
                
            elif method == "iqr":
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                multiplier = float(self.config.get("iqr_multiplier", 1.5))
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                # Mark as outliers where value is outside bounds
                result_df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = np.nan
                
            else:  # percentile
                lower = df[column].quantile(self.config.get("lower_percentile", 0.01))
                upper = df[column].quantile(self.config.get("upper_percentile", 0.99))
                # Mark as outliers where value is outside percentile bounds
                result_df.loc[(df[column] < lower) | (df[column] > upper), column] = np.nan
        
        return result_df

class ScalingProcessor(DataFrameProcessor):
    """Processor for scaling numeric columns in a DataFrame."""
    
    def _validate_config(self) -> None:
        """Validate the processor configuration."""
        required_keys = ["method", "columns"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        if self.config["method"] not in ["minmax", "standard", "robust"]:
            raise ValueError("Method must be one of: minmax, standard, robust")
        
        if not isinstance(self.config["columns"], (list, str)):
            raise ValueError("columns must be a list or string")

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric columns in the DataFrame."""
        method = self.config["method"]
        columns = self.config["columns"]
        if isinstance(columns, str):
            columns = [columns]
        
        result_df = df.copy()
        
        for column in columns:
            if column not in df.columns:
                continue
                
            if method == "minmax":
                min_val = df[column].min()
                max_val = df[column].max()
                result_df[column] = (df[column] - min_val) / (max_val - min_val)
                
            elif method == "standard":
                mean_val = df[column].mean()
                std_val = df[column].std()
                result_df[column] = (df[column] - mean_val) / std_val
                
            else:  # robust
                median_val = df[column].median()
                iqr_val = df[column].quantile(0.75) - df[column].quantile(0.25)
                result_df[column] = (df[column] - median_val) / iqr_val
        
        return result_df

class EncodingProcessor(DataFrameProcessor):
    """Processor for encoding categorical columns in a DataFrame."""
    
    def _validate_config(self) -> None:
        """Validate the processor configuration."""
        required_keys = ["method", "columns"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        if self.config["method"] not in ["onehot", "label", "target"]:
            raise ValueError("Method must be one of: onehot, label, target")
        
        if not isinstance(self.config["columns"], (list, str)):
            raise ValueError("columns must be a list or string")
        
        if self.config["method"] == "target" and "target_column" not in self.config:
            raise ValueError("target_column is required when method is 'target'")

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns in the DataFrame."""
        method = self.config["method"]
        columns = self.config["columns"]
        if isinstance(columns, str):
            columns = [columns]
        
        result_df = df.copy()
        
        for column in columns:
            if column not in df.columns:
                continue
                
            if method == "onehot":
                dummies = pd.get_dummies(df[column], prefix=column)
                result_df = pd.concat([result_df, dummies], axis=1)
                result_df.drop(column, axis=1, inplace=True)
                
            elif method == "label":
                # Use int64 for label encoding
                result_df[column] = df[column].astype("category").cat.codes.astype(np.int64)
                
            else:  # target encoding
                target_column = self.config["target_column"]
                if target_column not in df.columns:
                    continue
                    
                # Calculate mean target value for each category
                encoding_map = df.groupby(column)[target_column].mean()
                result_df[column] = df[column].map(encoding_map)
        
        return result_df 