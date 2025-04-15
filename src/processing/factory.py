from typing import Any, Dict, List, Optional, Type, Union
from .base import DataProcessor, DataPipeline
from .processors import (
    MissingValueProcessor,
    OutlierProcessor,
    ScalingProcessor,
    EncodingProcessor
)

class ProcessorFactory:
    """Factory for creating data processors."""
    
    @staticmethod
    def create_missing_value_processor(
        strategy: str,
        fill_value: Optional[Any] = None,
        **kwargs
    ) -> MissingValueProcessor:
        """Create a processor for handling missing values."""
        config = {
            "strategy": strategy,
            **kwargs
        }
        
        if strategy == "fill":
            if fill_value is None:
                raise ValueError("fill_value is required when strategy is 'fill'")
            config["fill_value"] = fill_value
        
        return MissingValueProcessor(config)

    @staticmethod
    def create_outlier_processor(
        method: str,
        columns: Union[str, List[str]],
        **kwargs
    ) -> OutlierProcessor:
        """Create a processor for handling outliers."""
        config = {
            "method": method,
            "columns": columns,
            **kwargs
        }
        return OutlierProcessor(config)

    @staticmethod
    def create_scaling_processor(
        method: str,
        columns: Union[str, List[str]],
        **kwargs
    ) -> ScalingProcessor:
        """Create a processor for scaling numeric columns."""
        config = {
            "method": method,
            "columns": columns,
            **kwargs
        }
        return ScalingProcessor(config)

    @staticmethod
    def create_encoding_processor(
        method: str,
        columns: Union[str, List[str]],
        target_column: Optional[str] = None,
        **kwargs
    ) -> EncodingProcessor:
        """Create a processor for encoding categorical columns."""
        config = {
            "method": method,
            "columns": columns,
            **kwargs
        }
        
        if method == "target":
            if target_column is None:
                raise ValueError("target_column is required when method is 'target'")
            config["target_column"] = target_column
        
        return EncodingProcessor(config)

    @staticmethod
    def create_data_pipeline(processors: Optional[List[DataProcessor]] = None) -> DataPipeline:
        """Create a data processing pipeline."""
        return DataPipeline(processors)

    @staticmethod
    def create_standard_pipeline(
        numeric_columns: List[str],
        categorical_columns: List[str],
        target_column: Optional[str] = None,
        handle_missing: bool = True,
        handle_outliers: bool = True,
        scale_numeric: bool = True,
        encode_categorical: bool = True
    ) -> DataPipeline:
        """Create a standard data processing pipeline."""
        processors = []
        
        # Handle missing values
        if handle_missing:
            # First handle missing values in numeric columns
            if numeric_columns:
                processors.append(
                    ProcessorFactory.create_missing_value_processor(
                        strategy="fill",
                        fill_value=0,
                        fill_params={"subset": numeric_columns}
                    )
                )
            
            # Then handle missing values in categorical columns
            if categorical_columns:
                processors.append(
                    ProcessorFactory.create_missing_value_processor(
                        strategy="fill",
                        fill_value="missing",
                        fill_params={"subset": categorical_columns}
                    )
                )
            
            # Handle any remaining missing values
            processors.append(
                ProcessorFactory.create_missing_value_processor(
                    strategy="fill",
                    fill_value=0
                )
            )
        
        # Handle outliers
        if handle_outliers and numeric_columns:
            processors.append(
                ProcessorFactory.create_outlier_processor(
                    method="iqr",
                    columns=numeric_columns
                )
            )
        
        # Scale numeric columns
        if scale_numeric and numeric_columns:
            processors.append(
                ProcessorFactory.create_scaling_processor(
                    method="standard",
                    columns=numeric_columns
                )
            )
        
        # Encode categorical columns
        if encode_categorical and categorical_columns:
            if target_column and target_column in categorical_columns:
                # Remove target from categorical columns for encoding
                cat_cols = [col for col in categorical_columns if col != target_column]
                if cat_cols:
                    processors.append(
                        ProcessorFactory.create_encoding_processor(
                            method="onehot",
                            columns=cat_cols
                        )
                    )
            else:
                processors.append(
                    ProcessorFactory.create_encoding_processor(
                        method="onehot",
                        columns=categorical_columns
                    )
                )
        
        return DataPipeline(processors) 