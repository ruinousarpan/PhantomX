import pytest
import pandas as pd
import numpy as np
from src.processing.base import ProcessingResult, DataProcessor, DataFrameProcessor, DataPipeline
from src.processing.processors import (
    MissingValueProcessor,
    OutlierProcessor,
    ScalingProcessor,
    EncodingProcessor
)
from src.processing.factory import ProcessorFactory
from src.processing.examples import create_sample_data

def test_missing_value_processor():
    """Test missing value processor with different strategies."""
    # Create a DataFrame with missing values
    df = pd.DataFrame({
        "col1": [1, np.nan, 3, np.nan, 5],
        "col2": [np.nan, 2, np.nan, 4, np.nan]
    })
    
    # Test drop strategy
    processor = ProcessorFactory.create_missing_value_processor(strategy="drop")
    result = processor.process(df)
    assert result.success
    assert result.data.shape[0] == 1  # Only one row has no missing values
    
    # Test fill strategy
    processor = ProcessorFactory.create_missing_value_processor(strategy="fill", fill_value=0)
    result = processor.process(df)
    assert result.success
    assert result.data.isnull().sum().sum() == 0  # No missing values
    
    # Test interpolate strategy
    processor = ProcessorFactory.create_missing_value_processor(strategy="interpolate")
    result = processor.process(df)
    assert result.success
    assert result.data.isnull().sum().sum() == 0  # No missing values

def test_outlier_processor():
    """Test outlier processor with different methods."""
    # Create a DataFrame with outliers
    df = pd.DataFrame({
        "col1": [1, 2, 3, 100, 200, 300]  # 100, 200, 300 are outliers
    })
    
    # Test zscore method
    processor = ProcessorFactory.create_outlier_processor(
        method="zscore",
        columns="col1",
        threshold=1  # Lower threshold to catch more outliers
    )
    result = processor.process(df)
    assert result.success
    assert result.data["col1"].isnull().sum() == 3  # Three outliers should be replaced with NaN
    
    # Test iqr method
    processor = ProcessorFactory.create_outlier_processor(method="iqr", columns="col1")
    result = processor.process(df)
    assert result.success
    assert result.data["col1"].isnull().sum() > 0  # Some outliers should be replaced with NaN
    
    # Test percentile method
    processor = ProcessorFactory.create_outlier_processor(
        method="percentile",
        columns="col1",
        lower_percentile=0.1,
        upper_percentile=0.9
    )
    result = processor.process(df)
    assert result.success
    assert result.data["col1"].isnull().sum() > 0  # Some outliers should be replaced with NaN

def test_scaling_processor():
    # Create a DataFrame with numeric values
    df = pd.DataFrame({
        "col1": [1, 2, 3, 4, 5]
    })
    
    # Test minmax scaling
    processor = ProcessorFactory.create_scaling_processor(method="minmax", columns="col1")
    result = processor.process(df)
    assert result.success
    assert result.data["col1"].min() == 0  # Min should be 0
    assert result.data["col1"].max() == 1  # Max should be 1
    
    # Test standard scaling
    processor = ProcessorFactory.create_scaling_processor(method="standard", columns="col1")
    result = processor.process(df)
    assert result.success
    assert abs(result.data["col1"].mean()) < 1e-10  # Mean should be close to 0
    assert abs(result.data["col1"].std() - 1) < 1e-10  # Std should be close to 1
    
    # Test robust scaling
    processor = ProcessorFactory.create_scaling_processor(method="robust", columns="col1")
    result = processor.process(df)
    assert result.success
    assert abs(result.data["col1"].median()) < 1e-10  # Median should be close to 0

def test_encoding_processor():
    # Create a DataFrame with categorical values
    df = pd.DataFrame({
        "col1": ["A", "B", "A", "C", "B"],
        "target": [0, 1, 0, 1, 0]
    })
    
    # Test onehot encoding
    processor = ProcessorFactory.create_encoding_processor(method="onehot", columns="col1")
    result = processor.process(df)
    assert result.success
    assert "col1_A" in result.data.columns
    assert "col1_B" in result.data.columns
    assert "col1_C" in result.data.columns
    assert "col1" not in result.data.columns
    
    # Test label encoding
    processor = ProcessorFactory.create_encoding_processor(method="label", columns="col1")
    result = processor.process(df)
    assert result.success
    assert result.data["col1"].dtype == np.int64
    
    # Test target encoding
    processor = ProcessorFactory.create_encoding_processor(
        method="target",
        columns="col1",
        target_column="target"
    )
    result = processor.process(df)
    assert result.success
    assert result.data["col1"].dtype == np.float64

def test_data_pipeline():
    """Test standard data pipeline with sample data."""
    # Create sample data
    df = create_sample_data()
    
    # Create a pipeline with multiple processors
    pipeline = ProcessorFactory.create_standard_pipeline(
        numeric_columns=["age", "income"],
        categorical_columns=["education", "occupation"],
        target_column="target",
        handle_missing=True,
        handle_outliers=True,
        scale_numeric=True,
        encode_categorical=True
    )
    
    # Process the data
    result = pipeline.process(df)
    if not result.success:
        print("\nPipeline errors:")
        for error in result.errors:
            print(f"- {error}")
    assert result.success
    assert isinstance(result.data, pd.DataFrame)
    
    # Check that the data has been processed
    assert all(col in result.data.columns for col in ["target"])  # Target column preserved
    assert result.data.shape[1] > df.shape[1]  # More columns due to one-hot encoding
    
    # Check that numeric columns have been processed
    assert "age" in result.data.columns
    assert "income" in result.data.columns
    assert result.data["age"].isnull().sum() == 0  # No missing values in age
    assert result.data["income"].isnull().sum() == 0  # No missing values in income
    
    # Check that categorical columns have been encoded
    assert "education_Bachelor's" in result.data.columns
    assert "occupation_Engineer" in result.data.columns

def test_custom_pipeline():
    """Test custom pipeline with multiple processors."""
    # Create sample data
    df = create_sample_data()
    
    # Create a custom pipeline
    pipeline = ProcessorFactory.create_data_pipeline([
        # First handle missing values in numeric columns
        ProcessorFactory.create_missing_value_processor(
            strategy="fill",
            fill_value=0,
            fill_params={"subset": ["age", "income"]}
        ),
        # Then handle missing values in categorical columns
        ProcessorFactory.create_missing_value_processor(
            strategy="fill",
            fill_value="missing",
            fill_params={"subset": ["education", "occupation"]}
        ),
        # Handle outliers
        ProcessorFactory.create_outlier_processor(
            method="zscore",
            columns=["age", "income"],
            threshold=2
        ),
        # Scale numeric columns
        ProcessorFactory.create_scaling_processor(
            method="minmax",
            columns=["age", "income"]
        ),
        # Encode categorical columns
        ProcessorFactory.create_encoding_processor(
            method="onehot",
            columns=["education", "occupation"]
        )
    ])
    
    # Process the data
    result = pipeline.process(df)
    if not result.success:
        print("\nPipeline errors:")
        for error in result.errors:
            print(f"- {error}")
    assert result.success
    assert isinstance(result.data, pd.DataFrame)
    
    # Check that the data has been processed
    assert result.data.isnull().sum().sum() == 0  # No missing values
    assert result.data.shape[1] > df.shape[1]  # More columns due to one-hot encoding
    
    # Check specific columns
    assert "age" in result.data.columns
    assert "income" in result.data.columns
    assert "education_Bachelor's" in result.data.columns
    assert "occupation_Engineer" in result.data.columns 