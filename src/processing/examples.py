import pandas as pd
import numpy as np
from .factory import ProcessorFactory

def create_sample_data() -> pd.DataFrame:
    """Create a sample DataFrame with various data types and issues."""
    np.random.seed(42)
    
    # Create a DataFrame with 100 rows
    n_samples = 100
    
    # Numeric columns with missing values and outliers
    age = np.random.normal(35, 10, n_samples)
    age[0:5] = np.nan  # Add missing values
    age[5:10] = 150  # Add outliers
    
    income = np.random.normal(50000, 20000, n_samples)
    income[10:15] = np.nan  # Add missing values
    income[15:20] = -1000  # Add outliers
    
    # Categorical columns
    education = np.random.choice(
        ["High School", "Bachelor's", "Master's", "PhD", np.nan],
        n_samples,
        p=[0.3, 0.4, 0.2, 0.05, 0.05]
    )
    
    occupation = np.random.choice(
        ["Engineer", "Teacher", "Doctor", "Artist", np.nan],
        n_samples,
        p=[0.3, 0.3, 0.2, 0.1, 0.1]
    )
    
    # Target variable (for demonstration)
    target = np.random.normal(0, 1, n_samples)
    target = (target > 0).astype(int)  # Convert to binary
    
    # Create DataFrame
    df = pd.DataFrame({
        "age": age,
        "income": income,
        "education": education,
        "occupation": occupation,
        "target": target
    })
    
    return df

def process_sample_data() -> None:
    """Demonstrate the data processing system with sample data."""
    # Create sample data
    df = create_sample_data()
    print("Original DataFrame:")
    print(df.head())
    print("\nDataFrame Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Create a standard processing pipeline
    numeric_columns = ["age", "income"]
    categorical_columns = ["education", "occupation"]
    target_column = "target"
    
    pipeline = ProcessorFactory.create_standard_pipeline(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        target_column=target_column
    )
    
    # Process the data
    result = pipeline.process(df)
    
    if result.success:
        processed_df = result.data
        print("\nProcessed DataFrame:")
        print(processed_df.head())
        print("\nProcessed DataFrame Info:")
        print(processed_df.info())
        print("\nMissing Values:")
        print(processed_df.isnull().sum())
        
        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"- {warning}")
    else:
        print("\nProcessing failed:")
        for error in result.errors:
            print(f"- {error}")

def create_custom_pipeline() -> None:
    """Demonstrate creating a custom processing pipeline."""
    # Create sample data
    df = create_sample_data()
    
    # Create a custom pipeline
    pipeline = ProcessorFactory.create_data_pipeline([
        # Handle missing values in age and income
        ProcessorFactory.create_missing_value_processor(
            strategy="fill",
            fill_value={"age": df["age"].mean(), "income": df["income"].median()}
        ),
        
        # Handle outliers in age using z-score
        ProcessorFactory.create_outlier_processor(
            method="zscore",
            columns="age",
            threshold=2.5
        ),
        
        # Scale income using min-max scaling
        ProcessorFactory.create_scaling_processor(
            method="minmax",
            columns="income"
        ),
        
        # Encode education using one-hot encoding
        ProcessorFactory.create_encoding_processor(
            method="onehot",
            columns="education"
        ),
        
        # Encode occupation using target encoding
        ProcessorFactory.create_encoding_processor(
            method="target",
            columns="occupation",
            target_column="target"
        )
    ])
    
    # Process the data
    result = pipeline.process(df)
    
    if result.success:
        processed_df = result.data
        print("\nCustom Pipeline - Processed DataFrame:")
        print(processed_df.head())
        print("\nCustom Pipeline - Processed DataFrame Info:")
        print(processed_df.info())
    else:
        print("\nCustom Pipeline - Processing failed:")
        for error in result.errors:
            print(f"- {error}")

if __name__ == "__main__":
    print("=== Standard Pipeline Example ===")
    process_sample_data()
    
    print("\n=== Custom Pipeline Example ===")
    create_custom_pipeline() 