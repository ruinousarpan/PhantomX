import pytest
import pandas as pd
import numpy as np
from src.processing.processors import MissingValueProcessor
from src.processing.base import ProcessingResult

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'A': [1, np.nan, 3, np.nan, 5],
        'B': ['x', None, 'z', 'w', None],
        'C': [1.1, 2.2, np.nan, 4.4, 5.5]
    })

def test_missing_value_processor_drop():
    """Test MissingValueProcessor with drop strategy."""
    processor = MissingValueProcessor({"strategy": "drop"})
    df = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': ['x', 'y', np.nan]
    })
    
    result = processor.process(df)
    assert result.success
    assert len(result.data) == 1  # Only one row should remain
    assert result.data.iloc[0]['A'] == 1
    assert result.data.iloc[0]['B'] == 'x'

def test_missing_value_processor_fill():
    """Test MissingValueProcessor with fill strategy."""
    processor = MissingValueProcessor({
        "strategy": "fill",
        "fill_value": 0
    })
    df = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': [4, 5, np.nan]
    })
    
    result = processor.process(df)
    assert result.success
    assert result.data['A'].isna().sum() == 0
    assert result.data['B'].isna().sum() == 0
    assert result.data['A'].iloc[1] == 0
    assert result.data['B'].iloc[2] == 0

def test_missing_value_processor_fill_dict():
    """Test MissingValueProcessor with fill strategy using dictionary."""
    processor = MissingValueProcessor({
        "strategy": "fill",
        "fill_value": {'A': -1, 'B': 999}
    })
    df = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': [4, 5, np.nan]
    })
    
    result = processor.process(df)
    assert result.success
    assert result.data['A'].iloc[1] == -1
    assert result.data['B'].iloc[2] == 999

def test_missing_value_processor_interpolate():
    """Test MissingValueProcessor with interpolate strategy."""
    processor = MissingValueProcessor({"strategy": "interpolate"})
    df = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': [4, np.nan, 6]
    })
    
    result = processor.process(df)
    assert result.success
    assert result.data['A'].iloc[1] == 2  # Linear interpolation
    assert result.data['B'].iloc[1] == 5  # Linear interpolation

def test_missing_value_processor_invalid_strategy():
    """Test MissingValueProcessor with invalid strategy."""
    with pytest.raises(ValueError, match="Strategy must be one of: drop, fill, interpolate"):
        MissingValueProcessor({"strategy": "invalid"})

def test_missing_value_processor_missing_strategy():
    """Test MissingValueProcessor with missing strategy."""
    with pytest.raises(ValueError, match="Missing required config key: strategy"):
        MissingValueProcessor({})

def test_missing_value_processor_fill_missing_value():
    """Test MissingValueProcessor with fill strategy but missing fill_value."""
    with pytest.raises(ValueError, match="fill_value is required when strategy is 'fill'"):
        MissingValueProcessor({"strategy": "fill"})

def test_missing_value_processor_invalid_input():
    """Test MissingValueProcessor with invalid input type."""
    processor = MissingValueProcessor({"strategy": "drop"})
    result = processor.process([1, 2, 3])  # Not a DataFrame
    assert not result.success
    assert "Expected pandas DataFrame" in result.errors[0]

def test_missing_value_processor_empty_df():
    """Test MissingValueProcessor with empty DataFrame."""
    processor = MissingValueProcessor({"strategy": "drop"})
    df = pd.DataFrame()
    result = processor.process(df)
    assert result.success
    assert len(result.data) == 0
    assert len(result.data.columns) == 0

def test_missing_value_processor_empty_df_with_columns():
    """Test MissingValueProcessor with empty DataFrame but with columns defined."""
    processor = MissingValueProcessor({"strategy": "drop"})
    df = pd.DataFrame(columns=['A', 'B', 'C'])
    result = processor.process(df)
    assert result.success
    assert len(result.data) == 0
    assert list(result.data.columns) == ['A', 'B', 'C']

def test_missing_value_processor_empty_df_fill():
    """Test MissingValueProcessor with empty DataFrame using fill strategy."""
    processor = MissingValueProcessor({
        "strategy": "fill",
        "fill_value": {'A': 0, 'B': 'missing', 'C': False}
    })
    df = pd.DataFrame(columns=['A', 'B', 'C'])
    result = processor.process(df)
    assert result.success
    assert len(result.data) == 0
    assert list(result.data.columns) == ['A', 'B', 'C']

def test_missing_value_processor_empty_df_interpolate():
    """Test MissingValueProcessor with empty DataFrame using interpolate strategy."""
    processor = MissingValueProcessor({"strategy": "interpolate"})
    df = pd.DataFrame(columns=['A', 'B', 'C'])
    result = processor.process(df)
    assert result.success
    assert len(result.data) == 0
    assert list(result.data.columns) == ['A', 'B', 'C']

def test_missing_value_processor_all_missing():
    """Test MissingValueProcessor with all missing values."""
    processor = MissingValueProcessor({"strategy": "drop"})
    df = pd.DataFrame({
        'A': [np.nan, np.nan],
        'B': [None, None]
    })
    result = processor.process(df)
    assert result.success
    assert len(result.data) == 0

def test_missing_value_processor_no_missing():
    """Test MissingValueProcessor with no missing values."""
    processor = MissingValueProcessor({"strategy": "drop"})
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['x', 'y', 'z']
    })
    result = processor.process(df)
    assert result.success
    assert len(result.data) == 3
    pd.testing.assert_frame_equal(result.data, df)

def test_missing_value_processor_mixed_types():
    """Test MissingValueProcessor with mixed data types."""
    processor = MissingValueProcessor({
        "strategy": "fill",
        "fill_value": {'A': 0, 'B': 'missing', 'C': False}
    })
    df = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': ['x', None, 'z'],
        'C': [True, False, None]
    })
    result = processor.process(df)
    assert result.success
    assert result.data['A'].iloc[1] == 0
    assert result.data['B'].iloc[1] == 'missing'
    assert result.data['C'].iloc[2] == False

def test_missing_value_processor_interpolate_non_numeric():
    """Test MissingValueProcessor interpolate with non-numeric data."""
    processor = MissingValueProcessor({"strategy": "interpolate"})
    df = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': ['x', None, 'z']  # Non-numeric column
    })
    result = processor.process(df)
    assert not result.success
    assert "Cannot interpolate non-numeric data" in result.errors[0]

def test_missing_value_processor_partial_fill_dict():
    """Test MissingValueProcessor with partial fill dictionary."""
    processor = MissingValueProcessor({
        "strategy": "fill",
        "fill_value": {'A': 0}  # Only specify fill for column A
    })
    df = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': [4, np.nan, 6]
    })
    result = processor.process(df)
    assert not result.success
    assert "Missing fill values for columns: B" in result.errors[0]

def test_missing_value_processor_empty_df_with_dtypes():
    """Test MissingValueProcessor with empty DataFrame with specific dtypes."""
    processor = MissingValueProcessor({"strategy": "drop"})
    df = pd.DataFrame({
        'A': pd.Series(dtype='int64'),
        'B': pd.Series(dtype='float64'),
        'C': pd.Series(dtype='object')
    })
    result = processor.process(df)
    assert result.success
    assert len(result.data) == 0
    assert result.data['A'].dtype == 'int64'
    assert result.data['B'].dtype == 'float64'
    assert result.data['C'].dtype == 'object'

def test_missing_value_processor_empty_df_with_index():
    """Test MissingValueProcessor with empty DataFrame with custom index."""
    processor = MissingValueProcessor({"strategy": "drop"})
    df = pd.DataFrame(columns=['A', 'B'], index=pd.date_range('2024-01-01', periods=0))
    result = processor.process(df)
    assert result.success
    assert len(result.data) == 0
    assert isinstance(result.data.index, pd.DatetimeIndex)

def test_missing_value_processor_empty_df_with_metadata():
    """Test MissingValueProcessor with empty DataFrame containing metadata."""
    processor = MissingValueProcessor({"strategy": "drop"})
    df = pd.DataFrame(columns=['A', 'B'])
    df.attrs['description'] = 'Test DataFrame'
    result = processor.process(df)
    assert result.success
    assert len(result.data) == 0
    assert result.data.attrs.get('description') == 'Test DataFrame'

def test_missing_value_processor_empty_df_with_multiindex():
    """Test MissingValueProcessor with empty DataFrame with MultiIndex columns."""
    processor = MissingValueProcessor({"strategy": "drop"})
    columns = pd.MultiIndex.from_tuples([
        ('A', 'x'), ('A', 'y'), ('B', 'z')
    ])
    df = pd.DataFrame(columns=columns)
    result = processor.process(df)
    assert result.success
    assert len(result.data) == 0
    assert isinstance(result.data.columns, pd.MultiIndex)
    assert list(result.data.columns) == list(columns) 