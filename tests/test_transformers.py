#!/usr/bin/env python3
"""
Tests for data transformers.
"""

import pytest
from datetime import datetime
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from src.transformation.transformers import (
    JsonTransformer,
    CsvTransformer,
    DateTimeTransformer
)
from src.transformation.base import TransformationResult, BaseTransformer

class TestData(BaseModel):
    """Test data model."""
    name: str
    age: int
    email: str = Field(..., pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

class ComplexTestData(BaseModel):
    """Test data model with complex types."""
    data: Dict[str, Any]
    timestamp: datetime

# Add a concrete transformer for testing base class
class TestTransformer(BaseTransformer):
    """Test transformer implementation."""
    def transform(self, data: Any) -> TransformationResult:
        return TransformationResult(is_successful=True, transformed_data=data)

@pytest.fixture
def sample_data() -> Dict[str, Any]:
    """Sample data for testing."""
    return {
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com"
    }

@pytest.fixture
def sample_data_list() -> List[Dict[str, Any]]:
    """Sample list of data for testing."""
    return [
        {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com"
        },
        {
            "name": "Jane Smith",
            "age": 25,
            "email": "jane@example.com"
        }
    ]

def test_json_transformer(sample_data):
    """Test JSON transformer."""
    transformer = JsonTransformer(input_schema=TestData)
    
    # Test transform to JSON
    result = transformer.transform(sample_data)
    assert result.is_successful
    assert isinstance(result.transformed_data, str)
    
    # Test transform from JSON
    json_str = result.transformed_data
    result = transformer.from_json(json_str)
    assert result.is_successful
    assert result.transformed_data == sample_data

def test_csv_transformer(sample_data_list):
    """Test CSV transformer."""
    transformer = CsvTransformer(input_schema=TestData)
    
    # Test transform to CSV
    result = transformer.transform(sample_data_list)
    assert result.is_successful
    assert isinstance(result.transformed_data, str)
    
    # Test transform from CSV
    csv_str = result.transformed_data
    result = transformer.from_csv(csv_str)
    assert result.is_successful
    assert len(result.transformed_data) == len(sample_data_list)
    assert all(
        key in result.transformed_data[0]
        for key in sample_data_list[0].keys()
    )

def test_datetime_transformer():
    """Test datetime transformer."""
    transformer = DateTimeTransformer()
    
    # Test string to datetime
    dt_str = "2024-01-01 12:00:00"
    result = transformer.transform(dt_str)
    assert result.is_successful
    assert isinstance(result.transformed_data, datetime)
    
    # Test datetime to string
    dt = result.transformed_data
    result = transformer.transform(dt)
    assert result.is_successful
    assert result.transformed_data == dt_str

def test_json_transformer_invalid_data():
    """Test JSON transformer with invalid data."""
    transformer = JsonTransformer(input_schema=TestData)
    invalid_data = {"name": "John Doe"}  # Missing required fields
    
    result = transformer.transform(invalid_data)
    assert not result.is_successful
    assert "Input data does not match schema" in result.errors

def test_csv_transformer_empty_data():
    """Test CSV transformer with empty data."""
    transformer = CsvTransformer(input_schema=TestData)
    empty_data = []
    
    result = transformer.transform(empty_data)
    assert not result.is_successful
    assert "Empty data list" in result.errors

def test_datetime_transformer_invalid_format():
    """Test datetime transformer with invalid format."""
    transformer = DateTimeTransformer()
    invalid_dt_str = "2024-01-01"  # Missing time
    
    result = transformer.transform(invalid_dt_str)
    assert not result.is_successful
    assert len(result.errors) > 0

# New test cases for improved coverage

def test_transformation_result_to_dict():
    """Test TransformationResult to_dict method."""
    result = TransformationResult(
        is_successful=True,
        transformed_data={"key": "value"},
        errors=["test error"]
    )
    result_dict = result.to_dict()
    assert result_dict["is_successful"] is True
    assert result_dict["transformed_data"] == {"key": "value"}
    assert result_dict["errors"] == ["test error"]

def test_json_transformer_batch():
    """Test JSON transformer batch processing."""
    transformer = JsonTransformer(input_schema=TestData)
    data_list = [
        {"name": "John Doe", "age": 30, "email": "john@example.com"},
        {"name": "Jane Smith", "age": 25, "email": "jane@example.com"}
    ]
    
    results = transformer.transform_batch(data_list)
    assert len(results) == 2
    assert all(result.is_successful for result in results)

def test_csv_transformer_custom_fieldnames():
    """Test CSV transformer with custom field names."""
    transformer = CsvTransformer(
        input_schema=TestData,
        fieldnames=["name", "age", "email"]
    )
    data = [{"name": "John", "age": 30, "email": "john@example.com"}]
    
    result = transformer.transform(data)
    assert result.is_successful
    assert "name,age,email" in result.transformed_data

def test_datetime_transformer_custom_format():
    """Test datetime transformer with custom format."""
    transformer = DateTimeTransformer(
        input_format="%Y-%m-%d",
        output_format="%d/%m/%Y"
    )
    
    # Test string to datetime with custom format
    dt_str = "2024-01-01"
    result = transformer.transform(dt_str)
    assert result.is_successful
    
    # Test datetime to string with custom format
    dt = result.transformed_data
    result = transformer.transform(dt)
    assert result.is_successful
    assert result.transformed_data == "01/01/2024"

def test_json_transformer_invalid_json():
    """Test JSON transformer with invalid JSON string."""
    transformer = JsonTransformer()
    invalid_json = "{'invalid': json}"
    
    result = transformer.from_json(invalid_json)
    assert not result.is_successful
    assert len(result.errors) > 0

def test_csv_transformer_invalid_csv():
    """Test CSV transformer with invalid CSV string."""
    transformer = CsvTransformer()
    invalid_csv = "invalid,csv\ndata"
    
    result = transformer.from_csv(invalid_csv)
    assert result.is_successful  # Should still parse but might not match schema
    assert len(result.transformed_data) > 0

def test_datetime_transformer_invalid_type():
    """Test datetime transformer with invalid input type."""
    transformer = DateTimeTransformer()
    invalid_input = 123  # Neither string nor datetime
    
    result = transformer.transform(invalid_input)
    assert not result.is_successful
    assert "Input must be string or datetime" in result.errors

# New test cases for error handling and remaining coverage

def test_base_transformer_validate_input_non_dict():
    """Test BaseTransformer validate_input with non-dict data."""
    class NonDictData(BaseModel):
        value: int
    
    transformer = TestTransformer(input_schema=NonDictData)
    result = transformer.validate_input(42)  # Pass an integer directly
    assert result is False

def test_base_transformer_validate_input_invalid_object():
    """Test BaseTransformer validate_input with invalid object data."""
    transformer = TestTransformer(input_schema=TestData)
    invalid_obj = object()  # Create an arbitrary object
    result = transformer.validate_input(invalid_obj)
    assert result is False

def test_json_transformer_serialize_error():
    """Test JSON transformer with unserializable data."""
    transformer = JsonTransformer()
    # Create a circular reference that can't be JSON serialized
    d = {}
    d['recursive'] = d
    
    result = transformer.transform(d)
    assert not result.is_successful
    assert "is not JSON serializable" in str(result.errors)

def test_csv_transformer_write_error():
    """Test CSV transformer with problematic data that causes write errors."""
    transformer = CsvTransformer()
    # Create data with inconsistent columns
    problematic_data = [
        {"col1": "value1", "col2": "value2"},
        {"col1": "value3", "col3": "value4"}  # Different columns
    ]
    
    result = transformer.transform(problematic_data)
    assert not result.is_successful
    assert "Failed to write CSV data" in str(result.errors)

def test_csv_transformer_read_error():
    """Test CSV transformer with malformed CSV that causes read errors."""
    transformer = CsvTransformer()
    malformed_csv = "header1,header2\nvalue1,value2,extra_value"  # Inconsistent columns
    
    result = transformer.from_csv(malformed_csv)
    assert not result.is_successful
    assert len(result.errors) > 0

def test_datetime_transformer_strftime_error():
    """Test datetime transformer with format that causes strftime error."""
    transformer = DateTimeTransformer()
    dt = datetime.now()
    
    # Test with an invalid format by modifying the format after construction
    transformer.output_format = "invalid_format"  # Completely invalid format
    result = transformer.transform(dt)
    assert not result.is_successful
    assert "Invalid output format" in str(result.errors)

def test_datetime_transformer_invalid_input_format():
    """Test datetime transformer with invalid input format."""
    transformer = DateTimeTransformer(input_format="%Y-%m-%d")  # Missing time component
    result = transformer.transform("2024-01-01 12:00:00")  # Input doesn't match format
    assert not result.is_successful
    assert "Invalid datetime format" in str(result.errors)

def test_json_transformer_complex_data():
    """Test JSON transformer with complex data types."""
    transformer = JsonTransformer(input_schema=ComplexTestData)
    complex_data = {
        "data": {"key": "value", "nested": {"x": 1}},
        "timestamp": datetime.now()
    }
    
    result = transformer.transform(complex_data)
    assert result.is_successful
    assert isinstance(result.transformed_data, str)

def test_csv_transformer_unicode():
    """Test CSV transformer with Unicode data."""
    transformer = CsvTransformer()
    unicode_data = [{
        "name": "测试",  # Chinese characters
        "value": "🌟"    # Emoji
    }]
    
    result = transformer.transform(unicode_data)
    assert result.is_successful
    assert isinstance(result.transformed_data, str)
    assert "测试" in result.transformed_data
    assert "🌟" in result.transformed_data

def test_datetime_transformer_invalid_strptime():
    """Test datetime transformer with invalid strptime format."""
    transformer = DateTimeTransformer(input_format="%Y-%m-%d %H:%M:%S.%f")
    invalid_datetime = "2024-01-01 12:00:00"  # Missing microseconds
    
    result = transformer.transform(invalid_datetime)
    assert not result.is_successful
    assert len(result.errors) > 0

def test_base_transformer_batch_empty():
    """Test batch transform with empty list."""
    transformer = TestTransformer()
    results = transformer.transform_batch([])
    assert isinstance(results, list)
    assert len(results) == 0

def test_base_transformer_batch_mixed():
    """Test batch transform with mixed success/failure."""
    class FailingTransformer(BaseTransformer):
        def transform(self, data: Any) -> TransformationResult:
            if isinstance(data, int):
                return TransformationResult(
                    is_successful=False,
                    errors=["Integer not allowed"]
                )
            return TransformationResult(
                is_successful=True,
                transformed_data=str(data)
            )
    
    transformer = FailingTransformer()
    results = transformer.transform_batch(["test", 42, "success"])
    assert len(results) == 3
    assert results[0].is_successful
    assert not results[1].is_successful
    assert "Integer not allowed" in results[1].errors
    assert results[2].is_successful

def test_csv_transformer_malformed():
    """Test CSV transformer with malformed data."""
    transformer = CsvTransformer()
    malformed_data = [
        {"col1": "value1", "col2": "value2"},
        {"col1": "value3"}  # Missing col2
    ]
    
    result = transformer.transform(malformed_data)
    assert result.is_successful  # Should still work but with empty values
    assert isinstance(result.transformed_data, str)
    assert "value1,value2" in result.transformed_data
    assert "value3," in result.transformed_data

def test_json_transformer_nested_validation():
    """Test JSON transformer with nested data validation."""
    class NestedModel(BaseModel):
        name: str
        details: Dict[str, Any]
    
    transformer = JsonTransformer(input_schema=NestedModel)
    valid_data = {
        "name": "test",
        "details": {"key": "value", "nested": {"x": 1}}
    }
    invalid_data = {
        "name": "test",
        "details": "not a dict"  # Should be a dict
    }
    
    result = transformer.transform(valid_data)
    assert result.is_successful
    
    result = transformer.transform(invalid_data)
    assert not result.is_successful
    assert "Input data does not match schema" in result.errors

def test_datetime_transformer_microseconds():
    """Test datetime transformer with microsecond precision."""
    transformer = DateTimeTransformer(
        input_format="%Y-%m-%d %H:%M:%S.%f",
        output_format="%Y-%m-%d %H:%M:%S.%f"
    )
    dt_str = "2024-01-01 12:00:00.123456"
    
    result = transformer.transform(dt_str)
    assert result.is_successful
    assert isinstance(result.transformed_data, datetime)
    assert result.transformed_data.microsecond == 123456 