#!/usr/bin/env python3
"""
Concrete implementations of data transformers.
"""

from typing import Any, Dict, List, Optional, Type, Union
from datetime import datetime
import json
import csv
from io import StringIO
from pydantic import BaseModel
from .base import BaseTransformer, TransformationResult

class JsonTransformer(BaseTransformer):
    """Transforms data to and from JSON format."""
    
    def transform(self, data: Any) -> TransformationResult:
        """Transform data to JSON string."""
        if not self.validate_input(data):
            return TransformationResult(
                is_successful=False,
                errors=["Input data does not match schema"]
            )
        
        try:
            json_str = json.dumps(data, default=str)
            return TransformationResult(
                is_successful=True,
                transformed_data=json_str
            )
        except (TypeError, ValueError) as e:
            return TransformationResult(
                is_successful=False,
                errors=[f"Data is not JSON serializable: {str(e)}"]
            )
        except Exception as e:
            return TransformationResult(
                is_successful=False,
                errors=[str(e)]
            )
    
    def from_json(self, json_str: str) -> TransformationResult:
        """Transform JSON string back to data."""
        try:
            data = json.loads(json_str)
            return TransformationResult(
                is_successful=True,
                transformed_data=data
            )
        except json.JSONDecodeError as e:
            return TransformationResult(
                is_successful=False,
                errors=[f"Invalid JSON format: {str(e)}"]
            )
        except Exception as e:
            return TransformationResult(
                is_successful=False,
                errors=[str(e)]
            )

class CsvTransformer(BaseTransformer):
    """Transforms data to and from CSV format."""
    
    def __init__(
        self,
        input_schema: Optional[Type[BaseModel]] = None,
        fieldnames: Optional[List[str]] = None
    ):
        super().__init__(input_schema)
        self.fieldnames = fieldnames
    
    def transform(self, data: List[Dict[str, Any]]) -> TransformationResult:
        """Transform list of dictionaries to CSV string."""
        if not data:
            return TransformationResult(
                is_successful=False,
                errors=["Empty data list"]
            )
        
        if not self.validate_input(data[0]):
            return TransformationResult(
                is_successful=False,
                errors=["Input data does not match schema"]
            )
        
        try:
            output = StringIO()
            fieldnames = self.fieldnames or list(data[0].keys())
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
            return TransformationResult(
                is_successful=True,
                transformed_data=output.getvalue()
            )
        except (TypeError, ValueError) as e:
            return TransformationResult(
                is_successful=False,
                errors=[f"Failed to write CSV data: {str(e)}"]
            )
        except Exception as e:
            return TransformationResult(
                is_successful=False,
                errors=[str(e)]
            )
    
    def from_csv(self, csv_str: str) -> TransformationResult:
        """Transform CSV string back to list of dictionaries."""
        try:
            input_data = StringIO(csv_str)
            reader = csv.DictReader(input_data)
            
            try:
                data = list(reader)
                if not all(len(row) == len(reader.fieldnames) for row in data):
                    return TransformationResult(
                        is_successful=False,
                        errors=["Inconsistent number of columns in CSV data"]
                    )
            except csv.Error as e:
                return TransformationResult(
                    is_successful=False,
                    errors=[f"Failed to read CSV data: {str(e)}"]
                )
            
            return TransformationResult(
                is_successful=True,
                transformed_data=data
            )
        except Exception as e:
            return TransformationResult(
                is_successful=False,
                errors=[str(e)]
            )

class DateTimeTransformer(BaseTransformer):
    """Transforms datetime objects to and from various string formats."""
    
    def __init__(
        self,
        input_format: str = "%Y-%m-%d %H:%M:%S",
        output_format: str = "%Y-%m-%d %H:%M:%S"
    ):
        super().__init__()
        self.input_format = input_format
        self.output_format = output_format
    
    def _validate_format(self, fmt: str) -> bool:
        """Validate a datetime format string."""
        if not any(c.startswith('%') for c in fmt):
            return False
        try:
            datetime.now().strftime(fmt)
            return True
        except ValueError:
            return False
    
    def transform(self, data: Union[str, datetime]) -> TransformationResult:
        """Transform datetime to string or string to datetime."""
        if not isinstance(data, (str, datetime)):
            return TransformationResult(
                is_successful=False,
                errors=["Input must be string or datetime"]
            )
        
        try:
            if isinstance(data, str):
                if not self._validate_format(self.input_format):
                    return TransformationResult(
                        is_successful=False,
                        errors=[f"Invalid input format: {self.input_format}"]
                    )
                # String to datetime
                dt = datetime.strptime(data, self.input_format)
                return TransformationResult(
                    is_successful=True,
                    transformed_data=dt
                )
            else:
                if not self._validate_format(self.output_format):
                    return TransformationResult(
                        is_successful=False,
                        errors=[f"Invalid output format: {self.output_format}"]
                    )
                # Datetime to string
                dt_str = data.strftime(self.output_format)
                return TransformationResult(
                    is_successful=True,
                    transformed_data=dt_str
                )
        except ValueError as e:
            return TransformationResult(
                is_successful=False,
                errors=[f"Invalid datetime format: {str(e)}"]
            )
        except Exception as e:
            return TransformationResult(
                is_successful=False,
                errors=[str(e)]
            )