#!/usr/bin/env python3
"""
Base classes for data transformation system.
"""

from typing import Any, Dict, List, Optional, Type, Union
from abc import ABC, abstractmethod
from pydantic import BaseModel

class TransformationResult:
    """Represents the result of a data transformation."""
    
    def __init__(
        self,
        is_successful: bool,
        transformed_data: Optional[Any] = None,
        errors: Optional[List[str]] = None
    ):
        self.is_successful = is_successful
        self.transformed_data = transformed_data
        self.errors = errors or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "is_successful": self.is_successful,
            "transformed_data": self.transformed_data,
            "errors": self.errors
        }

class BaseTransformer(ABC):
    """Base class for all data transformers."""
    
    def __init__(self, input_schema: Optional[Type[BaseModel]] = None):
        self.input_schema = input_schema
    
    @abstractmethod
    def transform(self, data: Any) -> TransformationResult:
        """Transform the input data."""
        pass
    
    def transform_batch(self, data_list: List[Any]) -> List[TransformationResult]:
        """Transform a batch of data items."""
        return [self.transform(item) for item in data_list]
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data against the schema if one is defined."""
        if self.input_schema is None:
            return True
        
        try:
            if isinstance(data, dict):
                self.input_schema(**data)
            else:
                self.input_schema.parse_obj(data)
            return True
        except Exception:
            return False 