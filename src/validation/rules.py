from typing import Any, Dict, List, Optional, Union
from .base import ValidationRule

class RequiredRule(ValidationRule):
    """Validates that a field is present and not None."""
    
    def __init__(self, field: str):
        super().__init__(field, "required")

    def validate(self, value: Any) -> bool:
        return value is not None

    def get_error_message(self, value: Any) -> str:
        return f"Field '{self.field}' is required"

class TypeRule(ValidationRule):
    """Validates that a field is of a specific type."""
    
    def __init__(self, field: str, expected_type: Union[type, tuple]):
        super().__init__(field, "type", {"expected_type": expected_type})

    def validate(self, value: Any) -> bool:
        return isinstance(value, self.params["expected_type"])

    def get_error_message(self, value: Any) -> str:
        expected_type = self.params["expected_type"]
        if isinstance(expected_type, tuple):
            expected_type = " or ".join(t.__name__ for t in expected_type)
        return f"Field '{self.field}' must be of type {expected_type}"

class RangeRule(ValidationRule):
    """Validates that a numeric field is within a specific range."""
    
    def __init__(self, field: str, min_value: Optional[float] = None, max_value: Optional[float] = None):
        super().__init__(field, "range", {
            "min_value": min_value,
            "max_value": max_value
        })

    def validate(self, value: Any) -> bool:
        if not isinstance(value, (int, float)):
            return False
        
        min_value = self.params["min_value"]
        max_value = self.params["max_value"]
        
        if min_value is not None and value < min_value:
            return False
        if max_value is not None and value > max_value:
            return False
        
        return True

    def get_error_message(self, value: Any) -> str:
        min_value = self.params["min_value"]
        max_value = self.params["max_value"]
        
        if min_value is not None and max_value is not None:
            return f"Field '{self.field}' must be between {min_value} and {max_value}"
        elif min_value is not None:
            return f"Field '{self.field}' must be greater than or equal to {min_value}"
        else:
            return f"Field '{self.field}' must be less than or equal to {max_value}"

class LengthRule(ValidationRule):
    """Validates the length of a string or list field."""
    
    def __init__(self, field: str, min_length: Optional[int] = None, max_length: Optional[int] = None):
        super().__init__(field, "length", {
            "min_length": min_length,
            "max_length": max_length
        })

    def validate(self, value: Any) -> bool:
        if not hasattr(value, "__len__"):
            return False
        
        length = len(value)
        min_length = self.params["min_length"]
        max_length = self.params["max_length"]
        
        if min_length is not None and length < min_length:
            return False
        if max_length is not None and length > max_length:
            return False
        
        return True

    def get_error_message(self, value: Any) -> str:
        min_length = self.params["min_length"]
        max_length = self.params["max_length"]
        
        if min_length is not None and max_length is not None:
            return f"Field '{self.field}' length must be between {min_length} and {max_length}"
        elif min_length is not None:
            return f"Field '{self.field}' length must be at least {min_length}"
        else:
            return f"Field '{self.field}' length must be at most {max_length}"

class PatternRule(ValidationRule):
    """Validates that a string field matches a regular expression pattern."""
    
    def __init__(self, field: str, pattern: str):
        import re
        super().__init__(field, "pattern", {"pattern": pattern})
        self.regex = re.compile(pattern)

    def validate(self, value: Any) -> bool:
        if not isinstance(value, str):
            return False
        return bool(self.regex.match(value))

    def get_error_message(self, value: Any) -> str:
        return f"Field '{self.field}' must match pattern {self.params['pattern']}" 