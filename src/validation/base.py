from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, ValidationError

class ValidationResult:
    """Represents the result of a validation operation."""
    
    def __init__(
        self,
        is_valid: bool,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        data: Optional[Any] = None
    ):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.data = data

    def __bool__(self) -> bool:
        return self.is_valid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "data": self.data
        }

class BaseValidator:
    """Base class for all validators."""
    
    def __init__(self, schema: Optional[Type[BaseModel]] = None):
        self.schema = schema

    def validate(self, data: Any) -> ValidationResult:
        """Validate the given data against the schema."""
        if self.schema is None:
            return ValidationResult(is_valid=True, data=data)

        try:
            validated_data = self.schema.model_validate(data)
            return ValidationResult(is_valid=True, data=validated_data)
        except ValidationError as e:
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            return ValidationResult(is_valid=False, errors=errors)

    def validate_batch(self, data_list: List[Any]) -> List[ValidationResult]:
        """Validate a batch of data items."""
        return [self.validate(item) for item in data_list]

class ValidationRule:
    """Base class for validation rules."""
    
    def __init__(self, field: str, rule_type: str, params: Optional[Dict[str, Any]] = None):
        self.field = field
        self.rule_type = rule_type
        self.params = params or {}

    def validate(self, value: Any) -> bool:
        """Validate a single value against the rule."""
        raise NotImplementedError("Subclasses must implement validate()")

    def get_error_message(self, value: Any) -> str:
        """Get the error message for a failed validation."""
        raise NotImplementedError("Subclasses must implement get_error_message()") 