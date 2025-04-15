from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel
from .base import BaseValidator, ValidationRule
from .rules import RequiredRule, TypeRule, RangeRule, LengthRule, PatternRule

class ValidatorFactory:
    """Factory for creating validators with common rules."""
    
    @staticmethod
    def create_validator(
        schema: Optional[Type[BaseModel]] = None,
        rules: Optional[List[ValidationRule]] = None
    ) -> BaseValidator:
        """Create a validator with the given schema and rules."""
        return BaseValidator(schema)

    @staticmethod
    def create_required_validator(field: str) -> RequiredRule:
        """Create a rule that requires a field to be present."""
        return RequiredRule(field)

    @staticmethod
    def create_type_validator(field: str, expected_type: Union[type, tuple]) -> TypeRule:
        """Create a rule that validates a field's type."""
        return TypeRule(field, expected_type)

    @staticmethod
    def create_range_validator(
        field: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> RangeRule:
        """Create a rule that validates a numeric field's range."""
        return RangeRule(field, min_value, max_value)

    @staticmethod
    def create_length_validator(
        field: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> LengthRule:
        """Create a rule that validates a field's length."""
        return LengthRule(field, min_length, max_length)

    @staticmethod
    def create_pattern_validator(field: str, pattern: str) -> PatternRule:
        """Create a rule that validates a string field against a pattern."""
        return PatternRule(field, pattern)

    @staticmethod
    def create_email_validator(field: str) -> PatternRule:
        """Create a rule that validates an email address."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return PatternRule(field, email_pattern)

    @staticmethod
    def create_url_validator(field: str) -> PatternRule:
        """Create a rule that validates a URL."""
        url_pattern = r'^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$'
        return PatternRule(field, url_pattern)

    @staticmethod
    def create_phone_validator(field: str) -> PatternRule:
        """Create a rule that validates a phone number."""
        phone_pattern = r'^\+?1?\d{9,15}$'
        return PatternRule(field, phone_pattern) 