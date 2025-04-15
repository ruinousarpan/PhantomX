import pytest
from src.validation.base import ValidationResult
from src.validation.rules import (
    RequiredRule,
    TypeRule,
    RangeRule,
    LengthRule,
    PatternRule
)
from src.validation.examples import validate_user

def test_required_rule():
    rule = RequiredRule("field")
    assert rule.validate("value") is True
    assert rule.validate(None) is False
    assert rule.get_error_message(None) == "Field 'field' is required"

def test_type_rule():
    rule = TypeRule("field", str)
    assert rule.validate("value") is True
    assert rule.validate(123) is False
    assert rule.get_error_message(123) == "Field 'field' must be of type str"

def test_range_rule():
    rule = RangeRule("field", min_value=0, max_value=100)
    assert rule.validate(50) is True
    assert rule.validate(-1) is False
    assert rule.validate(101) is False
    assert rule.get_error_message(-1) == "Field 'field' must be between 0 and 100"

def test_length_rule():
    rule = LengthRule("field", min_length=3, max_length=10)
    assert rule.validate("value") is True
    assert rule.validate("ab") is False
    assert rule.validate("toolongvalue") is False
    assert rule.get_error_message("ab") == "Field 'field' length must be between 3 and 10"

def test_pattern_rule():
    rule = PatternRule("field", r"^[A-Z]+$")
    assert rule.validate("ABC") is True
    assert rule.validate("abc") is False
    assert rule.get_error_message("abc") == "Field 'field' must match pattern ^[A-Z]+$"

def test_email_validator():
    from src.validation.factory import ValidatorFactory
    rule = ValidatorFactory.create_email_validator("email")
    assert rule.validate("test@example.com") is True
    assert rule.validate("invalid-email") is False

def test_url_validator():
    from src.validation.factory import ValidatorFactory
    rule = ValidatorFactory.create_url_validator("url")
    assert rule.validate("https://example.com") is True
    assert rule.validate("not-a-url") is False

def test_phone_validator():
    from src.validation.factory import ValidatorFactory
    rule = ValidatorFactory.create_phone_validator("phone")
    assert rule.validate("+1234567890") is True
    assert rule.validate("123") is False

def test_user_validation():
    # Test valid user
    valid_user = {
        "username": "john_doe",
        "email": "john@example.com",
        "age": 25,
        "phone": "+1234567890",
        "website": "https://example.com"
    }
    result = validate_user(valid_user)
    assert result["is_valid"] is True
    assert len(result["errors"]) == 0

    # Test invalid user
    invalid_user = {
        "username": "jo",  # Too short
        "email": "invalid-email",  # Invalid email
        "age": 5,  # Too young
        "phone": "123",  # Invalid phone
        "website": "not-a-url"  # Invalid URL
    }
    result = validate_user(invalid_user)
    assert result["is_valid"] is False
    assert len(result["errors"]) > 0 