import pytest
from src.validation.base import BaseValidator, ValidationResult
from src.validation.rules import RequiredRule, TypeRule, RangeRule, LengthRule, PatternRule
from src.validation.factory import ValidatorFactory
from src.validation.examples import UserSchema, validate_user

def test_validation_result():
    """Test ValidationResult class."""
    # Test valid result
    result = ValidationResult(is_valid=True, data={"key": "value"})
    assert result.is_valid
    assert result.data == {"key": "value"}
    assert not result.errors
    assert not result.warnings

    # Test invalid result
    result = ValidationResult(is_valid=False, errors=["Error 1"], warnings=["Warning 1"])
    assert not result.is_valid
    assert result.errors == ["Error 1"]
    assert result.warnings == ["Warning 1"]
    assert result.data is None

def test_base_validator():
    """Test BaseValidator class."""
    # Test without schema
    validator = BaseValidator()
    result = validator.validate({"key": "value"})
    assert result.is_valid
    assert result.data == {"key": "value"}

    # Test with schema
    validator = BaseValidator(UserSchema)
    valid_data = {
        "username": "john_doe",
        "email": "john@example.com"
    }
    result = validator.validate(valid_data)
    assert result.is_valid

    invalid_data = {
        "username": "jo",
        "email": "invalid-email"
    }
    result = validator.validate(invalid_data)
    assert not result.is_valid

def test_validation_rules():
    """Test validation rules."""
    # Test RequiredRule
    rule = RequiredRule("field")
    assert rule.validate("value")
    assert not rule.validate(None)

    # Test TypeRule
    rule = TypeRule("field", str)
    assert rule.validate("value")
    assert not rule.validate(123)

    # Test RangeRule
    rule = RangeRule("field", min_value=0, max_value=10)
    assert rule.validate(5)
    assert not rule.validate(-1)
    assert not rule.validate(11)

    # Test LengthRule
    rule = LengthRule("field", min_length=2, max_length=5)
    assert rule.validate("abc")
    assert not rule.validate("a")
    assert not rule.validate("abcdef")

    # Test PatternRule
    rule = PatternRule("field", r"^[a-z]+$")
    assert rule.validate("abc")
    assert not rule.validate("123")

def test_validator_factory():
    """Test ValidatorFactory class."""
    # Test create_validator
    validator = ValidatorFactory.create_validator(UserSchema)
    assert isinstance(validator, BaseValidator)

    # Test create_required_validator
    rule = ValidatorFactory.create_required_validator("field")
    assert isinstance(rule, RequiredRule)

    # Test create_type_validator
    rule = ValidatorFactory.create_type_validator("field", str)
    assert isinstance(rule, TypeRule)

    # Test create_range_validator
    rule = ValidatorFactory.create_range_validator("field", 0, 10)
    assert isinstance(rule, RangeRule)

    # Test create_length_validator
    rule = ValidatorFactory.create_length_validator("field", 2, 5)
    assert isinstance(rule, LengthRule)

    # Test create_pattern_validator
    rule = ValidatorFactory.create_pattern_validator("field", r"^[a-z]+$")
    assert isinstance(rule, PatternRule)

    # Test create_email_validator
    rule = ValidatorFactory.create_email_validator("email")
    assert isinstance(rule, PatternRule)
    assert rule.validate("user@example.com")
    assert not rule.validate("invalid-email")

    # Test create_url_validator
    rule = ValidatorFactory.create_url_validator("url")
    assert isinstance(rule, PatternRule)
    assert rule.validate("https://example.com")
    assert not rule.validate("not-a-url")

    # Test create_phone_validator
    rule = ValidatorFactory.create_phone_validator("phone")
    assert isinstance(rule, PatternRule)
    assert rule.validate("+1234567890")
    assert not rule.validate("123")

def test_user_validation():
    """Test user validation example."""
    # Test valid user
    valid_user = {
        "username": "john_doe",
        "email": "john@example.com",
        "age": 25,
        "phone": "+1234567890",
        "website": "https://example.com"
    }
    result = validate_user(valid_user)
    assert result["is_valid"]

    # Test invalid user
    invalid_user = {
        "username": "jo",  # Too short
        "email": "invalid-email",  # Invalid email
        "age": 5,  # Too young
        "phone": "123",  # Invalid phone
        "website": "not-a-url"  # Invalid URL
    }
    result = validate_user(invalid_user)
    assert not result["is_valid"]
    assert len(result["errors"]) > 0 