from typing import List, Optional
from pydantic import BaseModel, EmailStr
from .factory import ValidatorFactory

class UserSchema(BaseModel):
    """Example user schema with validation rules."""
    username: str
    email: EmailStr
    age: Optional[int] = None
    phone: Optional[str] = None
    website: Optional[str] = None

def create_user_validator() -> List[ValidatorFactory]:
    """Create a list of validation rules for the user schema."""
    rules = [
        ValidatorFactory.create_required_validator("username"),
        ValidatorFactory.create_length_validator("username", min_length=3, max_length=50),
        ValidatorFactory.create_email_validator("email"),
        ValidatorFactory.create_range_validator("age", min_value=13, max_value=120),
        ValidatorFactory.create_phone_validator("phone"),
        ValidatorFactory.create_url_validator("website")
    ]
    return rules

def validate_user(data: dict) -> dict:
    """Validate user data using the schema and rules."""
    # Create a validator with the schema
    validator = ValidatorFactory.create_validator(UserSchema)
    
    # Validate the data
    result = validator.validate(data)
    
    # If the schema validation passes, apply additional rules
    if result.is_valid:
        rules = create_user_validator()
        for rule in rules:
            field_value = data.get(rule.field)
            if field_value is not None and not rule.validate(field_value):
                result.is_valid = False
                result.errors.append(rule.get_error_message(field_value))
    
    return result.to_dict()

# Example usage
if __name__ == "__main__":
    # Valid user data
    valid_user = {
        "username": "john_doe",
        "email": "john@example.com",
        "age": 25,
        "phone": "+1234567890",
        "website": "https://example.com"
    }
    
    # Invalid user data
    invalid_user = {
        "username": "jo",  # Too short
        "email": "invalid-email",  # Invalid email
        "age": 5,  # Too young
        "phone": "123",  # Invalid phone
        "website": "not-a-url"  # Invalid URL
    }
    
    # Test validation
    print("Validating valid user:")
    print(validate_user(valid_user))
    
    print("\nValidating invalid user:")
    print(validate_user(invalid_user)) 