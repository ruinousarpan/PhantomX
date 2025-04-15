#!/usr/bin/env python3
"""
Script to help validate API endpoints.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import requests
from requests.exceptions import RequestException

def find_api_files(project_dir: str) -> List[str]:
    """Find all Python files that contain API endpoints."""
    api_files = []
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('test_'):
                api_files.append(os.path.join(root, file))
    return api_files

def extract_api_info(file_path: str) -> List[Dict]:
    """Extract API information from a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find FastAPI route decorators and their associated functions
    route_pattern = r'@(?:app|router)\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']\)\s+def\s+([^\(]+)\(([^\)]*)\)'
    routes = re.finditer(route_pattern, content)
    
    endpoints = []
    for route in routes:
        method = route.group(1).upper()
        path = route.group(2)
        func_name = route.group(3).strip()
        params = route.group(4).strip()
        
        # Find function docstring
        doc_pattern = rf'def\s+{func_name}\s*\([^\)]*\):\s*"""([^"]*)"""'
        doc_match = re.search(doc_pattern, content, re.DOTALL)
        docstring = doc_match.group(1).strip() if doc_match else ""
        
        # Find response model if specified
        response_pattern = rf'@(?:app|router)\.{method.lower()}\(.*?response_model=([^,\)]+)'
        response_match = re.search(response_pattern, content)
        response_model = response_match.group(1).strip() if response_match else None
        
        endpoints.append({
            'method': method,
            'path': path,
            'function': func_name,
            'parameters': params,
            'docstring': docstring,
            'response_model': response_model
        })
    
    return endpoints

def validate_endpoint(base_url: str, endpoint: Dict) -> Tuple[bool, List[str]]:
    """Validate an API endpoint."""
    url = f"{base_url.rstrip('/')}/{endpoint['path'].lstrip('/')}"
    method = endpoint['method'].lower()
    issues = []
    
    # Check if endpoint has docstring
    if not endpoint['docstring']:
        issues.append("Missing docstring")
    
    # Check if endpoint has response model
    if not endpoint['response_model']:
        issues.append("Missing response model")
    
    # Test endpoint
    try:
        if method == 'get':
            response = requests.get(url)
        elif method == 'post':
            response = requests.post(url)
        elif method == 'put':
            response = requests.put(url)
        elif method == 'delete':
            response = requests.delete(url)
        elif method == 'patch':
            response = requests.patch(url)
        else:
            issues.append(f"Unsupported HTTP method: {method}")
            return False, issues
        
        response.raise_for_status()
        
        # Validate response against schema if available
        if endpoint['response_model']:
            try:
                schema = get_response_schema(endpoint['response_model'])
                validate_response(response.json(), schema)
            except Exception as e:
                issues.append(f"Response validation failed: {str(e)}")
    
    except RequestException as e:
        issues.append(f"Request failed: {str(e)}")
    
    return len(issues) == 0, issues

def get_response_schema(model_name: str) -> Dict:
    """Get the JSON schema for a response model."""
    # This is a placeholder - you would need to implement
    # the actual schema retrieval logic based on your models
    return {}

def validate_response(data: Dict, schema: Dict) -> None:
    """Validate response data against a schema."""
    # This is a placeholder - you would need to implement
    # the actual validation logic based on your schema
    pass

def validate_endpoints(project_dir: str, base_url: str) -> None:
    """Validate all API endpoints."""
    api_files = find_api_files(project_dir)
    
    for file_path in api_files:
        endpoints = extract_api_info(file_path)
        if not endpoints:
            continue
        
        file_name = os.path.basename(file_path)
        print(f"\nValidating endpoints in {file_name}:")
        
        for endpoint in endpoints:
            is_valid, issues = validate_endpoint(base_url, endpoint)
            status = "✓" if is_valid else "✗"
            print(f"{status} {endpoint['method']} {endpoint['path']}")
            if issues:
                for issue in issues:
                    print(f"  - {issue}")

def main() -> None:
    """Main function to validate API endpoints."""
    project_dir = "python_ai_core"
    base_url = "http://localhost:8000"  # Change this to your API base URL
    
    print("Validating API endpoints...")
    validate_endpoints(project_dir, base_url)
    print("\nAPI validation complete!")

if __name__ == "__main__":
    main() 