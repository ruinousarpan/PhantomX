#!/usr/bin/env python3
"""
Script to help test API endpoints.
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
        
        endpoints.append({
            'method': method,
            'path': path,
            'function': func_name,
            'parameters': params,
            'docstring': docstring
        })
    
    return endpoints

def test_endpoint(base_url: str, endpoint: Dict) -> Tuple[bool, str]:
    """Test an API endpoint."""
    url = f"{base_url.rstrip('/')}/{endpoint['path'].lstrip('/')}"
    method = endpoint['method'].lower()
    
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
            return False, f"Unsupported HTTP method: {method}"
        
        response.raise_for_status()
        return True, f"Success: {response.status_code}"
    
    except RequestException as e:
        return False, f"Error: {str(e)}"

def test_endpoints(project_dir: str, base_url: str) -> None:
    """Test all API endpoints."""
    api_files = find_api_files(project_dir)
    
    for file_path in api_files:
        endpoints = extract_api_info(file_path)
        if not endpoints:
            continue
        
        file_name = os.path.basename(file_path)
        print(f"\nTesting endpoints in {file_name}:")
        
        for endpoint in endpoints:
            success, message = test_endpoint(base_url, endpoint)
            status = "✓" if success else "✗"
            print(f"{status} {endpoint['method']} {endpoint['path']}: {message}")

def main() -> None:
    """Main function to test API endpoints."""
    project_dir = "python_ai_core"
    base_url = "http://localhost:8000"  # Change this to your API base URL
    
    print("Testing API endpoints...")
    test_endpoints(project_dir, base_url)
    print("\nAPI testing complete!")

if __name__ == "__main__":
    main() 