#!/usr/bin/env python3
"""
Script to find all Python files containing API endpoints.
Excludes test files and provides detailed information about found endpoints.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

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

def analyze_api_files(project_dir: str) -> Dict[str, List[Dict]]:
    """Analyze all API files and return detailed information."""
    api_files = find_api_files(project_dir)
    api_info = {}
    
    for file_path in api_files:
        endpoints = extract_api_info(file_path)
        if endpoints:
            api_info[file_path] = endpoints
    
    return api_info

def print_api_summary(api_info: Dict[str, List[Dict]]) -> None:
    """Print a summary of found API endpoints."""
    total_files = len(api_info)
    total_endpoints = sum(len(endpoints) for endpoints in api_info.values())
    
    print(f"\nFound {total_endpoints} API endpoints in {total_files} files:")
    
    for file_path, endpoints in api_info.items():
        print(f"\n{file_path}:")
        for endpoint in endpoints:
            print(f"  {endpoint['method']} {endpoint['path']}")
            print(f"    Function: {endpoint['function']}")
            if endpoint['response_model']:
                print(f"    Response Model: {endpoint['response_model']}")
            if endpoint['docstring']:
                print(f"    Description: {endpoint['docstring'][:100]}...")

def main() -> None:
    """Main function to find and analyze API files."""
    project_dir = "python_ai_core"
    
    print("Searching for API endpoints...")
    api_info = analyze_api_files(project_dir)
    print_api_summary(api_info)

if __name__ == "__main__":
    main() 