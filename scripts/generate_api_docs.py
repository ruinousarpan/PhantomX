#!/usr/bin/env python3
"""
Script to generate API documentation from the codebase.
"""

import os
import re
import inspect
from pathlib import Path
from typing import Dict, List, Tuple

def find_api_files(project_dir: str) -> List[str]:
    """Find all Python files that contain API endpoints."""
    api_files = []
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('test_'):
                api_files.append(os.path.join(root, file))
    return api_files

def extract_api_info(file_path: str) -> List[Dict]:
    """Extract API endpoint information from a file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find FastAPI route decorators
    route_pattern = r'@(?:app|router)\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']'
    routes = re.finditer(route_pattern, content)
    
    endpoints = []
    for route in routes:
        method = route.group(1).upper()
        path = route.group(2)
        
        # Find the function definition following the decorator
        func_pattern = rf'@(?:app|router)\.{method.lower()}\("{path}"\)\s+def\s+(\w+)'
        func_match = re.search(func_pattern, content)
        
        if func_match:
            func_name = func_match.group(1)
            
            # Find the function's docstring
            doc_pattern = rf'def\s+{func_name}\(([^)]*)\):\s*"""([^"]*)"""'
            doc_match = re.search(doc_pattern, content, re.DOTALL)
            
            docstring = doc_match.group(2).strip() if doc_match else ""
            params = doc_match.group(1).strip() if doc_match else ""
            
            # Find response model if specified
            response_pattern = rf'@(?:app|router)\.{method.lower()}\(.*?response_model=([^,\)]+)'
            response_match = re.search(response_pattern, content)
            response_model = response_match.group(1).strip() if response_match else None
            
            endpoints.append({
                'method': method,
                'path': path,
                'name': func_name,
                'params': params,
                'docstring': docstring,
                'response_model': response_model
            })
    
    return endpoints

def generate_markdown(endpoints: List[Dict], output_dir: str) -> None:
    """Generate markdown documentation for API endpoints."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group endpoints by file
    endpoints_by_file = {}
    for endpoint in endpoints:
        file_name = os.path.basename(endpoint['file'])
        if file_name not in endpoints_by_file:
            endpoints_by_file[file_name] = []
        endpoints_by_file[file_name].append(endpoint)
    
    # Generate documentation for each file
    for file_name, file_endpoints in endpoints_by_file.items():
        output_file = os.path.join(output_dir, f"{file_name.replace('.py', '.md')}")
        
        with open(output_file, 'w') as f:
            f.write(f"# API Documentation: {file_name}\n\n")
            
            for endpoint in file_endpoints:
                f.write(f"## {endpoint['method']} {endpoint['path']}\n\n")
                f.write(f"**Function:** `{endpoint['name']}`\n\n")
                
                if endpoint['params']:
                    f.write("**Parameters:**\n")
                    f.write("```python\n")
                    f.write(endpoint['params'])
                    f.write("\n```\n\n")
                
                if endpoint['response_model']:
                    f.write(f"**Response Model:** `{endpoint['response_model']}`\n\n")
                
                if endpoint['docstring']:
                    f.write("**Description:**\n")
                    f.write(endpoint['docstring'])
                    f.write("\n\n")
                
                f.write("---\n\n")

def main() -> None:
    """Main function to generate API documentation."""
    project_dir = "python_ai_core"
    output_dir = "docs/api"
    
    print("Generating API documentation...")
    
    # Find API files
    api_files = find_api_files(project_dir)
    
    # Extract and generate documentation
    all_endpoints = []
    for file_path in api_files:
        endpoints = extract_api_info(file_path)
        for endpoint in endpoints:
            endpoint['file'] = file_path
        all_endpoints.extend(endpoints)
    
    # Generate markdown documentation
    generate_markdown(all_endpoints, output_dir)
    
    print(f"Documented {len(all_endpoints)} endpoints in {len(api_files)} files")

if __name__ == "__main__":
    main() 