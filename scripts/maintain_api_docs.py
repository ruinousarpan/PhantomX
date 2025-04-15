#!/usr/bin/env python3
"""
Script to help maintain API documentation.
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
        
        endpoints.append({
            'method': method,
            'path': path,
            'function': func_name,
            'parameters': params,
            'docstring': docstring
        })
    
    return endpoints

def generate_markdown(endpoints: List[Dict], file_path: str) -> str:
    """Generate markdown documentation for API endpoints."""
    file_name = os.path.basename(file_path)
    module_name = os.path.splitext(file_name)[0]
    
    markdown = f"# {module_name.replace('_', ' ').title()} API\n\n"
    
    for endpoint in endpoints:
        markdown += f"## {endpoint['method']} {endpoint['path']}\n\n"
        
        if endpoint['docstring']:
            markdown += f"{endpoint['docstring']}\n\n"
        
        if endpoint['parameters']:
            markdown += "### Parameters\n\n"
            for param in endpoint['parameters'].split(','):
                param = param.strip()
                if param:
                    markdown += f"- `{param}`\n"
            markdown += "\n"
        
        markdown += "---\n\n"
    
    return markdown

def update_documentation(project_dir: str, docs_dir: str) -> None:
    """Update API documentation for all endpoints."""
    api_files = find_api_files(project_dir)
    
    for file_path in api_files:
        endpoints = extract_api_info(file_path)
        if not endpoints:
            continue
        
        # Generate documentation
        markdown = generate_markdown(endpoints, file_path)
        
        # Create docs directory if it doesn't exist
        os.makedirs(docs_dir, exist_ok=True)
        
        # Write documentation file
        file_name = os.path.basename(file_path)
        doc_file = os.path.join(docs_dir, f"{os.path.splitext(file_name)[0]}.md")
        
        with open(doc_file, 'w') as f:
            f.write(markdown)
        
        print(f"Updated documentation for {file_name}")

def main() -> None:
    """Main function to update API documentation."""
    project_dir = "python_ai_core"
    docs_dir = "docs/api"
    
    print("Updating API documentation...")
    update_documentation(project_dir, docs_dir)
    print("Documentation update complete!")

if __name__ == "__main__":
    main() 