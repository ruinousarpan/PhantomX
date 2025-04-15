#!/usr/bin/env python3
"""
Script to validate API documentation against the codebase.
Checks if all API endpoints are properly documented.
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

def find_doc_files(docs_dir: str) -> List[str]:
    """Find all markdown documentation files."""
    doc_files = []
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith('.md'):
                doc_files.append(os.path.join(root, file))
    return doc_files

def extract_endpoints(file_path: str) -> Set[Tuple[str, str]]:
    """Extract API endpoints from a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find FastAPI route decorators
    route_pattern = r'@(?:app|router)\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']'
    routes = re.finditer(route_pattern, content)
    
    endpoints = set()
    for route in routes:
        method = route.group(1).upper()
        path = route.group(2)
        endpoints.add((method, path))
    
    return endpoints

def extract_documented_endpoints(file_path: str) -> Set[Tuple[str, str]]:
    """Extract documented endpoints from a markdown file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find documented endpoints
    doc_pattern = r'## (GET|POST|PUT|DELETE|PATCH) ([^\n]+)'
    docs = re.finditer(doc_pattern, content)
    
    endpoints = set()
    for doc in docs:
        method = doc.group(1)
        path = doc.group(2)
        endpoints.add((method, path))
    
    return endpoints

def validate_documentation(project_dir: str, docs_dir: str) -> Dict[str, List[str]]:
    """Validate API documentation against the codebase."""
    api_files = find_api_files(project_dir)
    doc_files = find_doc_files(docs_dir)
    
    # Collect all endpoints from code
    code_endpoints = set()
    for file_path in api_files:
        code_endpoints.update(extract_endpoints(file_path))
    
    # Collect all documented endpoints
    doc_endpoints = set()
    for file_path in doc_files:
        doc_endpoints.update(extract_documented_endpoints(file_path))
    
    # Find missing and extra documentation
    missing_docs = code_endpoints - doc_endpoints
    extra_docs = doc_endpoints - code_endpoints
    
    return {
        'missing_documentation': sorted(missing_docs),
        'extra_documentation': sorted(extra_docs)
    }

def main() -> None:
    """Main function to validate API documentation."""
    project_dir = "python_ai_core"
    docs_dir = "docs/api"
    
    print("Validating API documentation...")
    
    results = validate_documentation(project_dir, docs_dir)
    
    if results['missing_documentation']:
        print("\nMissing documentation for the following endpoints:")
        for method, path in results['missing_documentation']:
            print(f"- {method} {path}")
    
    if results['extra_documentation']:
        print("\nDocumentation exists for non-existent endpoints:")
        for method, path in results['extra_documentation']:
            print(f"- {method} {path}")
    
    if not results['missing_documentation'] and not results['extra_documentation']:
        print("All API endpoints are properly documented!")
    else:
        # Exit with error code if there are issues
        exit(1)

if __name__ == "__main__":
    main() 