#!/usr/bin/env python3
"""
Script to check documentation links.
"""

import os
import re
import requests
from pathlib import Path
from typing import Dict, List, Set, Tuple
from urllib.parse import urljoin, urlparse

def find_markdown_files(docs_dir: str) -> List[str]:
    """Find all markdown files in the documentation directory."""
    markdown_files = []
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith('.md'):
                markdown_files.append(os.path.join(root, file))
    return markdown_files

def extract_links(file_path: str) -> Set[str]:
    """Extract all links from a markdown file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all markdown links [text](url)
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    links = re.finditer(link_pattern, content)
    
    # Extract URLs
    urls = set()
    for link in links:
        url = link.group(2)
        # Skip anchor links
        if not url.startswith('#'):
            urls.add(url)
    
    return urls

def validate_link(url: str, base_url: str = None) -> Tuple[bool, str]:
    """Validate a link and return (is_valid, error_message)."""
    try:
        # Handle relative URLs
        if base_url and not urlparse(url).netloc:
            url = urljoin(base_url, url)
        
        # Skip external links in CI
        if urlparse(url).netloc:
            return True, ""
        
        # Check if file exists
        file_path = Path(url)
        if not file_path.exists():
            return False, f"File not found: {url}"
        
        return True, ""
    
    except Exception as e:
        return False, str(e)

def check_documentation(docs_dir: str) -> Dict[str, List[str]]:
    """Check all documentation links and return errors."""
    errors = {}
    
    # Find all markdown files
    markdown_files = find_markdown_files(docs_dir)
    
    # Check each file
    for file_path in markdown_files:
        file_errors = []
        links = extract_links(file_path)
        
        for link in links:
            is_valid, error = validate_link(link, os.path.dirname(file_path))
            if not is_valid:
                file_errors.append(f"Invalid link '{link}': {error}")
        
        if file_errors:
            errors[file_path] = file_errors
    
    return errors

def main() -> None:
    """Main function to check documentation links."""
    docs_dir = "docs"
    
    print("Checking documentation links...")
    errors = check_documentation(docs_dir)
    
    if errors:
        print("\nFound invalid links:")
        for file_path, file_errors in errors.items():
            print(f"\n{file_path}:")
            for error in file_errors:
                print(f"  - {error}")
        exit(1)
    else:
        print("All documentation links are valid!")

if __name__ == "__main__":
    main() 