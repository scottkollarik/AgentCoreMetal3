import os
from pathlib import Path
from typing import Optional

def ensure_directory_exists(path: str, base_dir: Optional[str] = None) -> str:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        path: The directory path to ensure exists
        base_dir: Optional base directory to resolve relative paths against
        
    Returns:
        The absolute path to the directory
    """
    if base_dir:
        path = os.path.join(base_dir, path)
    
    # Convert to absolute path
    abs_path = os.path.abspath(path)
    
    # Create directory if it doesn't exist
    os.makedirs(abs_path, exist_ok=True)
    
    return abs_path

def get_project_root() -> str:
    """Get the absolute path to the project root directory.
    
    Returns:
        The absolute path to the project root
    """
    # Assuming this file is in utils/ and project root is one level up
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 