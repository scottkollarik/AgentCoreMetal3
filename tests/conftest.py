import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up any test-specific environment variables
os.environ["TESTING"] = "true" 