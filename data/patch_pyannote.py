"""
Script to patch the pyannote.audio library to fix NumPy 2.0 compatibility issues.
"""

import os
import re
import sys
from pathlib import Path


def patch_pyannote(site_packages_path=None):
    """
    Patch pyannote.audio library to replace np.NaN with np.nan.
    
    Args:
        site_packages_path: Path to the site-packages directory where pyannote.audio is installed.
                           If None, it will attempt to find it automatically.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if site_packages_path is None:
            # Try to find site-packages automatically
            import site
            site_packages_paths = site.getsitepackages()
            
            if not site_packages_paths:
                print("Error: Could not find site-packages directory")
                return False
                
            site_packages_path = site_packages_paths[0]
        
        # Construct path to the inference.py file
        inference_path = Path(site_packages_path) / "pyannote" / "audio" / "core" / "inference.py"
        
        if not inference_path.exists():
            print(f"Error: Could not find pyannote.audio inference.py at {inference_path}")
            return False
        
        print(f"Found pyannote.audio inference.py at {inference_path}")
        
        # Read the file content
        with open(inference_path, "r") as f:
            content = f.read()
        
        # Replace np.NaN with np.nan
        new_content = re.sub(r'np\.NaN', 'np.nan', content)
        
        # Write the file back
        with open(inference_path, "w") as f:
            f.write(new_content)
        
        print("Successfully patched pyannote.audio to fix NumPy 2.0 compatibility")
        return True
        
    except Exception as e:
        print(f"Error patching pyannote.audio: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        patch_pyannote(sys.argv[1])
    else:
        patch_pyannote() 