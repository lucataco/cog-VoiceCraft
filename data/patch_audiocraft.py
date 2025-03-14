"""
Script to patch the AudioCraft library with our modified files.
This avoids AWS key requirements and other external dependencies.
"""

import os
import shutil
import sys
from pathlib import Path


def patch_audiocraft(audiocraft_path):
    """
    Patch the AudioCraft library with our modified files.
    
    Args:
        audiocraft_path: Path to the AudioCraft library
    """
    # Get the directory of this script
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    patched_dir = current_dir / "patched"
    
    # Make sure the provided path exists
    audiocraft_path = Path(audiocraft_path)
    if not audiocraft_path.exists():
        print(f"Error: AudioCraft path does not exist: {audiocraft_path}")
        return False
    
    # Create patched directory if it doesn't exist
    if not patched_dir.exists():
        print(f"Error: Patched directory not found at {patched_dir}")
        return False
        
    # Copy our patched environment.py
    env_target = audiocraft_path / "environment.py"
    env_source = patched_dir / "environment.py"
    if not env_source.exists():
        print(f"Error: Patched environment.py not found at {env_source}")
        return False
    
    print(f"Copying {env_source} to {env_target}")
    shutil.copy(env_source, env_target)
    
    # Copy our patched train.py
    train_target = audiocraft_path / "train.py"
    train_source = patched_dir / "train.py"
    if not train_source.exists():
        print(f"Error: Patched train.py not found at {train_source}")
        return False
    
    print(f"Copying {train_source} to {train_target}")
    shutil.copy(train_source, train_target)
    
    # Create directories for temporary files
    dora_dir = Path("/tmp/dora")
    ref_dir = Path("/tmp/reference")
    os.makedirs(dora_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
    
    print("AudioCraft library successfully patched!")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python patch_audiocraft.py <audiocraft_path>")
        sys.exit(1)
    
    success = patch_audiocraft(sys.argv[1])
    sys.exit(0 if success else 1) 