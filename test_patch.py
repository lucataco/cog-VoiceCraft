"""
Test script to verify that our patched audiocraft modules work correctly.
"""

import os
import sys
from pathlib import Path

# Apply the patch first
def patch_audiocraft():
    print("Patching AudioCraft library to bypass AWS key requirement...")
    audiocraft_path = "/audiocraft/audiocraft"
    
    # Run the patching script
    from data.patch_audiocraft import patch_audiocraft as apply_patch
    success = apply_patch(audiocraft_path)
    
    if not success:
        print("Warning: Failed to patch AudioCraft library")
        return False
    else:
        print("AudioCraft library patched successfully")
        
    # Create necessary directories
    os.makedirs("/tmp/dora", exist_ok=True)
    os.makedirs("/tmp/dora/xps", exist_ok=True)
    os.makedirs("/tmp/reference", exist_ok=True)
    return True

# Test importing the required modules
def test_imports():
    try:
        # First test importing environment directly
        from audiocraft.environment import AudioCraftEnvironment
        print("Successfully imported AudioCraftEnvironment")
        
        # Test getting dora dir
        dora_dir = AudioCraftEnvironment.get_dora_dir()
        print(f"Dora directory: {dora_dir}")
        
        # Test importing train
        from audiocraft import train
        print("Successfully imported train module")
        
        # Test accessing dora.dir
        xps_dir = train.main.dora.dir
        print(f"XPS directory: {xps_dir}")
        
        return True
    except Exception as e:
        print(f"Error testing imports: {e}")
        return False

# Test the AudioTokenizer class
def test_tokenizer():
    try:
        from data.tokenizer import AudioTokenizer
        print("Successfully imported AudioTokenizer")
        
        # Don't actually create the tokenizer as it requires model files
        # Just verify the import works
        
        return True
    except Exception as e:
        print(f"Error testing tokenizer: {e}")
        return False

if __name__ == "__main__":
    if not patch_audiocraft():
        print("Failed to patch AudioCraft")
        sys.exit(1)
        
    if not test_imports():
        print("Failed import tests")
        sys.exit(1)
        
    if not test_tokenizer():
        print("Failed tokenizer test")
        sys.exit(1)
        
    print("All tests passed successfully!")
    sys.exit(0) 