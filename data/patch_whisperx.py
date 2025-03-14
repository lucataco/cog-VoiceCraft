"""
Script to patch the WhisperX library to ensure compatibility with faster-whisper.
"""

import os
import re
import sys
import importlib
from pathlib import Path


def find_whisperx_path():
    """
    Find the installation path of WhisperX.
    
    Returns:
        Path: Path to the whisperx package directory, or None if not found
    """
    try:
        import whisperx
        whisperx_path = Path(whisperx.__file__).parent
        return whisperx_path
    except ImportError:
        print("WhisperX is not installed")
        return None


def patch_whisperx_asr():
    """
    Patch the WhisperX ASR module to handle changing APIs in faster-whisper.
    
    This handles the case where newer faster-whisper versions require additional
    parameters (multilingual, hotwords) in TranscriptionOptions.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        whisperx_path = find_whisperx_path()
        if not whisperx_path:
            return False
            
        asr_file = whisperx_path / "asr.py"
        
        if not asr_file.exists():
            print(f"Could not find WhisperX ASR module at {asr_file}")
            return False
            
        print(f"Found WhisperX ASR module at {asr_file}")
        
        # Read the file
        with open(asr_file, "r") as f:
            content = f.read()
            
        # Check if we need to modify the code
        if "def load_model(" in content and 'asr_options = {"beam_size":' in content:
            # Add multilingual and hotwords to the default ASR options
            modified_content = content.replace(
                'asr_options = {"beam_size":',
                'asr_options = {"multilingual": False, "hotwords": [], "beam_size":'
            )
            
            # If the file content was modified, write it back
            if modified_content != content:
                with open(asr_file, "w") as f:
                    f.write(modified_content)
                print("Successfully patched WhisperX ASR module")
                return True
            else:
                print("WhisperX ASR module already patched or uses a different structure")
                return False
        else:
            print("Could not find the expected code pattern in WhisperX ASR module")
            return False
            
    except Exception as e:
        print(f"Error patching WhisperX: {e}")
        return False


def apply_whisperx_patches():
    """
    Apply all WhisperX patches.
    
    Returns:
        bool: True if all patches were successful, False otherwise
    """
    asr_patch_success = patch_whisperx_asr()
    
    # If we add more patches, check them all here
    
    return asr_patch_success


if __name__ == "__main__":
    success = apply_whisperx_patches()
    sys.exit(0 if success else 1) 