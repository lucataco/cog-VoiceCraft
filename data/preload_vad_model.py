"""
Preload and install the VAD model to ensure WhisperX works correctly without HTTP 301 errors
"""

import os
import sys
import shutil
import importlib.util
from pathlib import Path

def preload_vad_model(model_cache_dir="model_cache"):
    """
    Preload the VAD model before WhisperX is imported
    
    Args:
        model_cache_dir: Directory to cache the VAD model
        
    Returns:
        bool: True if successful, False otherwise
    """
    # First check if we need to create the directory
    os.makedirs(model_cache_dir, exist_ok=True)
    
    # Download the VAD model
    from data.download_vad_model import download_vad_model
    
    print(f"Preloading VAD model to {model_cache_dir}")
    download_success = download_vad_model(model_cache_dir)
    
    if not download_success:
        print("Failed to download VAD model")
        return False
    
    # Directly place the VAD model in the WhisperX default cache location
    # This ensures it will be found even if the patch fails
    try:
        vad_model_path = os.path.join(model_cache_dir, "silero_vad.onnx")
        whisperx_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "vad")
        os.makedirs(whisperx_cache_dir, exist_ok=True)
        whisperx_model_path = os.path.join(whisperx_cache_dir, "silero_vad.onnx")
        
        # Copy the model to the WhisperX cache directory
        if os.path.exists(vad_model_path) and os.path.getsize(vad_model_path) > 0:
            shutil.copy(vad_model_path, whisperx_model_path)
            print(f"VAD model placed directly in WhisperX cache at {whisperx_model_path}")
        else:
            print(f"Warning: VAD model not found at {vad_model_path}")
    except Exception as e:
        print(f"Warning: Failed to place VAD model in WhisperX cache: {e}")
    
    # Also try to apply the patch to WhisperX VAD module
    # But consider it optional for overall success
    try:
        from data.patch_whisperx_vad import patch_whisperx_vad
        
        print("Patching WhisperX VAD module")
        patch_success = patch_whisperx_vad(model_cache_dir)
        
        if not patch_success:
            print("Note: WhisperX VAD module patch didn't modify anything (may already be patched)")
        else:
            print("WhisperX VAD module patched successfully")
    except Exception as e:
        print(f"Warning: Failed to patch WhisperX VAD module: {e}")
        print("This is not critical as long as the model was copied to the cache")
    
    # Verify the model exists in the WhisperX cache
    if os.path.exists(whisperx_model_path) and os.path.getsize(whisperx_model_path) > 0:
        print("Successfully preloaded VAD model")
        return True
    else:
        print("Failed to prepare VAD model")
        return False

if __name__ == "__main__":
    model_cache_dir = sys.argv[1] if len(sys.argv) > 1 else "model_cache"
    success = preload_vad_model(model_cache_dir)
    sys.exit(0 if success else 1) 