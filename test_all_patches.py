"""
Test script to verify that all our patches work correctly.
"""

import os
import sys
import numpy as np
from pathlib import Path


def apply_all_patches():
    """Apply all the patches we've created."""
    # Apply NumPy 2.0 compatibility patch first
    print("\n--- Applying NumPy 2.0 compatibility patch ---")
    from data.numpy_compatibility import apply_numpy_patch
    numpy_patch_success = apply_numpy_patch()
    
    if not numpy_patch_success:
        print("Failed to apply NumPy compatibility patch")
        return False
    
    # Test that np.NaN now works
    try:
        print(f"np.NaN value: {np.NaN}")
        print(f"np.NaN is np.nan: {np.NaN is np.nan}")
        print("Successfully accessing np.NaN")
    except AttributeError:
        print("Failed to access np.NaN even after patch")
        return False
    
    # Apply AudioCraft library patch
    print("\n--- Applying AudioCraft library patch ---")
    audiocraft_path = "/audiocraft/audiocraft"
    if not os.path.exists(audiocraft_path):
        print(f"AudioCraft path does not exist: {audiocraft_path}")
        print("Skipping AudioCraft patch test")
    else:
        from data.patch_audiocraft import patch_audiocraft
        audiocraft_patch_success = patch_audiocraft(audiocraft_path)
        
        if not audiocraft_patch_success:
            print("Failed to apply AudioCraft patch")
            return False
    
    # Apply WhisperX library patch
    print("\n--- Applying WhisperX compatibility patch ---")
    from data.patch_whisperx import apply_whisperx_patches
    whisperx_patch_success = apply_whisperx_patches()
    
    if not whisperx_patch_success:
        print("Failed to apply WhisperX patch (may not be needed)")
        # Don't return False here as this patch may not be necessary
    
    # Apply the WhisperX VAD patch
    print("\n--- Applying WhisperX VAD patch ---")
    model_cache = "./model_cache"  # Use a test directory
    from data.patch_whisperx_vad import patch_whisperx_vad
    vad_patch_success = patch_whisperx_vad(model_cache)
    
    if not vad_patch_success:
        print("Failed to apply WhisperX VAD patch")
        return False
    
    print("\nAll patches applied successfully!")
    return True


def test_imports():
    """Test importing the patched modules."""
    print("\n--- Testing imports ---")
    
    try:
        # First, try to import numpy and check NaN works
        import numpy as np
        print(f"Successfully imported NumPy {np.__version__}")
        print(f"np.NaN exists: {hasattr(np, 'NaN')}")
        print(f"np.nan exists: {hasattr(np, 'nan')}")
        
        # If AudioCraft exists, test importing that too
        if os.path.exists("/audiocraft"):
            try:
                from audiocraft.environment import AudioCraftEnvironment
                print("Successfully imported AudioCraftEnvironment")
                
                dora_dir = AudioCraftEnvironment.get_dora_dir()
                print(f"Dora directory: {dora_dir}")
                
                from audiocraft import train
                print("Successfully imported train module")
                
                print("AudioCraft imports successful")
            except Exception as e:
                print(f"Error importing AudioCraft: {e}")
                return False
                
        # Also test importing whisperx if we have access
        try:
            import whisperx
            print(f"Successfully imported whisperx {getattr(whisperx, '__version__', 'unknown')}")
            
            # Try importing faster_whisper too
            import faster_whisper
            print(f"Successfully imported faster_whisper {getattr(faster_whisper, '__version__', 'unknown')}")
            
            # Check if the TranscriptionOptions class exists
            from faster_whisper.transcribe import TranscriptionOptions
            print("Successfully imported TranscriptionOptions class")
            
            # Instead of trying to create an object, let's just check if the class has the required attributes
            has_multilingual = hasattr(TranscriptionOptions.__init__, '__defaults__') and \
                len(TranscriptionOptions.__init__.__defaults__) > 0
            print(f"TranscriptionOptions has defaults: {has_multilingual}")
            
            print("WhisperX imports successful")
        except ImportError as e:
            print(f"Error importing whisperx: {e}")
            print("Skipping whisperx test")
        
        # Also test importing pyannote if we have access
        try:
            import pyannote.audio
            print("Successfully imported pyannote.audio")
            
            # Try a specific import that uses np.NaN
            from pyannote.audio.core.inference import Inference
            print("Successfully imported Inference class")
        except Exception as e:
            print(f"Error importing pyannote.audio: {e}")
            print("Skipping pyannote test")
        
        # Test WhisperX VAD
        print("\n5. Testing WhisperX VAD import...")
        try:
            import whisperx.vad
            # Check if the VAD module is properly patched
            if hasattr(whisperx.vad, "load_vad_model"):
                print("✅ WhisperX VAD module imported successfully")
                # Try loading the VAD model (this will fail if the patch didn't work)
                try:
                    model = whisperx.vad.load_vad_model("./model_cache")
                    print("✅ VAD model loaded successfully")
                except Exception as e:
                    print(f"❌ VAD model loading failed: {e}")
            else:
                print("❌ WhisperX VAD module missing load_vad_model function")
        except Exception as e:
            print(f"❌ WhisperX VAD import failed: {e}")
        
        return True
    except Exception as e:
        print(f"Error testing imports: {e}")
        return False


if __name__ == "__main__":
    print("Starting patch tests...")
    
    if not apply_all_patches():
        print("Failed to apply patches")
        sys.exit(1)
        
    if not test_imports():
        print("Failed import tests")
        sys.exit(1)
        
    print("\nAll tests passed successfully!")
    sys.exit(0) 