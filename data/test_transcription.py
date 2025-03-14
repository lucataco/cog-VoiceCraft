#!/usr/bin/env python
"""
Test script for WhisperX transcription
This script tests the WhisperX transcription functionality with various fallback mechanisms
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from predict import patch_whisperx, load_model, WhisperxModel

def generate_test_audio(duration=3, sample_rate=16000, output_path="test_audio.wav"):
    """Generate a simple test audio file with silence and a sine wave"""
    try:
        import soundfile as sf
        # Create a 3-second audio file with silence
        audio = np.zeros(int(duration * sample_rate))
        
        # Add a sine wave in the middle (1kHz tone)
        t = np.linspace(0, 1, int(sample_rate))
        sine_wave = np.sin(2 * np.pi * 1000 * t) * 0.5
        
        # Put sine wave in the middle of the file
        start_idx = int(sample_rate * (duration - 1) / 2)
        end_idx = start_idx + len(sine_wave)
        audio[start_idx:end_idx] = sine_wave
        
        # Save the audio file
        sf.write(output_path, audio, sample_rate)
        return output_path
    except ImportError:
        print("soundfile package not installed, cannot generate test audio")
        return None

def test_transcription():
    """Test the WhisperX transcription functionality"""
    print("Testing WhisperX transcription...")
    
    # Try to patch WhisperX
    patch_result = patch_whisperx()
    print(f"Patch result: {patch_result}")
    
    # Generate test audio if possible
    audio_path = generate_test_audio()
    if not audio_path:
        # Use a dummy path for testing error handling
        audio_path = "nonexistent_audio.wav"
        print(f"Using dummy audio path: {audio_path}")
    
    try:
        # Load the whisper model
        print("Loading WhisperX model...")
        whisper_model = load_model()
        
        # Create the WhisperX model
        model = WhisperxModel(whisper_model)
        
        # Test transcription
        print(f"Transcribing audio: {audio_path}")
        try:
            result = model.transcribe(audio_path)
            print(f"Transcription result: {result}")
            return True
        except Exception as e:
            print(f"Transcription failed with error: {e}")
            return False
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False

def test_dummy_vad():
    """Test the dummy VAD model"""
    from predict import DummyVADModel
    import torch
    
    print("Testing DummyVADModel...")
    
    # Create a dummy audio tensor
    audio = torch.zeros(16000 * 3)  # 3 seconds of audio at 16kHz
    
    # Create a DummyVADModel
    model = DummyVADModel()
    
    # Test the __call__ method
    try:
        result = model(audio)
        print(f"DummyVADModel result: {result}")
        print(f"Result type: {type(result)}")
        print(f"Speech_probs shape: {result.speech_probs.shape}")
        return True
    except Exception as e:
        print(f"DummyVADModel test failed with error: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test WhisperX transcription")
    parser.add_argument("--test-vad", action="store_true", help="Test the dummy VAD model")
    args = parser.parse_args()
    
    if args.test_vad:
        success = test_dummy_vad()
    else:
        success = test_transcription()
    
    if success:
        print("Test completed successfully!")
        sys.exit(0)
    else:
        print("Test failed!")
        sys.exit(1) 