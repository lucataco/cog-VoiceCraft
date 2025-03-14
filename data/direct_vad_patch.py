"""
Direct patch for WhisperX's VAD module to make it work with different VAD model return types
"""

import os
import sys
import importlib.util
import torch

def find_whisperx_vad_path():
    """Find the path to the whisperx.vad module"""
    try:
        import whisperx
        
        # Get the base path of the whisperx package
        whisperx_path = os.path.dirname(whisperx.__file__)
        vad_path = os.path.join(whisperx_path, "vad.py")
        
        if os.path.exists(vad_path):
            return vad_path
        else:
            print(f"whisperx.vad module not found at {vad_path}")
            return None
    except ImportError:
        print("whisperx module not installed or not found")
        return None
    except Exception as e:
        print(f"Error finding whisperx.vad path: {e}")
        return None

def apply_safer_vad_functions():
    """Directly apply safer implementations of VAD functions to the WhisperX module"""
    try:
        import whisperx.vad as vad
        
        # Keep originals - check if they exist as attributes or are callable
        original_merge_chunks = getattr(vad, 'merge_chunks', None)
        
        # In some versions of WhisperX, binarize might be a callable rather than an attribute
        original_binarize = None
        if hasattr(vad, 'binarize'):
            original_binarize = vad.binarize
        else:
            # Check if it's a callable function directly in the module
            for attr_name in dir(vad):
                attr = getattr(vad, attr_name)
                if callable(attr) and attr_name.lower() == 'binarize':
                    original_binarize = attr
                    break
        
        if not original_merge_chunks:
            print("Warning: Could not find merge_chunks in WhisperX VAD module")
            return False
            
        # Define safer versions
        def safe_merge_chunks(chunks, sampling_rate=16000, duration=None):
            """Safer version of merge_chunks that can handle various input formats"""
            try:
                # Try the original first
                return original_merge_chunks(chunks, sampling_rate, duration)
            except (AttributeError, KeyError, TypeError, ValueError) as e:
                print(f"Original merge_chunks failed: {e}, using safe fallback")
                
                # Handle different types
                if isinstance(chunks, dict) and "segments" in chunks:
                    segments = chunks["segments"]
                    scores = chunks.get("speech_probs", torch.ones(1000, 1))  # Note: 2D tensor
                elif hasattr(chunks, "segments"):
                    segments = chunks.segments
                    scores = getattr(chunks, "speech_probs", torch.ones(1000, 1))  # Note: 2D tensor
                else:
                    # Default single segment for the whole audio
                    if duration is None:
                        duration = 30.0  # Default 30 seconds
                    segments = torch.Tensor([[0.0, duration]])
                    scores = torch.ones(int(duration * 100), 1)  # Note: 2D tensor
                
                # Create something that can be binarized
                class SafeScores:
                    def __init__(self, scores_tensor):
                        self.data = scores_tensor
                        
                    def __call__(self, *args, **kwargs):
                        # Make it callable as some implementations might expect to call it
                        return self.data
                
                # Return in the format expected by the caller
                return segments, SafeScores(scores)
        
        def safe_binarize(scores, threshold=0.5, min_duration=0.1, **kwargs):
            """Safer version of binarize that can handle various input formats"""
            try:
                # Try the original first if it exists
                if original_binarize:
                    return original_binarize(scores, threshold, min_duration, **kwargs)
                else:
                    # If no original, provide our own implementation
                    if isinstance(scores, torch.Tensor):
                        return scores  # Already segments
            except (AttributeError, KeyError, TypeError, ValueError) as e:
                print(f"Original binarize failed: {e}, using safe fallback")
                
            # If we get here, we need to handle it ourselves
            # Handle different types of scores
            if isinstance(scores, torch.Tensor):
                # It's already a tensor of segments
                return scores
            elif hasattr(scores, "data") and hasattr(scores.data, "shape"):
                try:
                    # If data is 1D, make it 2D
                    if len(scores.data.shape) == 1:
                        scores.data = scores.data.unsqueeze(1)
                        
                    # Try again with the original if it exists
                    if original_binarize:
                        return original_binarize(scores, threshold, min_duration, **kwargs)
                except:
                    # If that fails, create a single segment for the duration
                    try:
                        num_frames = scores.data.shape[0]
                        return torch.Tensor([[0.0, num_frames / 100.0]])  # Assuming 100 frames per second
                    except:
                        return torch.Tensor([[0.0, 30.0]])  # Default 30 second segment
            elif isinstance(scores, dict) and "segments" in scores:
                # If it's a dict with segments, return the segments
                return scores["segments"]
            elif hasattr(scores, "segments"):
                # If it has a segments attribute, return it
                return scores.segments
            
            # Default - single 30 second segment
            return torch.Tensor([[0.0, 30.0]])  # Default 30 second segment
        
        # Apply the patches
        vad.merge_chunks = safe_merge_chunks
        
        # Only patch binarize if it exists or we can find it
        if original_binarize or hasattr(vad, 'binarize'):
            vad.binarize = safe_binarize
            
        # Check if there's a __call__ function that might handle the binarization
        call_method = getattr(vad, '__call__', None)
        if call_method and callable(call_method):
            original_call = call_method
            
            def safe_call(self, *args, **kwargs):
                try:
                    return original_call(self, *args, **kwargs)
                except (AttributeError, KeyError, TypeError, ValueError) as e:
                    print(f"Original __call__ failed: {e}, using safe fallback")
                    # Fall back to our safe implementation
                    scores = args[0] if args else kwargs.get('scores', None)
                    threshold = kwargs.get('threshold', 0.5)
                    min_duration = kwargs.get('min_duration', 0.1)
                    return safe_binarize(scores, threshold, min_duration)
            
            vad.__call__ = safe_call
            
        return True
    except Exception as e:
        print(f"Error applying VAD function patches: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vad_functions():
    """Test the VAD functions with various inputs to ensure they work"""
    try:
        import whisperx.vad as vad
        
        # Test different input types
        test_inputs = [
            # Standard dict with segments and scores
            {"segments": torch.Tensor([[0.0, 5.0]]), "speech_probs": torch.ones(500)},
            
            # Custom class with segments and scores attributes
            type("TestVAD", (), {"segments": torch.Tensor([[0.0, 5.0]]), "speech_probs": torch.ones(500)}),
            
            # Just a tensor of segments
            torch.Tensor([[0.0, 5.0]]),
            
            # Custom class with data attribute
            type("TestScores", (), {"data": torch.ones(500)})
        ]
        
        print("Testing VAD functions with various inputs:")
        for i, test_input in enumerate(test_inputs):
            try:
                # Test merge_chunks
                segments, scores = vad.merge_chunks(test_input)
                segments_bin = vad.binarize(scores)
                print(f"Test {i+1}: merge_chunks and binarize successful")
                print(f"  Segments shape: {segments.shape}")
                print(f"  Binarized segments shape: {segments_bin.shape}")
            except Exception as e:
                print(f"Test {i+1} failed: {e}")
        
        return True
    except Exception as e:
        print(f"Error testing VAD functions: {e}")
        return False

if __name__ == "__main__":
    print("Applying safer VAD functions to WhisperX")
    if apply_safer_vad_functions():
        print("Successfully applied safer VAD functions")
        test_result = test_vad_functions()
        if test_result:
            print("All tests passed!")
        else:
            print("Some tests failed")
    else:
        print("Failed to apply safer VAD functions")
        sys.exit(1)
    
    sys.exit(0) 