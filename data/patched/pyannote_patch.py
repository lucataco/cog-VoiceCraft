"""
Monkey patch for pyannote.audio to fix NumPy 2.0 compatibility issues.
"""

import sys
import os
import importlib
import inspect
import types
import numpy as np


def apply_pyannote_patch():
    """
    Apply monkey patch to pyannote.audio to fix NumPy 2.0 compatibility issues.
    
    This specifically targets the use of np.NaN in pyannote.audio.core.inference,
    replacing it with np.nan which is the correct usage in NumPy 2.0.
    """
    try:
        # First try to import the module to make sure it's available
        import pyannote.audio.core.inference
        
        # Get the original Inference class definition
        from pyannote.audio.core.inference import Inference, BaseInference
        
        # Define our patched version of the class decorator
        def patched_inference_decorator(func):
            def wrapper(*args, **kwargs):
                # Replace np.NaN with np.nan in the kwargs
                if 'missing' in kwargs and kwargs['missing'] is np.NaN:
                    kwargs['missing'] = np.nan
                return func(*args, **kwargs)
            return wrapper
            
        # Apply our patch by directly modifying the source code
        inference_module = sys.modules['pyannote.audio.core.inference']
        
        # Find all instances of np.NaN in the module and replace with np.nan
        for name, obj in inspect.getmembers(inference_module):
            if isinstance(obj, type) and issubclass(obj, BaseInference):
                # Look for class attributes that use np.NaN
                for attr_name, attr_value in obj.__dict__.items():
                    if attr_value is np.NaN:
                        setattr(obj, attr_name, np.nan)
            
            # Replace in function defaults
            if isinstance(obj, types.FunctionType):
                if obj.__defaults__:
                    defaults = list(obj.__defaults__)
                    for i, default in enumerate(defaults):
                        if default is np.NaN:
                            defaults[i] = np.nan
                    obj.__defaults__ = tuple(defaults)
        
        print("Successfully applied patch to pyannote.audio for NumPy 2.0 compatibility")
        return True
        
    except ImportError as e:
        print(f"Failed to patch pyannote.audio: {e}")
        return False
    except Exception as e:
        print(f"Error while patching pyannote.audio: {e}")
        return False


if __name__ == "__main__":
    apply_pyannote_patch() 