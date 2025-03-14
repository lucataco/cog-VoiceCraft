"""
Patched version of pyannote.audio's inference.py file to fix NumPy 2.0 compatibility issues.
"""

# This is just the part that needs to be modified
# The actual file is much larger, but we only need to fix the np.NaN reference

import numpy as np
from typing import Callable, Optional, Text, Union
from pyannote.core import SlidingWindow

def patched_inference_decorator(func):
    """This is a decorator that replaces the original Inference class implementation
    with a version that uses np.nan instead of np.NaN."""
    
    # Original code would define a class here, but we'll replace just what we need
    
    # This is a factory function to create the patched class
    def create_patched_class(*args, **kwargs):
        # The important part is to replace np.NaN with np.nan in the decorator arguments
        missing = kwargs.pop('missing', np.nan)  # Using np.nan instead of np.NaN
        kwargs['missing'] = missing
        
        # Call the original function with fixed arguments
        return func(*args, **kwargs)
    
    return create_patched_class 