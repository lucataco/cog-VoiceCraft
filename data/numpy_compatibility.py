"""
NumPy 2.0 compatibility monkey patch.

This script adds backward compatibility for code that uses np.NaN 
(which was removed in NumPy 2.0 in favor of np.nan).
"""

import numpy as np
import sys


def apply_numpy_patch():
    """
    Apply a monkey patch to numpy to make np.NaN work again in NumPy 2.0.
    
    This is a compatibility shim for libraries that haven't been updated to use np.nan yet.
    """
    try:
        # Check if we need to apply the patch (if NaN doesn't exist)
        try:
            # If this doesn't throw an error, NaN already exists
            getattr(np, 'NaN')
            print("No need to patch numpy.NaN (already exists)")
            return True
        except AttributeError:
            # NaN doesn't exist, we should add it
            pass
            
        # Add NaN as an alias to nan
        setattr(np, 'NaN', np.nan)
        
        print("Successfully applied numpy.NaN compatibility patch")
        return True
        
    except Exception as e:
        print(f"Error applying numpy patch: {e}")
        return False


if __name__ == "__main__":
    apply_numpy_patch() 