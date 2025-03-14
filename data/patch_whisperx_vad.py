"""
Patch for fixing the WhisperX VAD module to use the correct URL and pre-downloaded model
"""

import os
import re
import sys
import importlib.util
import urllib.request
import ssl
from pathlib import Path

from data.download_vad_model import download_vad_model, copy_to_whisperx_cache

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

def create_opener_with_redirect_handler():
    """Create a URL opener that can handle redirects"""
    # Create a custom redirect handler
    class RedirectHandler(urllib.request.HTTPRedirectHandler):
        def http_error_301(self, req, fp, code, msg, headers):
            print(f"Handling HTTP 301 redirect...")
            result = urllib.request.HTTPRedirectHandler.http_error_301(
                self, req, fp, code, msg, headers)
            return result
            
        def http_error_302(self, req, fp, code, msg, headers):
            print(f"Handling HTTP 302 redirect...")
            result = urllib.request.HTTPRedirectHandler.http_error_302(
                self, req, fp, code, msg, headers)
            return result
    
    # Create and install the opener
    opener = urllib.request.build_opener(RedirectHandler)
    urllib.request.install_opener(opener)
    return opener

def patch_whisperx_vad(model_cache_dir="model_cache"):
    """
    Patch the whisperx.vad module to use the correct URL and pre-downloaded model
    
    Args:
        model_cache_dir: Directory where the VAD model should be cached
        
    Returns:
        bool: True if patch was applied successfully, False otherwise
    """
    # First, download the VAD model if it doesn't exist
    model_downloaded = download_vad_model(model_cache_dir)
    if not model_downloaded:
        print("Failed to download VAD model, patch may not work correctly")
    else:
        # Pre-copy the model to WhisperX's cache location to avoid download attempts
        copy_success = copy_to_whisperx_cache(model_cache_dir)
        if copy_success:
            print("Successfully copied VAD model to WhisperX cache location")
        else:
            print("Warning: Failed to copy VAD model to WhisperX cache location")
    
    # Create a redirect handler for URL requests
    create_opener_with_redirect_handler()
    
    # Find the whisperx.vad module
    vad_path = find_whisperx_vad_path()
    if vad_path is None:
        return False
    
    # Read the content of the module
    with open(vad_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "PATCHED_VAD_SEGMENTATION" in content:
        print("WhisperX VAD module already patched")
        return True
    
    # Prepare the patch
    modified = False
    
    # 1. Update the VAD_SEGMENTATION_URL
    url_pattern = r'VAD_SEGMENTATION_URL\s*=\s*[\'"]([^\'"]+)[\'"]'
    if re.search(url_pattern, content):
        new_url = "https://raw.githubusercontent.com/snakers4/silero-vad/8145ed9a9183399d2793fc51150dd413c4449a65/files/silero_vad.onnx"
        content = re.sub(
            url_pattern,
            f'VAD_SEGMENTATION_URL = "{new_url}"  # PATCHED_VAD_SEGMENTATION URL',
            content
        )
        modified = True
    
    # 2. Add the redirect handler import
    import_pattern = r'import urllib.request'
    if re.search(import_pattern, content):
        content = re.sub(
            import_pattern,
            'import urllib.request\nimport ssl  # PATCHED_VAD_SEGMENTATION import',
            content
        )
        modified = True
    
    # 3. Replace the entire load_vad_model function
    # First find it
    load_vad_pattern = r'def load_vad_model\([^)]*\):.*?return model'
    
    # Create the replacement function
    replacement_func = '''def load_vad_model(device="cpu", use_auth_token=None, **kwargs):
    """Load a Voice Activity Detection (VAD) model.
    
    PATCHED_VAD_SEGMENTATION by VoiceCraft to handle URL redirects and use pre-downloaded models.
    """
    model_name = "silero_vad"
    model_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "vad")
    model_fp = os.path.join(model_dir, f"{{model_name}}.onnx")
    
    # First check if model exists in the specified cache directory
    cache_model_path = os.path.join("{model_cache_dir}", "silero_vad.onnx")
    
    if not os.path.exists(model_fp):
        # Try to use the pre-downloaded model first
        if os.path.exists(cache_model_path):
            print(f"Using pre-downloaded VAD model from {{cache_model_path}}")
            import shutil
            os.makedirs(os.path.dirname(model_fp), exist_ok=True)
            shutil.copy(cache_model_path, model_fp)
        else:
            # If no pre-downloaded model, try to download it
            print(f"Downloading VAD model from {{VAD_SEGMENTATION_URL}}")
            os.makedirs(os.path.dirname(model_fp), exist_ok=True)
            
            # Create SSL context that ignores certificate validation
            ssl_context = ssl._create_unverified_context()
            
            try:
                # Create a request with a User-Agent to avoid some filtering
                headers = {{'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}}
                req = urllib.request.Request(VAD_SEGMENTATION_URL, headers=headers)
                
                with urllib.request.urlopen(req, context=ssl_context) as source, open(model_fp, "wb") as output:
                    output.write(source.read())
            except Exception as e:
                print(f"Error downloading VAD model: {{e}}")
                alternate_urls = [
                    "https://raw.githubusercontent.com/snakers4/silero-vad/8145ed9a9183399d2793fc51150dd413c4449a65/files/silero_vad.onnx",
                    "https://raw.githubusercontent.com/snakers4/silero-vad/master/files/silero_vad.onnx",
                    "https://huggingface.co/pyannote/silero-vad/resolve/main/silero_vad.onnx"
                ]
                
                for url in alternate_urls:
                    try:
                        print(f"Trying alternate URL: {{url}}")
                        req = urllib.request.Request(url, headers=headers)
                        with urllib.request.urlopen(req, context=ssl_context) as source, open(model_fp, "wb") as output:
                            output.write(source.read())
                        break
                    except Exception as e:
                        print(f"Error downloading from alternate URL: {{e}}")
    
    # If the model file doesn't exist after all attempts, raise an error
    if not os.path.exists(model_fp):
        raise RuntimeError(
            f"VAD model file not found at {{model_fp}} and all download attempts failed. "
            f"Please download the model manually and place it at this location."
        )
    
    # Load the model
    import onnxruntime as ort
    try:
        model = ort.InferenceSession(model_fp, providers=['CPUExecutionProvider'])
        return model
    except Exception as e:
        print(f"Error loading VAD model: {{e}}")
        if os.path.exists(model_fp):
            # If the model exists but can't be loaded, it might be corrupted
            # Try to download it again
            print("Model might be corrupted, removing and trying again...")
            os.remove(model_fp)
            
            # Try the cache again
            if os.path.exists(cache_model_path):
                print(f"Using pre-downloaded VAD model from {{cache_model_path}}")
                import shutil
                os.makedirs(os.path.dirname(model_fp), exist_ok=True)
                shutil.copy(cache_model_path, model_fp)
                model = ort.InferenceSession(model_fp, providers=['CPUExecutionProvider'])
                return model
            
            # If that doesn't work, give up
            raise RuntimeError(
                f"Failed to load VAD model after multiple attempts: {{e}}. "
                f"Please download the model manually and place it at {{model_fp}}."
            )
        raise'''
    
    # Format with actual model_cache_dir
    replacement_func = replacement_func.format(model_cache_dir=model_cache_dir)
    
    # Replace the entire function
    if re.search(load_vad_pattern, content, re.DOTALL):
        content = re.sub(
            load_vad_pattern,
            replacement_func,
            content,
            flags=re.DOTALL
        )
        modified = True
    
    # Write back the modified content if changes were made
    if modified:
        with open(vad_path, 'w') as f:
            f.write(content)
        print(f"Successfully patched WhisperX VAD module at {vad_path}")
        
        # Try to import the module to verify it works
        try:
            # Find and load the module to ensure our patch works
            spec = importlib.util.spec_from_file_location("vad_module", vad_path)
            vad_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vad_module)
            print("Successfully verified VAD module can be imported after patching")
        except Exception as e:
            print(f"Warning: Patched module failed import verification: {e}")
        
        return True
    else:
        print("No changes were made to the WhisperX VAD module")
        return False

if __name__ == "__main__":
    # If run directly, patch the module
    model_cache_dir = sys.argv[1] if len(sys.argv) > 1 else "model_cache"
    success = patch_whisperx_vad(model_cache_dir)
    print(f"Patch {'successful' if success else 'failed'}") 