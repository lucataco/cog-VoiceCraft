"""
Download the VAD model from updated URLs for WhisperX
"""

import os
import urllib.request
import ssl
import time
import shutil
from pathlib import Path

# Updated and backup URLs with pinned version to avoid future changes
VAD_MODEL_URLS = [
    # Primary URL - pinned to specific commit
    "https://raw.githubusercontent.com/snakers4/silero-vad/8145ed9a9183399d2793fc51150dd413c4449a65/files/silero_vad.onnx",
    # Backup URLs
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/files/silero_vad.onnx",
    "https://huggingface.co/pyannote/silero-vad/resolve/main/silero_vad.onnx",
]

def download_vad_model(cache_dir, max_retries=3):
    """
    Download the VAD model from updated URLs to the specified cache directory.
    
    Args:
        cache_dir: The directory to save the model to
        max_retries: Maximum number of retry attempts
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    model_path = os.path.join(cache_dir, "silero_vad.onnx")
    
    # Check if the model already exists
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        print(f"VAD model already exists at {model_path}")
        return True
    
    # Create SSL context that ignores certificate validation
    ssl_context = ssl._create_unverified_context()
    
    # Create redirect handler
    class RedirectHandler(urllib.request.HTTPRedirectHandler):
        def http_error_301(self, req, fp, code, msg, headers):
            print(f"Handling HTTP 301 redirect...")
            return urllib.request.HTTPRedirectHandler.http_error_301(
                self, req, fp, code, msg, headers)
            
        def http_error_302(self, req, fp, code, msg, headers):
            print(f"Handling HTTP 302 redirect...")
            return urllib.request.HTTPRedirectHandler.http_error_302(
                self, req, fp, code, msg, headers)
    
    # Install the handler
    opener = urllib.request.build_opener(RedirectHandler)
    urllib.request.install_opener(opener)
    
    # Temporary file to download to first, to avoid corrupted downloads
    temp_file = os.path.join(cache_dir, "temp_vad_model.onnx")
    
    # Try each URL in sequence
    for retry in range(max_retries):
        for url_index, url in enumerate(VAD_MODEL_URLS):
            try:
                print(f"Downloading VAD model from {url} (attempt {retry+1}/{max_retries}, URL {url_index+1}/{len(VAD_MODEL_URLS)})")
                
                # Create request with user agent
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                req = urllib.request.Request(url, headers=headers)
                
                start_time = time.time()
                with urllib.request.urlopen(req, context=ssl_context) as source:
                    with open(temp_file, "wb") as output:
                        # Read and write in chunks to handle large files
                        shutil.copyfileobj(source, output)
                
                download_time = time.time() - start_time
                
                # Verify the file exists and has size > 0
                if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                    # Move the temp file to the final location
                    shutil.move(temp_file, model_path)
                    print(f"Downloaded VAD model in {download_time:.2f} seconds to {model_path}")
                    return True
                else:
                    print(f"Downloaded file is empty")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            
            except Exception as e:
                print(f"Error downloading from {url}: {e}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                # Continue to the next URL
                continue
    
    print("Failed to download VAD model after all attempts")
    return False

# Also add a function to copy the VAD model to the WhisperX cache location
def copy_to_whisperx_cache(cache_dir):
    """
    Copy the VAD model to the WhisperX cache location
    
    Args:
        cache_dir: The directory where the model is cached
        
    Returns:
        bool: True if copy was successful, False otherwise
    """
    model_path = os.path.join(cache_dir, "silero_vad.onnx")
    
    if not os.path.exists(model_path):
        print(f"VAD model not found at {model_path}, cannot copy to WhisperX cache")
        return False
    
    # WhisperX looks for the model in ~/.cache/torch/vad/
    whisperx_cache = os.path.join(os.path.expanduser("~"), ".cache", "torch", "vad")
    os.makedirs(whisperx_cache, exist_ok=True)
    
    whisperx_model_path = os.path.join(whisperx_cache, "silero_vad.onnx")
    
    try:
        shutil.copy(model_path, whisperx_model_path)
        print(f"Copied VAD model to WhisperX cache at {whisperx_model_path}")
        return True
    except Exception as e:
        print(f"Error copying VAD model to WhisperX cache: {e}")
        return False

if __name__ == "__main__":
    # If run directly, download to the current directory
    cache_dir = "./model_cache"
    success = download_vad_model(cache_dir)
    print(f"Download {'successful' if success else 'failed'}")
    
    # Also copy to WhisperX cache
    if success:
        copy_success = copy_to_whisperx_cache(cache_dir)
        print(f"Copy to WhisperX cache {'successful' if copy_success else 'failed'}") 