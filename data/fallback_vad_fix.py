"""
Fallback script to directly download and place the VAD model in the WhisperX cache
Simpler approach that bypasses patching and only ensures the model file is present
"""

import os
import sys
import urllib.request
import ssl
import shutil
from pathlib import Path

def direct_vad_model_download():
    """Direct download of the VAD model to the WhisperX cache location"""
    # The VAD model URL (pinned to specific commit)
    vad_model_url = "https://raw.githubusercontent.com/snakers4/silero-vad/8145ed9a9183399d2793fc51150dd413c4449a65/files/silero_vad.onnx"
    
    # Backup URLs in case the primary one fails
    backup_urls = [
        "https://raw.githubusercontent.com/snakers4/silero-vad/master/files/silero_vad.onnx",
        "https://huggingface.co/pyannote/silero-vad/resolve/main/silero_vad.onnx",
        "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx",
        "https://github.com/pyannote/silero-vad/raw/main/silero_vad.onnx", # Another potential mirror
        "https://drive.google.com/uc?export=download&id=1Oqy7cjbA6DzYVh-pv7G5JsKZ8xNVDEUS"  # Google Drive link
    ]
    
    # WhisperX cache location
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "vad")
    os.makedirs(cache_dir, exist_ok=True)
    
    model_path = os.path.join(cache_dir, "silero_vad.onnx")
    
    # Check if model already exists
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        print(f"VAD model already exists at {model_path}")
        return True
    
    print(f"Downloading VAD model directly to {model_path}")
    
    # Create custom redirect handler for HTTP 301/302 redirects
    class SmartRedirectHandler(urllib.request.HTTPRedirectHandler):
        def http_error_301(self, req, fp, code, msg, headers):
            print(f"Handling HTTP 301 redirect from {req.full_url}")
            if headers and 'Location' in headers:
                print(f"  → Redirecting to {headers['Location']}")
            result = urllib.request.HTTPRedirectHandler.http_error_301(
                self, req, fp, code, msg, headers)
            return result
            
        def http_error_302(self, req, fp, code, msg, headers):
            print(f"Handling HTTP 302 redirect from {req.full_url}")
            if headers and 'Location' in headers:
                print(f"  → Redirecting to {headers['Location']}")
            result = urllib.request.HTTPRedirectHandler.http_error_302(
                self, req, fp, code, msg, headers)
            return result
    
    # Install the opener with our custom redirect handler
    opener = urllib.request.build_opener(SmartRedirectHandler)
    urllib.request.install_opener(opener)
    
    # Create SSL context that ignores certificate validation
    ssl_context = ssl._create_unverified_context()
    
    # Temporary file to download to
    temp_file = os.path.join(cache_dir, "temp_vad_model.onnx")
    
    # Try all URLs
    all_urls = [vad_model_url] + backup_urls
    success = False
    
    for url in all_urls:
        try:
            print(f"\nAttempting to download VAD model from: {url}")
            # Create request with different user agent
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': '*/*'  # Accept any content type
            }
            req = urllib.request.Request(url, headers=headers)
            
            # Custom download with progress reporting
            with urllib.request.urlopen(req, context=ssl_context) as source:
                # Print response headers for debugging
                print(f"Response headers:")
                for header, value in source.headers.items():
                    print(f"  {header}: {value}")
                
                file_size = int(source.headers.get("Content-Length", 0))
                print(f"File size: {file_size} bytes")
                
                with open(temp_file, "wb") as output:
                    # Read and write in chunks
                    chunk_size = 8192
                    total_read = 0
                    while True:
                        chunk = source.read(chunk_size)
                        if not chunk:
                            break
                        output.write(chunk)
                        total_read += len(chunk)
                        print(f"Downloaded: {total_read} bytes  ", end="\r")
                
                print(f"\nDownload completed: {total_read} bytes")
            
            # Verify the download was successful and not empty
            if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                print(f"Moving downloaded file ({os.path.getsize(temp_file)} bytes) to final location: {model_path}")
                shutil.move(temp_file, model_path)
                success = True
                break
            else:
                print(f"Error: Downloaded file is empty or missing. Size: {os.path.getsize(temp_file) if os.path.exists(temp_file) else 'file does not exist'}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
        except Exception as e:
            print(f"Error downloading from {url}:")
            import traceback
            traceback.print_exc()
            print(f"Exception details: {str(e)}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            # Continue to the next URL
    
    if not success:
        print("CRITICAL: All URLs failed for VAD model download")
        
        # Desperate measures: try curl or wget as a last resort
        try:
            print("\nAttempting download using curl as a last resort...")
            # Try using curl (often pre-installed on Unix systems)
            import subprocess
            result = subprocess.run(
                ["curl", "-L", "-o", model_path, vad_model_url],
                capture_output=True
            )
            
            if result.returncode == 0 and os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                print(f"Successfully downloaded using curl. Size: {os.path.getsize(model_path)} bytes")
                success = True
            else:
                print(f"curl failed: {result.stderr.decode()}")
                
                # Try wget as another alternative
                print("\nAttempting download using wget...")
                result = subprocess.run(
                    ["wget", "-O", model_path, vad_model_url],
                    capture_output=True
                )
                
                if result.returncode == 0 and os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                    print(f"Successfully downloaded using wget. Size: {os.path.getsize(model_path)} bytes")
                    success = True
                else:
                    print(f"wget failed: {result.stderr.decode()}")
        except Exception as e:
            print(f"External download tools attempt failed: {e}")
    
    # Final verification
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        print(f"VAD model successfully available at: {model_path} (Size: {os.path.getsize(model_path)} bytes)")
        
        # Try to verify it's a valid ONNX file
        try:
            import onnx
            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
            print("VAD model successfully validated as a valid ONNX file")
        except Exception as e:
            print(f"Warning: Could not validate ONNX model (but file exists): {e}")
        
        return True
    else:
        print("FATAL ERROR: Could not obtain VAD model through any means")
        return False

if __name__ == "__main__":
    success = direct_vad_model_download()
    print(f"Direct VAD model download {'succeeded' if success else 'failed'}")
    sys.exit(0 if success else 1) 