# VoiceCraft Patches

This directory contains patches to fix issues with VoiceCraft's dependencies.

## Patches Included

### 1. AudioCraft Patch

Files:
- `patched/environment.py` - A simplified version of AudioCraft's environment.py that doesn't require AWS keys
- `patched/train.py` - A simplified version of AudioCraft's train.py that provides dummy Dora objects
- `patch_audiocraft.py` - The script that applies these patches

This fixes the error where AudioCraft tries to access AWS configuration that doesn't exist:
```
omegaconf.errors.ConfigKeyError: Missing key aws
    full_key: aws
    object_type=dict
```

### 2. NumPy 2.0 Compatibility Patch

Files:
- `numpy_compatibility.py` - A monkey patch that adds np.NaN back to NumPy 2.0

This fixes the error with pyannote.audio:
```
AttributeError: `np.NaN` was removed in the NumPy 2.0 release. Use `np.nan` instead.
```

### 3. WhisperX Compatibility Patch

Files:
- `patch_whisperx.py` - A script to patch WhisperX for compatibility with newer faster-whisper versions

This fixes the error with WhisperX transcription options:
```
TypeError: TranscriptionOptions.__init__() missing 2 required positional arguments: 'multilingual' and 'hotwords'
```

## WhisperX VAD Patch

This patch addresses the `HTTP Error 301: Moved Permanently` issue that occurs when WhisperX tries to download the Voice Activity Detection (VAD) model. The URL for this model has changed, and the patch ensures that the model can be downloaded from the correct URL.

The issue manifests as:
```
urllib.error.HTTPError: HTTP Error 301: Moved Permanently
```

### How to Apply the Patch

Apply the patch in `predict.py`:

```python
from data.preload_vad_model import preload_vad_model
preload_vad_model(MODEL_CACHE)
```

### What the Patch Does

The solution uses a multi-layered approach with several fallback mechanisms:

1. Downloads the VAD model from a pinned URL to the specified cache directory
2. Pre-copies the model to WhisperX's default cache location (`~/.cache/torch/vad/`)
3. Patches the WhisperX VAD module to:
   - Use the updated URL if it needs to download the model
   - Check for the pre-downloaded model in both the cache directory and the default location
   - Handle HTTP redirects properly
   - Add proper user agent headers to avoid being blocked
4. If any of these steps fail, it will try progressively simpler approaches:
   - Direct file copy from the cache to WhisperX's default location
   - Emergency direct download to WhisperX's default location

### Safety Measures

- Multiple fallback mechanisms ensure the VAD model will be available
- Error handling at each step prevents the patch from breaking the application
- The patch preserves the original behavior of WhisperX while fixing the URL issue
- If the VAD model is already downloaded, it will not attempt to download it again
- The download process uses a temporary file to prevent corrupted downloads
- SSL certificate validation is disabled to work around potential issues
- HTTP redirects are handled properly to follow URL changes
- User agent headers are added to bypass some filtering

### Testing

Use the `test_vad_download.py` script to test that the VAD patch works correctly:

```python
python data/test_vad_download.py
```

The test will check:
1. The VAD model can be downloaded
2. The WhisperX VAD module can be patched
3. The VAD model can be loaded from WhisperX

## How to Apply

These patches are automatically applied in predict.py before importing any modules that use them:

```python
# Apply NumPy 2.0 compatibility patch first
patch_numpy()

# Then apply the AudioCraft patch
patch_audiocraft()

# Then apply the WhisperX patch
patch_whisperx()
```

Additional safety measures:
- NumPy has been pinned to version 1.26.4 in cog.yaml to avoid NumPy 2.0 compatibility issues
- WhisperxModel class includes robust error handling for parameter mismatches
- faster-whisper has been pinned to a compatible version in cog.yaml

## Testing

You can test the patches using:
- `test_all_patches.py` - Tests all patches in one go 