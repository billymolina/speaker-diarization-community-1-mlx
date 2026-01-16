# Upload Instructions for Hugging Face mlx-community

## Quick Start

```bash
# 1. Install Hugging Face Hub
pip install huggingface_hub

# 2. Login to Hugging Face
huggingface-cli login

# 3. Upload the model
python upload_to_huggingface.py
```

## Detailed Steps

### 1. Prerequisites

Make sure you have:
- A Hugging Face account (https://huggingface.co/join)
- Access token with write permissions (https://huggingface.co/settings/tokens)

Install required package:
```bash
pip install huggingface_hub
```

### 2. Login to Hugging Face

```bash
huggingface-cli login
```

Enter your access token when prompted.

### 3. Upload Options

#### Option A: Upload to mlx-community (Recommended)

If you have write access to mlx-community:

```bash
python upload_to_huggingface.py
```

This will upload to: `mlx-community/wespeaker-voxceleb-resnet34-LM`

#### Option B: Upload to Your Personal Account

If you don't have mlx-community access yet:

1. Edit `upload_to_huggingface.py`:
   ```python
   organization = "YOUR_USERNAME"  # Change from "mlx-community"
   ```

2. Run the upload:
   ```bash
   python upload_to_huggingface.py
   ```

3. Later, request transfer to mlx-community or submit a PR

### 4. Verify Upload

After upload completes, verify at:
- https://huggingface.co/mlx-community/wespeaker-voxceleb-resnet34-LM

Or your personal repo:
- https://huggingface.co/YOUR_USERNAME/wespeaker-voxceleb-resnet34-LM

## Files Being Uploaded

The following files will be uploaded from `models/embedding_mlx/`:

| File | Size | Description |
|------|------|-------------|
| `weights.npz` | ~25 MB | Model weights (MLX format) |
| `config.json` | <1 KB | Model configuration |
| `README.md` | ~6 KB | Model card with usage instructions |
| `resnet_embedding.py` | ~13 KB | Model implementation code |
| `example_usage.py` | ~4 KB | Usage example script |

## Using the Uploaded Model

Once uploaded, users can load the model like this:

```python
from huggingface_hub import hf_hub_download
import mlx.core as mx
import mlx.nn as nn

# Download files
repo_id = "mlx-community/wespeaker-voxceleb-resnet34-LM"
weights_path = hf_hub_download(repo_id=repo_id, filename="weights.npz")
model_code = hf_hub_download(repo_id=repo_id, filename="resnet_embedding.py")

# Load model (after importing resnet_embedding module)
from resnet_embedding import load_resnet34_embedding
model = load_resnet34_embedding(weights_path)

# Use model
mel_spec = mx.array(...)  # Your mel spectrogram (batch, time, freq)
embeddings = model(mel_spec)
```

## Alternative: Manual Upload via Web Interface

You can also upload manually:

1. Go to https://huggingface.co/new
2. Create a new model repository
3. Upload files via the "Files" tab
4. Edit README directly in the web interface

## Getting mlx-community Access

If you want to upload directly to mlx-community:

1. Join the MLX community Discord: https://discord.gg/mlx
2. Request access to contribute models
3. Or submit a PR with your model link

## Troubleshooting

### "Authentication error"
```bash
huggingface-cli login
# Make sure to create a token with write permissions
```

### "Repository not found" or "Permission denied"
- Change `organization` to your username in `upload_to_huggingface.py`
- Upload to your personal account first

### "File too large"
- The weights file is ~25MB, which should be fine
- If issues persist, try using Git LFS (automatically enabled for .npz files)

### Import errors when testing
```bash
# Make sure you're in the right directory
cd models/embedding_mlx
python example_usage.py
```

## Next Steps After Upload

1. **Test the model**: Download and test using the Hugging Face Hub
2. **Share**: Tweet/post about the model with #MLX #AppleSilicon
3. **Documentation**: Add examples and use cases to the README
4. **Integration**: Create a HF Spaces demo (optional)

## Contact

For issues with mlx-community uploads:
- GitHub: https://github.com/ml-explore/mlx
- Discord: https://discord.gg/mlx
- Hugging Face: https://huggingface.co/mlx-community

---

**Model**: WeSpeaker ResNet34 Speaker Embedding (MLX)
**Original**: https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
**Conversion Date**: 2026-01-16
**Status**: ✅ Validated and ready for upload
