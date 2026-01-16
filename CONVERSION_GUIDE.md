# MLX Conversion Guide for pyannote/speaker-diarization-community-1

This guide explains how to convert the pyannote speaker diarization model from PyTorch to MLX format for optimized inference on Apple Silicon.

## Overview

The conversion process involves:
1. Understanding the model architecture
2. Downloading PyTorch weights
3. Converting weights to MLX format
4. Validating the converted model

## Key MLX Conversion Concepts

### 1. Weight Format
- **PyTorch**: Uses `.pt`, `.pth`, or `.bin` files
- **MLX**: Uses `.safetensors` or `.npz` format
- MLX provides `mx.save_safetensors()` for efficient serialization

### 2. Tensor Dimension Order
Different frameworks use different dimension ordering for certain layers:

**Convolution Layers:**
- PyTorch Conv1D: `(out_channels, in_channels, kernel_size)`
- MLX Conv1D: `(out_channels, kernel_size, in_channels)`
- PyTorch Conv2D: `(out_channels, in_channels, height, width)`
- MLX Conv2D: `(out_channels, height, width, in_channels)`

**Solution:** Use `swapaxes()` or `transpose()` during conversion

### 3. Data Types
- MLX supports: `float16`, `float32`, `bfloat16`
- Default recommendation: `float16` for good performance/accuracy balance
- bfloat16 requires special handling (not directly NumPy-convertible)

### 4. Model Sharding
For large models (>5GB):
- Split weights into multiple shard files
- Create an index file mapping weights to shards
- Use `model-00001-of-00003.safetensors` naming convention

## Conversion Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `mlx` - Apple's ML framework
- `torch` - For loading PyTorch weights
- `transformers` - For downloading from Hugging Face
- `huggingface-hub` - For model downloading
- `safetensors` - For efficient serialization

### Step 2: Obtain PyTorch Model

**Option A: Download from Hugging Face**

```python
from huggingface_hub import snapshot_download

# Requires authentication for pyannote models
model_path = snapshot_download(
    repo_id="pyannote/speaker-diarization-community-1",
    allow_patterns=["*.bin", "*.pt", "*.json"],
)
```

**Option B: Use Local Checkpoint**

If you have a local PyTorch checkpoint:
```python
model_path = "/path/to/local/checkpoint.pt"
```

### Step 3: Run Conversion

Use the provided conversion script:

```bash
python src/convert.py \
    --pytorch-path "pyannote/speaker-diarization-community-1" \
    --mlx-path "models/mlx_model" \
    --model-type "full" \
    --dtype "float16"
```

Or programmatically:

```python
from src.convert import convert_pyannote_model

convert_pyannote_model(
    pytorch_path="path/to/pytorch/model",
    mlx_path="models/mlx_model",
    model_type="segmentation",  # or "embedding" or "full"
    dtype="float16"
)
```

### Step 4: Verify Conversion

Check the output directory structure:
```
models/mlx_model/
├── model.safetensors (or model-00001-of-00003.safetensors for sharded)
├── model.safetensors.index.json
└── config.json
```

## Understanding the pyannote Architecture

The `speaker-diarization-community-1` pipeline consists of:

### 1. Segmentation Model
- **Purpose**: Detects speech boundaries and speaker changes
- **Architecture**: LSTM-based neural network
- **Input**: Mel spectrogram features
- **Output**: Speaker activity and change point probabilities

### 2. Embedding Model  
- **Purpose**: Extracts speaker embeddings for clustering
- **Architecture**: LSTM + pooling + embedding layers
- **Input**: Mel spectrogram features
- **Output**: L2-normalized speaker embeddings

### 3. Clustering (VBx)
- **Purpose**: Groups embeddings to identify unique speakers
- **Method**: Variational Bayes HMM clustering
- **Not neural**: Implemented in NumPy/SciPy (no conversion needed)

## Common Conversion Issues

### Issue 1: Missing Dependencies

**Error:** `ImportError: No module named 'torch'`

**Solution:**
```bash
pip install torch transformers
```

### Issue 2: Authentication Required

**Error:** `Repository not found` or `401 Unauthorized`

**Solution:**
Pyannote models require authentication:
1. Accept the license on Hugging Face
2. Create an access token at https://huggingface.co/settings/tokens
3. Login: `huggingface-cli login`

### Issue 3: Dimension Mismatch

**Error:** `Shape mismatch in conv layers`

**Solution:**
The conversion script automatically handles dimension reordering for conv layers. If you encounter issues, check that `sanitize_weights()` is being applied.

### Issue 4: Memory Issues

**Error:** `Out of memory` during conversion

**Solution:**
- Convert on a device with sufficient RAM (16GB+ recommended)
- Enable weight sharding for large models
- Process in smaller batches

## Weight Mapping Examples

### LSTM Layers
```python
# PyTorch naming
"lstm.weight_ih_l0"  # Input-hidden weights layer 0
"lstm.weight_hh_l0"  # Hidden-hidden weights layer 0
"lstm.bias_ih_l0"    # Input-hidden bias layer 0
"lstm.bias_hh_l0"    # Hidden-hidden bias layer 0

# MLX naming (handled by remap_pytorch_keys)
"lstm_layers.0.Wx"   # Input weights layer 0
"lstm_layers.0.Wh"   # Hidden weights layer 0  
"lstm_layers.0.bias" # Combined bias layer 0
```

### Attention Layers
```python
# PyTorch
"self_attn.q_proj.weight"
"self_attn.k_proj.weight"
"self_attn.v_proj.weight"
"self_attn.out_proj.weight"

# MLX (handled automatically)
"attention.query_proj.weight"
"attention.key_proj.weight"
"attention.value_proj.weight"
"attention.output_proj.weight"
```

## Advanced: Manual Conversion

If you need more control over the conversion process:

```python
import torch
import mlx.core as mx
from pathlib import Path

# Load PyTorch model
checkpoint = torch.load("model.pt", map_location="cpu")
pytorch_weights = checkpoint["state_dict"]

# Convert each weight
mlx_weights = {}
for key, value in pytorch_weights.items():
    # Convert to numpy, then to MLX
    numpy_weight = value.to(torch.float16).numpy()
    mlx_weight = mx.array(numpy_weight, mx.float16)
    
    # Handle conv layers
    if "conv" in key and mlx_weight.ndim == 3:
        mlx_weight = mlx_weight.swapaxes(1, 2)
    
    mlx_weights[key] = mlx_weight

# Save
mx.save_safetensors("model.safetensors", mlx_weights)
```

## Performance Optimization

### Quantization
For faster inference and smaller model size:

```python
# Future feature - quantize to 4-bit or 8-bit
import mlx.nn as nn

# After loading model
nn.quantize(model, group_size=64, bits=4)
```

### Device Selection
```python
# Use GPU (Apple Neural Engine)
pipeline = SpeakerDiarizationPipeline(device="gpu")

# Use CPU only
pipeline = SpeakerDiarizationPipeline(device="cpu")
```

## Testing the Converted Model

After conversion, test with a sample audio file:

```python
from src.pipeline import SpeakerDiarizationPipeline

# Load pipeline with converted model
pipeline = SpeakerDiarizationPipeline(
    model_path="models/mlx_model",
    device="gpu"
)

# Test diarization
diarization = pipeline("test_audio.wav")

# Verify results
for turn, speaker in diarization.speaker_diarization:
    print(f"{speaker}: {turn.start:.2f}s - {turn.end:.2f}s")
```

## References

### MLX Documentation
- [MLX GitHub](https://github.com/ml-explore/mlx)
- [MLX Examples](https://github.com/ml-explore/mlx-examples)
- [MLX API Docs](https://ml-explore.github.io/mlx/)

### pyannote Resources
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [Model Card](https://huggingface.co/pyannote/speaker-diarization-community-1)

### Conversion Examples
- [MLX Whisper Conversion](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
- [MLX LLaMA Conversion](https://github.com/ml-explore/mlx-examples/tree/main/llms/llama)
- [MLX CLIP Conversion](https://github.com/ml-explore/mlx-examples/tree/main/clip)

## Troubleshooting

For issues or questions:
1. Check the error message carefully
2. Verify all dependencies are installed
3. Ensure you have proper authentication for pyannote models
4. Review the MLX examples for similar conversion patterns
5. Open an issue on GitHub with details about your environment

## Next Steps

After successful conversion:
1. Run tests: `python tests/test_basic.py`
2. Try examples: `python examples/basic_usage.py`
3. Benchmark performance against PyTorch version
4. Fine-tune thresholds for your use case
