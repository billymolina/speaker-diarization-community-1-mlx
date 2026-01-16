# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MLX port of pyannote speaker diarization models, optimized for Apple Silicon. The project converts PyTorch models to MLX format and provides real-time speaker diarization capabilities using Apple's Neural Engine.

**Converted Models:**

1. **Segmentation Model** (`pyannote/segmentation-3.0`)
   - **SincNet**: Learnable bandpass filter bank for raw waveform processing (80 filters, kernel size 251)
   - **Bidirectional LSTM**: 4-layer bidirectional LSTM with 128 hidden units per direction (manually implemented as MLX lacks native bidirectional support)
   - **Classification Head**: Linear layers + 7-class speaker classifier for frame-level predictions
   - **Stats:** 1,473,515 parameters | 5.6MB | ~17ms frame resolution

2. **Embedding Model** (`pyannote/wespeaker-voxceleb-resnet34-LM`) ✨ NEW!
   - **ResNet34**: 34-layer residual network with [3, 4, 6, 3] block configuration
   - **Temporal Statistics Pooling**: Pools mean/std over frequency dimension
   - **Embedding Output**: 256-dimensional L2-normalized speaker vectors
   - **Stats:** 6.6M parameters | 25MB | 256-dim embeddings

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Quick setup with script
./setup.sh
```

### Running Tests

The test suite validates the MLX implementation against PyTorch:

```bash
# Test model architecture and forward pass
python test_pyannote_model.py

# Test weight loading (1,473,515 parameters)
python test_weight_loading.py

# Test real audio inference
python test_real_inference.py

# Run basic test suite
python tests/test_basic.py
```

All architecture tests (5/5) should pass:
- SincNet filter generation
- SincNet frontend (3 layers)
- Complete model forward pass
- Weight loading
- Real audio inference

### Running Inference

```bash
# CLI inference with RTTM output
python diarize.py audio.wav --output results.rttm

# With custom parameters
python diarize.py audio.wav --output results.rttm \
  --model models/segmentation_mlx/weights.npz \
  --min-duration 0.5 \
  --chunk-size 160000

# Compare MLX output with PyTorch
python run_pytorch_comparison.py
python compare_rttm.py
```

### Model Conversion

```bash
# Convert segmentation model from Hugging Face
python scripts/convert_pyannote.py \
  --repo-id pyannote/segmentation-3.0 \
  --output models/custom_mlx

# Convert embedding model
python convert_embedding_model.py \
  --output models/embedding_mlx

# Test conversions
python test_conversion.py
python test_embedding_model.py
```

## Architecture Details

### Critical MLX vs PyTorch Differences

When working with this codebase, be aware of these key differences:

1. **Conv1D Dimension Ordering:**
   - PyTorch: `(batch, channels, length)` for input, `(out_channels, in_channels, kernel)` for weights
   - MLX: `(batch, length, channels)` for input, `(out_channels, kernel, in_channels)` for weights
   - **Action:** Always transpose when converting between formats

2. **Array Reversal:**
   - PyTorch: `torch.flip(tensor, dims=[1])`
   - MLX: `array[::-1]` (no native flip function)

3. **Bidirectional LSTM:**
   - PyTorch: Native `bidirectional=True` parameter
   - MLX: Manual implementation with separate forward/backward LSTMs
   - **Location:** `src/models.py:PyannoteSegmentationModel.__call__()` lines 523-531

4. **InstanceNorm Format:**
   - MLX expects channels in last dimension: `(..., channels)`
   - Transpose from PyTorch format before normalization

### SincNet Implementation (src/models.py:213-362)

The SincNet layer implements learnable bandpass filters using sinc functions:
```
h_BP(t) = 2*f_high*sinc(2*f_high*t) - 2*f_low*sinc(2*f_low*t)
```

- Filters initialized on mel-scale (lines 256-278)
- Both cosine (real) and sine (imaginary) filters generated (lines 321-325)
- 80 learnable filters total (40 cos + 40 sin)
- Constraints ensure valid frequency ranges (lines 313-319)

### Model Architecture Flow

1. **SincNet Frontend** (src/models.py:364-457):
   - Layer 1: SincNet (80 filters) → abs() → MaxPool(3) → InstanceNorm → LeakyReLU(0.01)
   - Layer 2: Conv1d(80→60) → MaxPool(3) → InstanceNorm → LeakyReLU(0.01)
   - Layer 3: Conv1d(60→60) → MaxPool(3) → InstanceNorm → LeakyReLU(0.01)
   - Output: `(batch, time_frames, 60)` features

2. **Bidirectional LSTM** (src/models.py:492-531):
   - 4 stacked layers, each with manual bidirectional implementation
   - Forward LSTM processes sequence left-to-right
   - Backward LSTM processes reversed sequence
   - Outputs concatenated: `(batch, time_frames, 256)`

3. **Classification Head** (src/models.py:533-544):
   - Linear(256→128) → ReLU
   - Linear(128→128) → ReLU
   - Linear(128→7) → LogSoftmax
   - Output: Frame-level log probabilities for 7 speaker classes

### Weight Loading (src/models.py:547-621)

When loading converted weights:
- SincNet learnable parameters: `low_hz_` and `band_hz_` (lines 559-560)
- Conv layers: Transpose PyTorch weights `(out, in, kernel)` → MLX `(out, kernel, in)` (lines 574-588)
- LSTM weights: PyTorch format compatible; combine bias_ih + bias_hh (lines 594-608)
- Linear/classifier: Direct mapping (lines 610-618)

## Project Structure

```
src/
├── models.py           # MLX segmentation model (SincNet + BiLSTM + Classifier)
├── resnet_embedding.py # MLX embedding model (ResNet34 + TSTP) ✨ NEW!
├── convert.py          # PyTorch→MLX conversion utilities
├── pipeline.py         # Diarization pipeline (segmentation + embedding + clustering)
├── audio.py            # Audio preprocessing utilities
├── clustering.py       # Speaker clustering algorithms
└── diarization_types.py  # Data structures for diarization output

models/
├── segmentation_mlx/
│   └── weights.npz     # Segmentation model weights (5.6MB)
└── embedding_mlx/      # ✨ NEW!
    ├── weights.npz     # Embedding model weights (25MB)
    └── config.json     # Model configuration

diarize.py              # Main CLI inference script
scripts/convert_pyannote.py     # Segmentation model conversion
convert_embedding_model.py  # Embedding model conversion ✨ NEW!

tests/
├── evaluation/         # Test audio and ground truth RTTM files
└── test_embedding_model.py  # Embedding model tests ✨ NEW!

examples/
├── basic_usage.py      # Simple API usage example
├── advanced_usage.py   # Advanced features example
└── complete_example.py # End-to-end diarization example
```

## Important Notes

### When Modifying Model Architecture

1. **SincNet filters must remain learnable** - don't replace with fixed filters
2. **Preserve bidirectional nature of LSTMs** - both forward and backward passes required
3. **Maintain dimension orders** - incorrect transposes will cause silent failures
4. **Keep abs() only on first SincNet layer** - critical for correct feature extraction (line 435)

### When Adding New Tests

- Compare MLX output with PyTorch reference using `run_pytorch_comparison.py`
- Check numerical accuracy to at least 3 decimal places
- Test with real audio files in `tests/evaluation/`
- Validate RTTM output format matches pyannote.audio standard

### When Converting Models

- Use `scripts/convert_pyannote.py` for official pyannote models
- Ensure model uses 16kHz sample rate
- Verify weight shapes match expected dimensions
- Test with `test_weight_loading.py` after conversion

### Common Pitfalls

1. **Forgetting to transpose Conv1d weights** - causes shape mismatches
2. **Using incorrect LSTM bias format** - must combine ih and hh biases
3. **Applying abs() to all SincNet layers** - only first layer needs abs()
4. **Not reversing backward LSTM output** - breaks temporal alignment
5. **Using MaxPool2d instead of MaxPool1d** - dimensionality error

## Output Format

RTTM (Rich Transcription Time Marked) files follow this format:
```
SPEAKER filename 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
```

Example:
```
SPEAKER audio 1 0.000 2.500 <NA> <NA> speaker_2 <NA> <NA>
SPEAKER audio 1 2.500 1.800 <NA> <NA> speaker_1 <NA> <NA>
```

## Dependencies

Core runtime dependencies:
- `mlx>=0.20.0` - Apple's ML framework
- `numpy>=1.21.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing
- `librosa>=0.10.0` - Audio processing
- `soundfile>=0.12.1` - Audio I/O
- `torchaudio>=2.0.0` - Audio loading (no PyTorch inference)

Conversion dependencies (optional):
- `torch>=2.0.0` - Only needed for model conversion
- `transformers>=4.30.0` - Hugging Face model loading
- `huggingface-hub>=0.16.0` - Model downloading
