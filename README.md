# Speaker Diarization Community 1 - MLX Implementation

An MLX implementation of the pyannote/segmentation-3.0 model, optimized for Apple Silicon.

## Overview

This project ports the [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) model to MLX (Apple's machine learning framework), enabling efficient speaker diarization on Apple Silicon devices (M1/M2/M3/M4).

## Repository Structure

```
speaker-diarization-community-1-mlx/
├── src/                    # Main package code
│   ├── models.py          # MLX model implementations
│   ├── pipeline.py        # Speaker diarization pipeline
│   └── ...
├── scripts/               # Utility scripts
│   ├── convert_pyannote.py
│   └── upload_to_hf.py
├── analysis/              # Analysis and validation scripts
│   ├── compare_*.py
│   ├── debug_*.py
│   └── test_*.py
├── experiments/           # Experimental scripts
├── examples/              # Usage examples
├── tests/                 # Unit tests
├── models/                # Model weights and configs
├── docs/                  # Documentation
└── diarize.py            # Main CLI script
```

## Features

- **Apple Silicon Optimized**: Leverages MLX for efficient computation on Apple Neural Engine
- **Unified Memory**: Uses MLX's unified memory model for seamless device operations
- **Complete Implementation**: Full SincNet + Bidirectional LSTM architecture ported from PyTorch
- **Real-time Capable**: Process audio efficiently with chunked inference
- **RTTM Output**: Standard format compatible with evaluation tools

## Architecture

The segmentation model consists of:

1. **SincNet Frontend**: 3-layer learnable bandpass filter bank for raw waveform processing
   - 80 learnable filters with sinc function constraints
   - Processes raw audio directly (no MFCC/spectrogram needed)
   
2. **Bidirectional LSTM**: 4-layer bidirectional LSTM (128 hidden units per direction)
   - Manual implementation (MLX doesn't have native bidirectional LSTM)
   - 256-dim output per frame (128 forward + 128 backward)

3. **Classification Head**: Linear layers + 7-class speaker classifier
   - Frame-level predictions (~17ms resolution)
   - Outputs speaker activity probabilities

**Model Stats**: 1,473,515 parameters | 5.6MB weights file

## Installation

```bash
# Clone the repository
git clone https://github.com/billymolina/speaker-diarization-community-1-mlx.git
cd speaker-diarization-community-1-mlx

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Directory Guide

| Directory | Purpose |
|-----------|---------|
| [`src/`](src/) | Main package code (models, pipeline, utilities) |
| [`scripts/`](scripts/) | Utility scripts (conversion, deployment) |
| [`analysis/`](analysis/) | Analysis and validation scripts |
| [`experiments/`](experiments/) | Experimental and research code |
| [`examples/`](examples/) | Usage examples and tutorials |
| [`tests/`](tests/) | Unit tests |
| [`models/`](models/) | Model weights and configurations |
| [`docs/`](docs/) | Documentation and analysis reports |

## Quick Start

### Command Line Interface

Process an audio file and generate RTTM output:

```bash
python diarize.py audio.wav --output results.rttm
```

Options:
- `--model`: Path to model weights (default: `models/segmentation_mlx/weights.npz`)
- `--min-duration`: Minimum segment duration in seconds (default: 0.5)
- `--chunk-size`: Chunk size in samples for processing (default: 160000 = 10s)

### Python API

```python
from src.models import load_pyannote_model
import mlx.core as mx
import torchaudio

# Load model
model = load_pyannote_model("models/segmentation_mlx/weights.npz")

# Load audio
waveform, sr = torchaudio.load("audio.wav")
audio_mx = mx.array(waveform.numpy(), dtype=mx.float32)

# Run inference
logits = model(audio_mx)

# Apply softmax to get probabilities
probs = mx.softmax(logits, axis=-1)

# Get predicted speaker per frame
predictions = mx.argmax(probs, axis=-1)
```

### Complete Pipeline Example

See [`examples/complete_example.py`](examples/complete_example.py) for a full pipeline example.
```

## Model Conversion

The pyannote/segmentation-3.0 model has been pre-converted and is included in `models/segmentation_mlx/weights.npz`.

To convert other pyannote models:

```bash
python scripts/convert_pyannote.py --repo-id pyannote/segmentation-3.0 --output models/custom_mlx
```

## Testing

Run the test suite to verify the implementation:

```bash
# Test model architecture and forward pass
python test_pyannote_model.py

# Test weight loading  
python test_weight_loading.py

# Test real audio inference
python test_real_inference.py
```

All tests pass (5/5):
- ✅ SincNet filter generation
- ✅ SincNet frontend (3 layers)
- ✅ Complete model forward pass
- ✅ Weight loading (1,473,515 parameters)
- ✅ Real audio inference

## Performance

Tested on Apple Silicon:
- **Model size**: 5.6MB (1.47M parameters)
- **Frame resolution**: ~17ms per prediction
- **Throughput**: Processes 77-minute audio efficiently
- **Memory efficient**: Handles long files via chunked processing

## Output Format

The diarization script generates RTTM (Rich Transcription Time Marked) files:

```
SPEAKER filename 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
```

Example output:
```
SPEAKER audio 1 0.000 2.500 <NA> <NA> speaker_2 <NA> <NA>
SPEAKER audio 1 2.500 1.800 <NA> <NA> speaker_1 <NA> <NA>
```

## Project Structure

```
speaker-diarization-community-1-mlx/
├── src/
│   ├── models.py          # MLX model implementations  
│   ├── convert.py         # PyTorch to MLX conversion
│   ├── pipeline.py        # Diarization pipeline
│   └── audio.py           # Audio processing utilities
├── models/
│   └── segmentation_mlx/
│       └── weights.npz    # Converted model weights (5.6MB)
├── tests/
│   └── evaluation/        # Test audio and ground truth
├── diarize.py            # CLI inference script
├── convert_pyannote.py   # Model conversion script
├── test_pyannote_model.py # Architecture tests (5/5 passing)
├── test_weight_loading.py # Weight loading tests
├── test_real_inference.py # Real audio tests
└── README.md
```

## Implementation Details

### MLX API Differences from PyTorch

Key adaptations made for MLX compatibility:

1. **Conv1D dimension ordering**:
   - PyTorch: `(batch, channels, length)` and `(out_channels, in_channels, kernel)`
   - MLX: `(batch, length, channels)` and `(out_channels, kernel, in_channels)`

2. **No native `flip()` function**:
   - Replaced with array slicing: `array[::-1]`

3. **No bidirectional LSTM**:
   - Manual implementation with forward + backward passes
   - Time reversal using array slicing

4. **InstanceNorm format**:
   - Expects channels in last dimension: `(..., channels)`

### SincNet Implementation

The SincNet layer implements learnable bandpass filters using the sinc function:

```
h_BP(t) = 2*f_high*sinc(2*f_high*t) - 2*f_low*sinc(2*f_low*t)
```

- 80 learnable filters initialized on mel-scale
- Kernel size: 251 samples  
- Stride: 10 samples
- Constraints ensure valid frequency ranges

## License

This project is based on pyannote.audio. Please see the original project for licensing information.

## Citation

If you use this implementation, please cite the original pyannote.audio work:

```bibtex
@inproceedings{Bredin2020,
  Title = {{pyannote.audio: neural building blocks for speaker diarization}},
  Author = {Bredin, Herv{\'e} and Yin, Ruiqing and Coria, Juan Manuel and Gelly, Gregory and Korshunov, Pavel and Lavechin, Marvin and Fustes, Diego and Titeux, Hadrien and Bouaziz, Wassim and Gill, Marie-Philippe},
  Booktitle = {ICASSP 2020, IEEE International Conference on Acoustics, Speech, and Signal Processing},
  Year = {2020},
}
```

## Acknowledgments

- [pyannote.audio](https://github.com/pyannote/pyannote-audio) for the original model
- [MLX](https://github.com/ml-explore/mlx) team at Apple for the framework


- Original pyannote model by Hervé Bredin and team
- MLX framework by Apple ML Research
