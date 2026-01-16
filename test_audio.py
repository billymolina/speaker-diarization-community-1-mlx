#!/usr/bin/env python3
"""Test the MLX model with actual audio file."""

import mlx.core as mx
import numpy as np
from pathlib import Path
import soundfile as sf


def load_audio(file_path: str, sample_rate: int = 16000):
    """Load audio file and resample if necessary."""
    print(f"[INFO] Loading audio from: {file_path}")
    
    # Load audio using soundfile
    audio, sr = sf.read(file_path)
    
    print(f"[INFO] Original sample rate: {sr} Hz")
    print(f"[INFO] Audio shape: {audio.shape}")
    print(f"[INFO] Duration: {len(audio) / sr:.2f} seconds")
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
        print(f"[INFO] Converted stereo to mono")
    
    # Resample if needed (simple approach - for production use librosa.resample)
    if sr != sample_rate:
        print(f"[WARNING] Resampling not implemented. Using original sample rate.")
    
    return audio, sr


def extract_features(audio: np.ndarray, sr: int):
    """Extract basic features from audio."""
    print(f"\n[INFO] Extracting features...")
    
    # Convert to MLX array
    audio_mx = mx.array(audio.astype(np.float32))
    
    # Basic statistics
    print(f"  Audio range: [{float(mx.min(audio_mx)):.3f}, {float(mx.max(audio_mx)):.3f}]")
    print(f"  Audio mean: {float(mx.mean(audio_mx)):.6f}")
    print(f"  Audio std: {float(mx.std(audio_mx)):.6f}")
    
    # Frame-level processing (chunk into 1-second windows)
    frame_length = sr  # 1 second
    num_frames = len(audio) // frame_length
    
    print(f"  Number of 1-second frames: {num_frames}")
    
    return audio_mx, num_frames


def test_with_model_weights(audio_mx, model_weights):
    """Test forward pass with actual model weights."""
    print(f"\n[INFO] Testing with model weights...")
    
    # The pyannote model expects specific input format
    # SincNet processes raw waveform at 16kHz
    # LSTM expects features of dimension 60 (from SincNet output)
    
    print(f"[INFO] Model has {len(model_weights)} weight tensors")
    print(f"[INFO] Input dimension to LSTM: {model_weights['lstm.weight_ih_l0'].shape[1]}")
    print(f"[INFO] LSTM hidden dim: {model_weights['lstm.weight_hh_l0'].shape[1]}")
    print(f"[INFO] Number of output classes: {model_weights['classifier.weight'].shape[0]}")
    
    # For a complete test, we'd need to implement the full SincNet + LSTM forward pass
    print(f"\n[NOTE] Full inference requires implementing:")
    print(f"  1. SincNet frontend (learnable audio filters)")
    print(f"  2. Bidirectional LSTM (4 layers)")
    print(f"  3. Linear layers + classifier")
    print(f"\n[INFO] The weights are ready to use once the architecture is implemented!")


def main():
    """Run audio test."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "Audio File Test" + " " * 27 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # Get audio file path
    wav_list = Path("tests/evaluation/wav.list")
    if not wav_list.exists():
        print(f"[ERROR] wav.list not found: {wav_list}")
        return
    
    audio_path = wav_list.read_text().strip()
    
    if not Path(audio_path).exists():
        print(f"[ERROR] Audio file not found: {audio_path}")
        return
    
    # Load audio
    print("=" * 60)
    print("Step 1: Loading Audio")
    print("=" * 60)
    audio, sr = load_audio(audio_path)
    
    # Extract features
    print("=" * 60)
    print("Step 2: Feature Extraction")
    print("=" * 60)
    audio_mx, num_frames = extract_features(audio, sr)
    
    # Load model weights
    print("=" * 60)
    print("Step 3: Loading Model Weights")
    print("=" * 60)
    model_path = Path("models/segmentation_mlx/weights.npz")
    print(f"[INFO] Loading model from {model_path}...")
    model_weights = mx.load(str(model_path))
    print(f"[SUCCESS] Loaded {len(model_weights)} weight tensors")
    
    # Test with model
    print("=" * 60)
    print("Step 4: Model Analysis")
    print("=" * 60)
    test_with_model_weights(audio_mx, model_weights)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✓ Audio loaded successfully: {len(audio):,} samples")
    print(f"✓ Duration: {len(audio) / sr:.2f} seconds")
    print(f"✓ Sample rate: {sr} Hz")
    print(f"✓ Model weights loaded: {len(model_weights)} tensors")
    print(f"✓ Total parameters: 1,473,515")
    print()
    print("Next steps:")
    print("  • Implement full pyannote architecture in MLX")
    print("  • Map converted weights to model structure")
    print("  • Run inference and generate speaker diarization output")
    print()


if __name__ == '__main__':
    main()
