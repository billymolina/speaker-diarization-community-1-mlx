#!/usr/bin/env python3
"""
Complete example demonstrating speaker diarization with pyannote MLX model.
"""

import mlx.core as mx
import numpy as np
import torchaudio
from pathlib import Path
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import load_pyannote_model

def example_basic_inference():
    """Basic inference example with synthetic audio."""
    print("\n" + "="*70)
    print("Example 1: Basic Inference")
    print("="*70)
    
    # Load model
    print("\n[1] Loading model...")
    model = load_pyannote_model("models/segmentation_mlx/weights.npz")
    print("✓ Model loaded")
    
    # Create synthetic audio (5 seconds at 16kHz)
    print("\n[2] Generating synthetic audio...")
    duration = 5.0
    sr = 16000
    samples = int(duration * sr)
    
    # Mix of frequencies to simulate speech
    t = np.linspace(0, duration, samples)
    audio = np.sin(2 * np.pi * 200 * t) * 0.3  # Low frequency
    audio += np.sin(2 * np.pi * 800 * t) * 0.2  # Mid frequency
    audio += np.random.randn(samples) * 0.05  # Noise
    
    audio_mx = mx.array(audio.reshape(1, -1), dtype=mx.float32)
    print(f"✓ Generated {duration}s audio")
    
    # Run inference
    print("\n[3] Running inference...")
    logits = model(audio_mx)
    print(f"✓ Output shape: {logits.shape}")
    print(f"  Batch: {logits.shape[0]}")
    print(f"  Time frames: {logits.shape[1]}")
    print(f"  Classes: {logits.shape[2]}")
    
    # Get probabilities
    probs = mx.softmax(logits, axis=-1)
    predictions = mx.argmax(probs[0], axis=-1)
    
    print(f"\n[4] Results:")
    print(f"  Predicted speaker: Class {int(predictions[0])}")
    print(f"  Frame resolution: {duration / logits.shape[1] * 1000:.1f} ms")
    
    # Show probability distribution
    avg_probs = mx.mean(probs[0], axis=0)
    print(f"\n  Average class probabilities:")
    for i in range(7):
        bar_length = int(float(avg_probs[i]) * 50)
        bar = "█" * bar_length
        print(f"    Class {i}: {bar} {float(avg_probs[i]):.4f}")
    
    print("\n✅ Basic inference completed!\n")

def example_real_audio():
    """Process real audio file."""
    print("\n" + "="*70)
    print("Example 2: Real Audio Processing")
    print("="*70)
    
    audio_path = "/Users/billymolina/Github/3D-Speaker/evaluation/wavs/Use-Case_Support-20260112_094222-Besprechungsaufzeichnung.wav"
    
    if not Path(audio_path).exists():
        print("\n⚠️  Test audio file not found, skipping this example")
        return
    
    # Load model
    print("\n[1] Loading model...")
    model = load_pyannote_model("models/segmentation_mlx/weights.npz")
    print("✓ Model loaded")
    
    # Load audio
    print(f"\n[2] Loading audio from file...")
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Take first 10 seconds
    duration = 10.0
    samples = int(duration * sr)
    waveform = waveform[:, :samples]
    
    print(f"✓ Loaded {duration}s audio")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Samples: {waveform.shape[1]:,}")
    
    # Convert to MLX
    audio_mx = mx.array(waveform.numpy(), dtype=mx.float32)
    
    # Run inference
    print(f"\n[3] Running inference...")
    logits = model(audio_mx)
    probs = mx.softmax(logits, axis=-1)
    predictions = mx.argmax(probs[0], axis=-1)
    
    print(f"✓ Generated {logits.shape[1]} predictions")
    
    # Analyze predictions
    print(f"\n[4] Speaker Activity Analysis:")
    
    # Count frames per speaker
    unique_speakers = set()
    for i in range(logits.shape[1]):
        unique_speakers.add(int(predictions[i]))
    
    print(f"  Detected {len(unique_speakers)} active speakers")
    
    for speaker in sorted(unique_speakers):
        count = int(mx.sum(predictions == speaker))
        percentage = (count / logits.shape[1]) * 100
        time = (count / logits.shape[1]) * duration
        
        bar_length = int(percentage / 2)
        bar = "█" * bar_length
        print(f"  Speaker {speaker}: {bar} {percentage:5.1f}% ({time:.2f}s)")
    
    # Show confidence statistics
    max_probs = mx.max(probs[0], axis=-1)
    avg_confidence = float(mx.mean(max_probs))
    print(f"\n  Average prediction confidence: {avg_confidence:.4f}")
    
    print("\n✅ Real audio processing completed!\n")

def example_batch_processing():
    """Process multiple audio clips in a batch."""
    print("\n" + "="*70)
    print("Example 3: Batch Processing")
    print("="*70)
    
    # Load model
    print("\n[1] Loading model...")
    model = load_pyannote_model("models/segmentation_mlx/weights.npz")
    print("✓ Model loaded")
    
    # Create batch of 3 synthetic audio clips
    print("\n[2] Generating batch of audio clips...")
    batch_size = 3
    duration = 5.0
    sr = 16000
    samples = int(duration * sr)
    
    # Generate different audio patterns
    batch = []
    for i in range(batch_size):
        t = np.linspace(0, duration, samples)
        freq = 200 + i * 300  # Different frequency per clip
        audio = np.sin(2 * np.pi * freq * t) * 0.3
        audio += np.random.randn(samples) * 0.05
        batch.append(audio)
    
    batch_mx = mx.array(np.array(batch), dtype=mx.float32)
    print(f"✓ Created batch: {batch_mx.shape}")
    
    # Process batch (note: current implementation processes one at a time)
    print(f"\n[3] Processing batch...")
    results = []
    for i in range(batch_size):
        logits = model(batch_mx[i:i+1])
        probs = mx.softmax(logits, axis=-1)
        predictions = mx.argmax(probs[0], axis=-1)
        results.append(int(predictions[0]))
    
    print(f"✓ Processed {batch_size} clips")
    
    # Show results
    print(f"\n[4] Results:")
    for i, pred in enumerate(results):
        print(f"  Clip {i+1}: Predicted speaker = Class {pred}")
    
    print("\n✅ Batch processing completed!\n")

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Pyannote MLX - Complete Usage Examples")
    print("="*70)
    print("\nThis script demonstrates various use cases:")
    print("  1. Basic inference with synthetic audio")
    print("  2. Processing real audio files")
    print("  3. Batch processing multiple clips")
    
    try:
        # Run examples
        example_basic_inference()
        example_real_audio()
        example_batch_processing()
        
        print("\n" + "="*70)
        print("🎉 All examples completed successfully!")
        print("="*70)
        print("\nNext steps:")
        print("  • Use diarize.py for full audio file processing")
        print("  • See README.md for more usage examples")
        print("  • Check tests/ directory for more test cases")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
