#!/usr/bin/env python3
"""
Test script to verify weight loading from converted pyannote model.
"""

import mlx.core as mx
import numpy as np
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import load_pyannote_model

def test_weight_loading():
    """Test loading weights and running inference."""
    print("\n" + "="*70)
    print("Testing Weight Loading and Inference")
    print("="*70)
    
    # Load model with weights
    weights_path = "models/segmentation_mlx/weights.npz"
    print(f"\n[INFO] Loading model from {weights_path}")
    
    model = load_pyannote_model(weights_path)
    
    print("\n[INFO] Model loaded successfully!")
    print(f"  SincNet low_hz_ shape: {model.sincnet.sinc_conv.low_hz_.shape}")
    print(f"  LSTM forward layer 0 Wx shape: {model.lstm_forward[0].Wx.shape}")
    print(f"  Classifier weight shape: {model.classifier.weight.shape}")
    
    # Test with random audio
    print("\n[INFO] Testing inference with 5 seconds of audio...")
    audio = mx.random.normal((1, 16000 * 5)) * 0.1
    
    # Run inference
    logits = model(audio)
    
    print(f"\n[INFO] Inference successful!")
    print(f"  Output shape: {logits.shape}")
    print(f"  Output range: [{float(mx.min(logits)):.3f}, {float(mx.max(logits)):.3f}]")
    
    # Apply softmax to get probabilities
    probs = mx.softmax(logits, axis=-1)
    
    print(f"\n[INFO] Class probabilities (first frame):")
    for i in range(7):
        print(f"  Class {i}: {float(probs[0, 0, i]):.4f}")
    
    print("\n✅ Weight loading test passed!")
    return True

if __name__ == "__main__":
    try:
        test_weight_loading()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
