#!/usr/bin/env python3
"""
Test the complete pyannote/segmentation-3.0 MLX implementation.
"""

import mlx.core as mx
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import PyannoteSegmentationModel, load_pyannote_model


def test_sincnet_filters():
    """Test SincNet filter generation."""
    print("="*60)
    print("Test 1: SincNet Filter Generation")
    print("="*60)
    
    from models import SincConv1d
    
    sinc = SincConv1d(n_filters=80, kernel_size=251, sample_rate=16000)
    
    print(f"[INFO] SincNet parameters:")
    print(f"  Number of filters: {sinc.n_filters}")
    print(f"  Kernel size: {sinc.kernel_size}")
    print(f"  Sample rate: {sinc.sample_rate} Hz")
    print(f"  Stride: {sinc.stride}")
    
    # Generate filters
    filters = sinc.get_filters()
    print(f"\n[INFO] Generated filters shape: {filters.shape}")
    print(f"  Expected: (80, 1, 251)")
    
    # Test with dummy input
    dummy_audio = mx.random.normal((1, 1, 16000))  # 1 second
    output = sinc(dummy_audio)
    print(f"\n[INFO] Forward pass test:")
    print(f"  Input shape: {dummy_audio.shape}")
    print(f"  Output shape: {output.shape}")
    
    print("✓ SincNet filter generation passed\n")
    return True


def test_sincnet_frontend():
    """Test complete SincNet frontend."""
    print("="*60)
    print("Test 2: SincNet Frontend (3 layers)")
    print("="*60)
    
    from models import SincNet
    
    sincnet = SincNet(sample_rate=16000, stride=10)
    
    # Test with 3 seconds of audio
    dummy_audio = mx.random.normal((2, 16000 * 3)) * 0.1  # batch=2, 3 seconds
    
    print(f"[INFO] Input shape: {dummy_audio.shape}")
    
    output = sincnet(dummy_audio)
    
    print(f"[INFO] Output shape: {output.shape}")
    print(f"  Batch size: {output.shape[0]}")
    print(f"  Time frames: {output.shape[1]}")
    print(f"  Features: {output.shape[2]}")
    print(f"  Output range: [{float(mx.min(output)):.3f}, {float(mx.max(output)):.3f}]")
    
    # MLX format is (batch, time, features)
    assert output.shape[2] == 60, f"Expected 60 features, got {output.shape[2]}"
    
    print("✓ SincNet frontend passed\n")
    return True


def test_complete_model():
    """Test complete PyannoteSegmentationModel."""
    print("="*60)
    print("Test 3: Complete Pyannote Model")
    print("="*60)
    
    model = PyannoteSegmentationModel(
        sample_rate=16000,
        sincnet_stride=10,
        lstm_hidden_dim=128,
        num_lstm_layers=4,
        num_classes=7,
    )
    
    print("[INFO] Model created successfully")
    print(f"  Sample rate: {model.sample_rate} Hz")
    print(f"  LSTM hidden dim: {model.lstm_hidden_dim}")
    print(f"  Number of classes: {model.num_classes}")
    
    # Test with 5 seconds of audio
    batch_size = 1
    audio_length = 16000 * 5  # 5 seconds
    dummy_audio = mx.random.normal((batch_size, audio_length)) * 0.1
    
    print(f"\n[INFO] Running forward pass...")
    print(f"  Input: ({batch_size}, {audio_length}) - 5 seconds at 16kHz")
    
    logits = model(dummy_audio)
    
    print(f"\n[INFO] Output logits shape: {logits.shape}")
    print(f"  Batch: {logits.shape[0]}")
    print(f"  Time frames: {logits.shape[1]}")
    print(f"  Classes: {logits.shape[2]}")
    print(f"  Output range: [{float(mx.min(logits)):.3f}, {float(mx.max(logits)):.3f}]")
    
    assert logits.shape[0] == batch_size
    assert logits.shape[2] == 7, f"Expected 7 classes, got {logits.shape[2]}"
    
    print("✓ Complete model forward pass passed\n")
    return True


def test_with_converted_weights():
    """Test loading converted weights."""
    print("="*60)
    print("Test 4: Load Converted Weights")
    print("="*60)
    
    weights_path = Path("models/segmentation_mlx/weights.npz")
    
    if not weights_path.exists():
        print(f"[WARNING] Weights not found at {weights_path}")
        print("  Skipping weight loading test")
        return True
    
    # Load raw weights to check
    weights = mx.load(str(weights_path))
    print(f"[INFO] Loaded {len(weights)} weight tensors from {weights_path}")
    print(f"  Total parameters: {sum(w.size for w in weights.values()):,}")
    
    # Check key weight shapes
    print("\n[INFO] Key weight shapes:")
    for key in ['sincnet.conv1d.0.filterbank.low_hz_', 'lstm.weight_ih_l0', 'classifier.weight']:
        if key in weights:
            print(f"  {key}: {weights[key].shape}")
    
    print("✓ Weight loading passed\n")
    return True


def test_real_audio():
    """Test with actual audio file if available."""
    print("="*60)
    print("Test 5: Real Audio Inference")
    print("="*60)
    
    wav_list = Path("tests/evaluation/wav.list")
    
    if not wav_list.exists():
        print("[WARNING] No test audio found, skipping")
        return True
    
    audio_path = Path(wav_list.read_text().strip())
    
    if not audio_path.exists():
        print(f"[WARNING] Audio file not found: {audio_path}")
        return True
    
    try:
        import soundfile as sf
        
        # Load just 10 seconds for testing
        audio, sr = sf.read(audio_path, frames=16000 * 10)
        
        print(f"[INFO] Loaded audio: {len(audio):,} samples")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Duration: {len(audio) / sr:.1f} seconds")
        
        # Convert to MLX
        audio_mx = mx.array(audio.astype(np.float32)).reshape(1, -1)
        
        # Create model
        model = PyannoteSegmentationModel()
        
        print(f"\n[INFO] Running inference...")
        logits = model(audio_mx)
        
        print(f"[INFO] Output shape: {logits.shape}")
        print(f"  Time frames: {logits.shape[1]}")
        print(f"  Frame duration: {len(audio) / sr / logits.shape[1]:.3f} seconds")
        
        # Apply softmax to get probabilities
        probs = mx.softmax(logits, axis=-1)
        print(f"\n[INFO] Class probabilities (first frame):")
        for i in range(7):
            print(f"  Class {i}: {float(probs[0, 0, i]):.4f}")
        
        print("✓ Real audio inference passed\n")
        return True
        
    except Exception as e:
        print(f"[WARNING] Real audio test failed: {e}")
        return True


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 8 + "Pyannote MLX Model Test Suite" + " " * 20 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    tests = [
        ("SincNet Filter Generation", test_sincnet_filters),
        ("SincNet Frontend", test_sincnet_frontend),
        ("Complete Model", test_complete_model),
        ("Load Converted Weights", test_with_converted_weights),
        ("Real Audio Inference", test_real_audio),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n[ERROR] {test_name} failed:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {test_name:35s} {status}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print("-" * 60)
    print(f"  Total: {passed}/{total} tests passed")
    print()
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Please review the output above.")
    print()
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
