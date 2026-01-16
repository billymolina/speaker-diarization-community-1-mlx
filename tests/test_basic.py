"""
Tests for speaker diarization components
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mlx.core as mx
from diarization_types import Turn, SpeakerDiarization
from audio import AudioProcessor
from models import SegmentationModel, EmbeddingModel
from clustering import SpeakerClusterer


def test_turn():
    """Test Turn data structure"""
    print("Testing Turn...")
    turn = Turn(start=1.5, end=3.7)
    assert turn.start == 1.5
    assert turn.end == 3.7
    assert abs(turn.duration - 2.2) < 0.01
    print("✓ Turn tests passed")


def test_speaker_diarization():
    """Test SpeakerDiarization data structure"""
    print("\nTesting SpeakerDiarization...")
    diarization = SpeakerDiarization()
    diarization.add_segment(0.0, 2.5, "SPEAKER_00")
    diarization.add_segment(2.5, 5.0, "SPEAKER_01")
    
    segments = list(diarization.speaker_diarization)
    assert len(segments) == 2
    assert segments[0][1] == "SPEAKER_00"
    assert segments[1][1] == "SPEAKER_01"
    print("✓ SpeakerDiarization tests passed")


def test_audio_processor():
    """Test AudioProcessor"""
    print("\nTesting AudioProcessor...")
    processor = AudioProcessor()
    
    # Test waveform preprocessing
    waveform = np.random.randn(16000)  # 1 second of audio at 16kHz
    processed = processor.preprocess_audio(waveform)
    assert np.max(np.abs(processed)) <= 1.0
    
    # Test MFCC computation
    mfcc = processor.compute_mfcc(waveform)
    assert mfcc.shape[0] == 13  # 13 MFCC coefficients
    
    # Test mel spectrogram
    mel_spec = processor.compute_mel_spectrogram(waveform)
    assert mel_spec.shape[0] == 128  # 128 mel bands
    
    print("✓ AudioProcessor tests passed")


def test_segmentation_model():
    """Test SegmentationModel"""
    print("\nTesting SegmentationModel...")
    model = SegmentationModel(input_dim=128, hidden_dim=64)
    
    # Create dummy input
    batch_size = 2
    seq_len = 100
    input_dim = 128
    x = mx.random.normal((batch_size, seq_len, input_dim))
    
    # Forward pass
    activity, change = model(x)
    
    assert activity.shape == (batch_size, seq_len, 1)
    assert change.shape == (batch_size, seq_len, 1)
    print("✓ SegmentationModel tests passed")


def test_embedding_model():
    """Test EmbeddingModel"""
    print("\nTesting EmbeddingModel...")
    model = EmbeddingModel(input_dim=128, hidden_dim=64, embedding_dim=256)
    
    # Create dummy input
    batch_size = 2
    seq_len = 100
    input_dim = 128
    x = mx.random.normal((batch_size, seq_len, input_dim))
    
    # Forward pass
    embeddings = model(x)
    
    assert embeddings.shape == (batch_size, 256)
    
    # Check L2 normalization
    norms = mx.linalg.norm(embeddings, axis=1)
    assert mx.allclose(norms, mx.ones(batch_size), atol=1e-5)
    
    print("✓ EmbeddingModel tests passed")


def test_speaker_clusterer():
    """Test SpeakerClusterer"""
    print("\nTesting SpeakerClusterer...")
    clusterer = SpeakerClusterer(threshold=0.7)
    
    # Create dummy embeddings (3 groups)
    embeddings = np.concatenate([
        np.random.randn(5, 256) + np.array([1.0, 0.0] + [0.0] * 254),  # Group 1
        np.random.randn(5, 256) + np.array([0.0, 1.0] + [0.0] * 254),  # Group 2
        np.random.randn(5, 256) + np.array([0.0, 0.0, 1.0] + [0.0] * 253),  # Group 3
    ])
    
    # Normalize embeddings
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-6)
    
    # Cluster
    labels, n_clusters = clusterer.cluster(embeddings)
    
    assert len(labels) == 15
    assert n_clusters >= 1  # At least one cluster
    
    print(f"  Found {n_clusters} clusters")
    print("✓ SpeakerClusterer tests passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("Running Speaker Diarization Tests")
    print("=" * 70)
    
    try:
        test_turn()
        test_speaker_diarization()
        test_audio_processor()
        test_segmentation_model()
        test_embedding_model()
        test_speaker_clusterer()
        
        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
