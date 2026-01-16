#!/usr/bin/env python3
"""
Speaker diarization inference script using pyannote MLX model.
"""

import mlx.core as mx
import numpy as np
import torchaudio
from pathlib import Path
import argparse
import sys
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import load_pyannote_model

def load_audio(path: str, target_sr: int = 16000):
    """Load audio file and resample to target sample rate."""
    waveform, sr = torchaudio.load(path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    return waveform.numpy(), target_sr

def process_audio_chunks(audio: np.ndarray, model, chunk_size: int = 160000, overlap: int = 16000):
    """
    Process audio in chunks with overlap to handle long files.
    
    Args:
        audio: Audio array (1, samples)
        model: Pyannote model
        chunk_size: Size of each chunk in samples (default 10s at 16kHz)
        overlap: Overlap between chunks in samples (default 1s)
    
    Returns:
        logits: Concatenated predictions
        frame_times: Start time of each frame in seconds
    """
    total_samples = audio.shape[1]
    sr = 16000
    
    all_logits = []
    frame_times = []
    
    start = 0
    current_time = 0.0
    
    while start < total_samples:
        end = min(start + chunk_size, total_samples)
        chunk = audio[:, start:end]
        
        # Pad if needed
        if chunk.shape[1] < chunk_size:
            pad_size = chunk_size - chunk.shape[1]
            chunk = np.pad(chunk, ((0, 0), (0, pad_size)), mode='constant')
        
        # Convert to MLX and run inference
        chunk_mx = mx.array(chunk, dtype=mx.float32)
        logits = model(chunk_mx)
        
        # Store logits (remove overlap region except for first chunk)
        if start == 0:
            all_logits.append(logits[0])
            # Calculate frame times
            frame_duration = (end - start) / sr / logits.shape[1]
            for i in range(logits.shape[1]):
                frame_times.append(current_time + i * frame_duration)
        else:
            # Skip overlap frames
            overlap_frames = int(logits.shape[1] * overlap / chunk_size)
            all_logits.append(logits[0, overlap_frames:])
            frame_duration = (end - start) / sr / logits.shape[1]
            for i in range(overlap_frames, logits.shape[1]):
                frame_times.append(current_time + (i - overlap_frames) * frame_duration)
        
        # Move to next chunk
        start += (chunk_size - overlap)
        current_time = start / sr
    
    # Concatenate all predictions
    full_logits = mx.concatenate(all_logits, axis=0)
    
    return full_logits, np.array(frame_times)

def logits_to_segments(logits: mx.array, frame_times: np.ndarray, 
                       min_duration: float = 0.5) -> List[Tuple[float, float, int]]:
    """
    Convert frame-level predictions to speaker segments.
    
    Args:
        logits: Model logits (time_frames, num_classes)
        frame_times: Start time of each frame
        min_duration: Minimum segment duration in seconds
    
    Returns:
        List of (start_time, end_time, speaker_id) tuples
    """
    # Apply softmax to get probabilities
    probs = mx.softmax(logits, axis=-1)
    
    # Get predicted class for each frame
    predictions = mx.argmax(probs, axis=-1)
    confidences = mx.max(probs, axis=-1)
    
    # Convert to numpy for easier processing
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    
    # Build segments
    segments = []
    current_speaker = int(predictions[0])
    start_time = frame_times[0]
    
    for i in range(1, len(predictions)):
        # If speaker changes, close current segment
        if predictions[i] != current_speaker:
            end_time = frame_times[i]
            duration = end_time - start_time
            
            # Only keep segments longer than minimum duration
            if duration >= min_duration:
                segments.append((start_time, end_time, current_speaker))
            
            # Start new segment
            current_speaker = int(predictions[i])
            start_time = frame_times[i]
    
    # Add final segment
    end_time = frame_times[-1]
    duration = end_time - start_time
    if duration >= min_duration:
        segments.append((start_time, end_time, current_speaker))
    
    return segments

def save_rttm(segments: List[Tuple[float, float, int]], output_path: str, audio_id: str = "audio"):
    """
    Save segments in RTTM format.
    
    RTTM format: SPEAKER <file-id> 1 <start-time> <duration> <NA> <NA> <speaker-id> <NA> <NA>
    """
    with open(output_path, 'w') as f:
        for start, end, speaker in segments:
            duration = end - start
            line = f"SPEAKER {audio_id} 1 {start:.3f} {duration:.3f} <NA> <NA> speaker_{speaker} <NA> <NA>\n"
            f.write(line)

def main():
    parser = argparse.ArgumentParser(description="Speaker diarization with pyannote MLX")
    parser.add_argument("audio", type=str, help="Input audio file")
    parser.add_argument("--output", "-o", type=str, help="Output RTTM file (default: same as input with .rttm)")
    parser.add_argument("--model", type=str, default="models/segmentation_mlx/weights.npz",
                       help="Path to model weights")
    parser.add_argument("--min-duration", type=float, default=0.5,
                       help="Minimum segment duration in seconds")
    parser.add_argument("--chunk-size", type=int, default=160000,
                       help="Chunk size in samples (default: 10s at 16kHz)")
    
    args = parser.parse_args()
    
    # Setup output path
    if args.output is None:
        args.output = Path(args.audio).with_suffix('.rttm')
    
    print(f"\n{'='*70}")
    print(f"Speaker Diarization - Pyannote MLX")
    print(f"{'='*70}")
    
    # Load model
    print(f"\n[INFO] Loading model from {args.model}")
    model = load_pyannote_model(args.model)
    print("[INFO] Model loaded successfully!")
    
    # Load audio
    print(f"\n[INFO] Loading audio: {args.audio}")
    waveform, sr = load_audio(args.audio)
    duration = waveform.shape[1] / sr
    print(f"  Duration: {duration:.1f} seconds")
    print(f"  Sample rate: {sr} Hz")
    
    # Process audio
    print(f"\n[INFO] Running inference...")
    logits, frame_times = process_audio_chunks(waveform, model, chunk_size=args.chunk_size)
    print(f"  Generated {logits.shape[0]} time frames")
    print(f"  Frame resolution: {(frame_times[1] - frame_times[0])*1000:.1f} ms")
    
    # Convert to segments
    print(f"\n[INFO] Generating speaker segments...")
    segments = logits_to_segments(logits, frame_times, min_duration=args.min_duration)
    print(f"  Found {len(segments)} segments")
    
    # Print segment summary
    if segments:
        print(f"\n[INFO] Speaker activity:")
        speakers = set(s[2] for s in segments)
        for speaker in sorted(speakers):
            speaker_segs = [s for s in segments if s[2] == speaker]
            total_time = sum(s[1] - s[0] for s in speaker_segs)
            percentage = (total_time / duration) * 100
            print(f"  Speaker {speaker}: {total_time:6.1f}s ({percentage:5.1f}%) - {len(speaker_segs)} segments")
    
    # Save RTTM
    audio_id = Path(args.audio).stem
    save_rttm(segments, args.output, audio_id)
    print(f"\n[INFO] Saved RTTM to: {args.output}")
    
    print(f"\n{'='*70}")
    print("✅ Diarization completed successfully!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
