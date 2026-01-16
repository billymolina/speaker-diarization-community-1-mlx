#!/usr/bin/env python3
"""
Compare MLX and PyTorch RTTM outputs.
"""

from collections import defaultdict
from pathlib import Path

def parse_rttm(filepath):
    """Parse RTTM file and return segments."""
    segments = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            speaker = parts[7]
            start = float(parts[3])
            duration = float(parts[4])
            end = start + duration
            segments.append((start, end, speaker))
    return segments

def calculate_statistics(segments, total_duration):
    """Calculate speaker statistics."""
    speaker_times = defaultdict(float)
    speaker_counts = defaultdict(int)
    
    for start, end, speaker in segments:
        speaker_times[speaker] += (end - start)
        speaker_counts[speaker] += 1
    
    stats = {}
    for speaker in speaker_times:
        stats[speaker] = {
            'time': speaker_times[speaker],
            'percentage': (speaker_times[speaker] / total_duration) * 100,
            'segments': speaker_counts[speaker],
            'avg_duration': speaker_times[speaker] / speaker_counts[speaker]
        }
    
    return stats

def compare_outputs():
    """Compare MLX and PyTorch outputs."""
    
    print("\n" + "="*70)
    print("MLX vs PyTorch Comparison")
    print("="*70)
    
    mlx_file = "test_full_diarization.rttm"
    pytorch_file = "pytorch_output.rttm"
    total_duration = 4666.2
    
    if not Path(mlx_file).exists():
        print(f"❌ MLX output not found: {mlx_file}")
        return
    
    if not Path(pytorch_file).exists():
        print(f"❌ PyTorch output not found: {pytorch_file}")
        return
    
    # Parse both files
    print("\n[1] Parsing RTTM files...")
    mlx_segments = parse_rttm(mlx_file)
    pytorch_segments = parse_rttm(pytorch_file)
    
    print(f"  MLX segments: {len(mlx_segments)}")
    print(f"  PyTorch segments: {len(pytorch_segments)}")
    
    # Calculate statistics
    print("\n[2] Calculating statistics...")
    mlx_stats = calculate_statistics(mlx_segments, total_duration)
    pytorch_stats = calculate_statistics(pytorch_segments, total_duration)
    
    # Speaker mapping (by most similar time allocation)
    print("\n[3] Speaker Activity Comparison:")
    print("-" * 70)
    print(f"{'Speaker':<15} {'MLX Time':>12} {'MLX %':>8} {'PyTorch Time':>15} {'PyTorch %':>10}")
    print("-" * 70)
    
    # Get all unique speaker IDs
    all_mlx_speakers = sorted(mlx_stats.keys())
    all_pytorch_speakers = sorted(pytorch_stats.keys())
    
    print("\nMLX Output:")
    for speaker in all_mlx_speakers:
        s = mlx_stats[speaker]
        print(f"  {speaker:<13} {s['time']:>10.1f}s {s['percentage']:>7.1f}%  ({s['segments']} segments)")
    
    print("\nPyTorch Output:")
    for speaker in all_pytorch_speakers:
        s = pytorch_stats[speaker]
        print(f"  {speaker:<13} {s['time']:>10.1f}s {s['percentage']:>7.1f}%  ({s['segments']} segments)")
    
    # Calculate overlap metrics
    print("\n[4] Overlap Analysis:")
    print("-" * 70)
    
    # Create time-based representation (1 frame = 100ms)
    frame_duration = 0.1
    num_frames = int(total_duration / frame_duration) + 1
    
    mlx_frames = ['none'] * num_frames
    pytorch_frames = ['none'] * num_frames
    
    for start, end, speaker in mlx_segments:
        start_frame = int(start / frame_duration)
        end_frame = int(end / frame_duration)
        for i in range(start_frame, min(end_frame, num_frames)):
            mlx_frames[i] = speaker
    
    for start, end, speaker in pytorch_segments:
        start_frame = int(start / frame_duration)
        end_frame = int(end / frame_duration)
        for i in range(start_frame, min(end_frame, num_frames)):
            pytorch_frames[i] = speaker
    
    # Calculate frame-level agreement
    matching_frames = sum(1 for m, p in zip(mlx_frames, pytorch_frames) if m != 'none' and p != 'none' and m == p)
    total_active_frames = sum(1 for m, p in zip(mlx_frames, pytorch_frames) if m != 'none' or p != 'none')
    
    if total_active_frames > 0:
        agreement = (matching_frames / total_active_frames) * 100
        print(f"  Frame-level agreement: {agreement:.1f}%")
        print(f"  Matching frames: {matching_frames:,} / {total_active_frames:,}")
    
    # Calculate boundary differences
    mlx_boundaries = set()
    for start, end, _ in mlx_segments:
        mlx_boundaries.add(round(start, 2))
        mlx_boundaries.add(round(end, 2))
    
    pytorch_boundaries = set()
    for start, end, _ in pytorch_segments:
        pytorch_boundaries.add(round(start, 2))
        pytorch_boundaries.add(round(end, 2))
    
    common_boundaries = mlx_boundaries & pytorch_boundaries
    print(f"\n  Common boundaries: {len(common_boundaries)}")
    print(f"  MLX unique boundaries: {len(mlx_boundaries - pytorch_boundaries)}")
    print(f"  PyTorch unique boundaries: {len(pytorch_boundaries - mlx_boundaries)}")
    
    # Summary
    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    print(f"  Total segments: MLX={len(mlx_segments)}, PyTorch={len(pytorch_segments)}")
    print(f"  Number of speakers: MLX={len(mlx_stats)}, PyTorch={len(pytorch_stats)}")
    print(f"  Segment difference: {abs(len(mlx_segments) - len(pytorch_segments))} ({abs(len(mlx_segments) - len(pytorch_segments)) / max(len(mlx_segments), len(pytorch_segments)) * 100:.1f}%)")
    
    if total_active_frames > 0:
        print(f"  Frame agreement: {agreement:.1f}%")
    
    print("\n✅ Comparison complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    compare_outputs()
