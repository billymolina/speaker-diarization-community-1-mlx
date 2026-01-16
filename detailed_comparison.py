#!/usr/bin/env python3
"""
Detailed comparison with speaker mapping analysis.
"""

from collections import defaultdict
from pathlib import Path
import numpy as np

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

def analyze_speaker_mapping():
    """Analyze how speakers map between MLX and PyTorch."""
    
    print("\n" + "="*70)
    print("Detailed Speaker Mapping Analysis")
    print("="*70)
    
    mlx_file = "test_full_diarization.rttm"
    pytorch_file = "pytorch_output.rttm"
    total_duration = 4666.2
    
    mlx_segments = parse_rttm(mlx_file)
    pytorch_segments = parse_rttm(pytorch_file)
    
    # Create overlap matrix
    print("\n[1] Creating overlap matrix...")
    
    mlx_speakers = sorted(set(s[2] for s in mlx_segments))
    pytorch_speakers = sorted(set(s[2] for s in pytorch_segments))
    
    # Calculate overlap time between each pair of speakers
    overlap_matrix = np.zeros((len(mlx_speakers), len(pytorch_speakers)))
    
    frame_duration = 0.1  # 100ms frames
    num_frames = int(total_duration / frame_duration) + 1
    
    mlx_frames = [None] * num_frames
    pytorch_frames = [None] * num_frames
    
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
    
    # Count overlaps
    for i in range(num_frames):
        if mlx_frames[i] is not None and pytorch_frames[i] is not None:
            mlx_idx = mlx_speakers.index(mlx_frames[i])
            pytorch_idx = pytorch_speakers.index(pytorch_frames[i])
            overlap_matrix[mlx_idx, pytorch_idx] += frame_duration
    
    # Print overlap matrix
    print("\n[2] Overlap Matrix (seconds):")
    print("-" * 70)
    print(f"{'MLX Speaker':<15}", end='')
    for pytorch_speaker in pytorch_speakers:
        print(f"{pytorch_speaker:>12}", end='')
    print()
    print("-" * 70)
    
    for i, mlx_speaker in enumerate(mlx_speakers):
        print(f"{mlx_speaker:<15}", end='')
        for j in range(len(pytorch_speakers)):
            overlap = overlap_matrix[i, j]
            if overlap > 0:
                print(f"{overlap:>11.1f}s", end='')
            else:
                print(f"{'':>12}", end='')
        print()
    
    # Find best matches
    print("\n[3] Best Speaker Matches:")
    print("-" * 70)
    
    for i, mlx_speaker in enumerate(mlx_speakers):
        best_match_idx = np.argmax(overlap_matrix[i, :])
        best_match = pytorch_speakers[best_match_idx]
        overlap_time = overlap_matrix[i, best_match_idx]
        
        # Calculate what % of MLX speaker time overlaps with PyTorch speaker
        mlx_total = sum(end - start for start, end, spk in mlx_segments if spk == mlx_speaker)
        match_percentage = (overlap_time / mlx_total * 100) if mlx_total > 0 else 0
        
        print(f"{mlx_speaker:<15} → {best_match:<15} ({overlap_time:>7.1f}s, {match_percentage:>5.1f}% match)")
    
    # Analyze differences
    print("\n[4] Key Differences:")
    print("-" * 70)
    
    mlx_stats = defaultdict(lambda: {'time': 0, 'count': 0})
    pytorch_stats = defaultdict(lambda: {'time': 0, 'count': 0})
    
    for start, end, speaker in mlx_segments:
        mlx_stats[speaker]['time'] += (end - start)
        mlx_stats[speaker]['count'] += 1
    
    for start, end, speaker in pytorch_segments:
        pytorch_stats[speaker]['time'] += (end - start)
        pytorch_stats[speaker]['count'] += 1
    
    # Compare segmentation granularity
    mlx_avg_seg = sum(s['time'] for s in mlx_stats.values()) / sum(s['count'] for s in mlx_stats.values())
    pytorch_avg_seg = sum(s['time'] for s in pytorch_stats.values()) / sum(s['count'] for s in pytorch_stats.values())
    
    print(f"\nSegmentation Style:")
    print(f"  MLX average segment: {mlx_avg_seg:.2f}s ({len(mlx_segments)} total)")
    print(f"  PyTorch average segment: {pytorch_avg_seg:.2f}s ({len(pytorch_segments)} total)")
    print(f"  → MLX uses {mlx_avg_seg/pytorch_avg_seg:.1f}x longer segments (more merged)")
    
    # Check for speaker confusion patterns
    print(f"\nSpeaker Confusion Patterns:")
    
    # Where does MLX speaker_3 differ from PyTorch?
    mlx_speaker3_pytorch = []
    for i in range(num_frames):
        if mlx_frames[i] == 'speaker_3' and pytorch_frames[i] is not None:
            mlx_speaker3_pytorch.append(pytorch_frames[i])
    
    if mlx_speaker3_pytorch:
        from collections import Counter
        confusion = Counter(mlx_speaker3_pytorch)
        print(f"  When MLX predicts speaker_3:")
        for pytorch_spk, count in confusion.most_common():
            percentage = (count / len(mlx_speaker3_pytorch)) * 100
            print(f"    {percentage:5.1f}% → PyTorch {pytorch_spk}")
    
    print("\n" + "="*70)
    print("✅ Analysis complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    analyze_speaker_mapping()
