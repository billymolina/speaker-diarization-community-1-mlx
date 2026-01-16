"""
Compare existing RTTM files from MLX and PyTorch diarization
"""
from pathlib import Path

print("=" * 80)
print("RTTM FILE COMPARISON")
print("=" * 80)

# Files to compare
mlx_rttm = "test_full_diarization.rttm"
pytorch_rttm = "pytorch_output.rttm"

def read_rttm(filename):
    """Read RTTM file and return segments"""
    segments = []
    if not Path(filename).exists():
        print(f"❌ File not found: {filename}")
        return segments
    
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("SPEAKER"):
                parts = line.strip().split()
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                segments.append((start, start + duration, speaker))
    return segments

# Read both files
mlx_segs = read_rttm(mlx_rttm)
pytorch_segs = read_rttm(pytorch_rttm)

print(f"\n📄 Files:")
print(f"  MLX:     {mlx_rttm}")
print(f"  PyTorch: {pytorch_rttm}")

print(f"\n📊 Basic Statistics:")
print(f"  MLX segments:     {len(mlx_segs)}")
print(f"  PyTorch segments: {len(pytorch_segs)}")
print(f"  Difference:       {abs(len(mlx_segs) - len(pytorch_segs))} segments ({abs(len(mlx_segs) - len(pytorch_segs))/max(len(mlx_segs), len(pytorch_segs))*100:.1f}%)")

# Speaker analysis
mlx_speakers = sorted(set(s[2] for s in mlx_segs))
pytorch_speakers = sorted(set(s[2] for s in pytorch_segs))

print(f"\n👥 Speakers:")
print(f"  MLX:     {len(mlx_speakers)} speakers - {mlx_speakers[:5]}{'...' if len(mlx_speakers) > 5 else ''}")
print(f"  PyTorch: {len(pytorch_speakers)} speakers - {pytorch_speakers[:5]}{'...' if len(pytorch_speakers) > 5 else ''}")

# Total duration
def get_total_time(segments):
    return sum(end - start for start, end, _ in segments)

mlx_total = get_total_time(mlx_segs)
pytorch_total = get_total_time(pytorch_segs)

print(f"\n⏱️  Total Speaking Time:")
print(f"  MLX:     {mlx_total:.1f}s ({mlx_total/60:.1f}min)")
print(f"  PyTorch: {pytorch_total:.1f}s ({pytorch_total/60:.1f}min)")
print(f"  Difference: {abs(mlx_total - pytorch_total):.1f}s ({abs(mlx_total - pytorch_total)/max(mlx_total, pytorch_total)*100:.1f}%)")

# Time range
if mlx_segs and pytorch_segs:
    mlx_end = max(e for _,e,_ in mlx_segs)
    pytorch_end = max(e for _,e,_ in pytorch_segs)
    print(f"\n📐 Time Range:")
    print(f"  MLX:     0.0s - {mlx_end:.1f}s")
    print(f"  PyTorch: 0.0s - {pytorch_end:.1f}s)")

# Speaking time per speaker
def get_speaker_times(segments):
    times = {}
    for start, end, speaker in segments:
        times[speaker] = times.get(speaker, 0) + (end - start)
    return times

mlx_times = get_speaker_times(mlx_segs)
pytorch_times = get_speaker_times(pytorch_segs)

print(f"\n🗣️  Top 5 Speakers by Time:")
print(f"\n  MLX:")
for speaker, time in sorted(mlx_times.items(), key=lambda x: -x[1])[:5]:
    pct = time/mlx_total*100 if mlx_total > 0 else 0
    print(f"    {speaker}: {time:.1f}s ({pct:.1f}%)")

print(f"\n  PyTorch:")
for speaker, time in sorted(pytorch_times.items(), key=lambda x: -x[1])[:5]:
    pct = time/pytorch_total*100 if pytorch_total > 0 else 0
    print(f"    {speaker}: {time:.1f}s ({pct:.1f}%)")

# Segment length analysis
mlx_lengths = [e-s for s,e,_ in mlx_segs]
pytorch_lengths = [e-s for s,e,_ in pytorch_segs]

if mlx_lengths and pytorch_lengths:
    import statistics
    print(f"\n📏 Segment Length Statistics:")
    print(f"\n  MLX:")
    print(f"    Mean:   {statistics.mean(mlx_lengths):.2f}s")
    print(f"    Median: {statistics.median(mlx_lengths):.2f}s")
    print(f"    Min:    {min(mlx_lengths):.2f}s")
    print(f"    Max:    {max(mlx_lengths):.2f}s")
    
    print(f"\n  PyTorch:")
    print(f"    Mean:   {statistics.mean(pytorch_lengths):.2f}s")
    print(f"    Median: {statistics.median(pytorch_lengths):.2f}s")
    print(f"    Min:    {min(pytorch_lengths):.2f}s")
    print(f"    Max:    {max(pytorch_lengths):.2f}s")

# Sample comparison (first 10 segments)
print(f"\n📋 First 10 Segments:")
print(f"\n  MLX:")
for i, (start, end, speaker) in enumerate(mlx_segs[:10]):
    print(f"    {i+1}. [{start:7.2f}s - {end:7.2f}s] {speaker:15s} ({end-start:.2f}s)")

print(f"\n  PyTorch:")
for i, (start, end, speaker) in enumerate(pytorch_segs[:10]):
    print(f"    {i+1}. [{start:7.2f}s - {end:7.2f}s] {speaker:15s} ({end-start:.2f}s)")

# Frame-level agreement
def segments_to_frames(segments, max_time, frame_duration=0.01):
    """Convert segments to frame-level labels"""
    if not segments:
        return []
    num_frames = int(max_time / frame_duration) + 1
    frames = [''] * num_frames
    
    for start, end, speaker in segments:
        start_frame = int(start / frame_duration)
        end_frame = int(end / frame_duration)
        for i in range(start_frame, min(end_frame, num_frames)):
            frames[i] = speaker
    
    return frames

if mlx_segs and pytorch_segs:
    max_time = max(
        max(e for _,e,_ in mlx_segs),
        max(e for _,e,_ in pytorch_segs)
    )
    
    print(f"\n🎯 Frame-Level Analysis (10ms frames):")
    mlx_frames = segments_to_frames(mlx_segs, max_time)
    pytorch_frames = segments_to_frames(pytorch_segs, max_time)
    
    min_len = min(len(mlx_frames), len(pytorch_frames))
    
    # Count matches
    total_frames = 0
    matching_frames = 0
    mlx_speech = 0
    pytorch_speech = 0
    both_speech = 0
    
    for i in range(min_len):
        mlx_label = mlx_frames[i]
        pt_label = pytorch_frames[i]
        
        if mlx_label or pt_label:
            total_frames += 1
            if mlx_label: mlx_speech += 1
            if pt_label: pytorch_speech += 1
            if mlx_label and pt_label: both_speech += 1
            if mlx_label == pt_label and mlx_label:
                matching_frames += 1
    
    print(f"    Total frames analyzed: {min_len:,} ({min_len*0.01/60:.1f} minutes)")
    print(f"    Frames with speech (MLX): {mlx_speech:,} ({mlx_speech/min_len*100:.1f}%)")
    print(f"    Frames with speech (PyTorch): {pytorch_speech:,} ({pytorch_speech/min_len*100:.1f}%)")
    print(f"    Frames with speech (both): {both_speech:,} ({both_speech/min_len*100:.1f}%)")
    
    if total_frames > 0:
        agreement = matching_frames / total_frames * 100
        print(f"    Speaker agreement (on speech frames): {agreement:.1f}% ({matching_frames:,}/{total_frames:,})")
    
    if both_speech > 0:
        overlap_agreement = matching_frames / both_speech * 100
        print(f"    Agreement (when both detect speech): {overlap_agreement:.1f}%")

print("\n" + "=" * 80)
print("Analysis Complete")
print("=" * 80)
