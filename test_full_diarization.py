"""
Test full diarization pipeline: MLX vs PyTorch
Compare RTTM outputs from both models on the same audio file
"""
import sys
from pathlib import Path
import subprocess

print("=" * 80)
print("FULL DIARIZATION PIPELINE COMPARISON")
print("=" * 80)

# Test audio file
audio_file = "tests/evaluation/wavs/Use-Case_Support-20260112_094222-Besprechungsaufzeichnung.wav"

# Check if file exists
if not Path(audio_file).exists():
    print(f"❌ Audio file not found: {audio_file}")
    sys.exit(1)

print(f"\n📁 Audio file: {audio_file}")

# Output files
mlx_output = "test_output_mlx.rttm"
pytorch_output = "test_output_pytorch.rttm"

print("\n" + "=" * 80)
print("1. Running MLX Diarization")
print("=" * 80)

# Run MLX diarization
print(f"\nExecuting: .venv/bin/python3.13 diarize.py {audio_file} --output {mlx_output}")
result = subprocess.run(
    [".venv/bin/python3.13", "diarize.py", audio_file, "--output", mlx_output],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print(f"❌ MLX diarization failed!")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
else:
    print("✅ MLX diarization completed")
    print(result.stdout)

print("\n" + "=" * 80)
print("2. Running PyTorch Diarization")
print("=" * 80)

# Run PyTorch diarization using pyannote
print("\nRunning PyTorch reference implementation...")

# Create a script to run PyTorch diarization
pytorch_script = """
import torch
from pyannote.audio import Model
import torchaudio
from pathlib import Path

# Load model
print("Loading PyTorch model...")
model = Model.from_pretrained("pyannote/segmentation-3.0")
model.eval()

# Load audio
print("Loading audio...")
waveform, sr = torchaudio.load("{audio_file}")

# Get frame-level predictions
print("Running inference...")
with torch.no_grad():
    logits = model(waveform.unsqueeze(0))
    probs = torch.exp(logits)  # Convert log probs to probs
    predictions = torch.argmax(probs, dim=-1)

print(f"Predictions shape: {{predictions.shape}}")
print(f"Unique speakers detected: {{predictions.unique().tolist()}}")

# Simple segmentation: group consecutive frames with same speaker
segments = []
current_speaker = None
start_time = 0

frame_duration = 10 * 3 * 3 * 3 / 16000  # SincNet stride=10, 3 maxpools

for i, speaker in enumerate(predictions[0]):
    speaker_id = int(speaker)
    time = i * frame_duration
    
    if speaker_id != current_speaker:
        if current_speaker is not None:
            segments.append((start_time, time, f"speaker_{{current_speaker}}"))
        current_speaker = speaker_id
        start_time = time

# Add final segment
if current_speaker is not None:
    time = len(predictions[0]) * frame_duration
    segments.append((start_time, time, f"speaker_{{current_speaker}}"))

print(f"\\nGenerated {{len(segments)}} segments")
print(f"Speakers: {{sorted(set(s[2] for s in segments))}}")

# Write RTTM
print(f"Writing to {{'{pytorch_output}'}}...")
with open("{pytorch_output}", "w") as f:
    for start, end, speaker in segments:
        duration = end - start
        f.write(f"SPEAKER audio 1 {{start:.3f}} {{duration:.3f}} <NA> <NA> {{speaker}} <NA> <NA>\\n")

print("✅ Done")
""".format(audio_file=audio_file, pytorch_output=pytorch_output)

with open("_run_pytorch.py", "w") as f:
    f.write(pytorch_script)

result = subprocess.run(
    [".venv/bin/python3.13", "_run_pytorch.py"],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.returncode != 0:
    print(f"❌ PyTorch diarization failed!")
    print("STDERR:", result.stderr)
else:
    print("✅ PyTorch diarization completed")

# Clean up temp script
Path("_run_pytorch.py").unlink(missing_ok=True)

print("\n" + "=" * 80)
print("3. Comparing Results")
print("=" * 80)

# Read both RTTM files
def read_rttm(filename):
    segments = []
    if not Path(filename).exists():
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

mlx_segments = read_rttm(mlx_output)
pytorch_segments = read_rttm(pytorch_output)

print(f"\n📊 Segment Counts:")
print(f"  MLX:     {len(mlx_segments)} segments")
print(f"  PyTorch: {len(pytorch_segments)} segments")
print(f"  Difference: {abs(len(mlx_segments) - len(pytorch_segments))} segments")

# Speaker analysis
mlx_speakers = sorted(set(s[2] for s in mlx_segments))
pytorch_speakers = sorted(set(s[2] for s in pytorch_segments))

print(f"\n👥 Speakers Detected:")
print(f"  MLX:     {len(mlx_speakers)} speakers - {mlx_speakers}")
print(f"  PyTorch: {len(pytorch_speakers)} speakers - {pytorch_speakers}")

# Calculate total speaking time per speaker
def calculate_speaking_time(segments):
    speaker_times = {}
    for start, end, speaker in segments:
        duration = end - start
        speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
    return speaker_times

mlx_times = calculate_speaking_time(mlx_segments)
pytorch_times = calculate_speaking_time(pytorch_segments)

print(f"\n⏱️  Speaking Time per Speaker:")
print(f"\n  MLX:")
for speaker, time in sorted(mlx_times.items(), key=lambda x: -x[1]):
    print(f"    {speaker}: {time:.1f}s ({time/sum(mlx_times.values())*100:.1f}%)")

print(f"\n  PyTorch:")
for speaker, time in sorted(pytorch_times.items(), key=lambda x: -x[1]):
    print(f"    {speaker}: {time:.1f}s ({time/sum(pytorch_times.values())*100:.1f}%)")

# Temporal overlap analysis
print(f"\n📈 Temporal Analysis:")
print(f"  MLX coverage:     {sum(e-s for s,e,_ in mlx_segments):.1f}s")
print(f"  PyTorch coverage: {sum(e-s for s,e,_ in pytorch_segments):.1f}s")

if mlx_segments and pytorch_segments:
    mlx_end = max(e for _,e,_ in mlx_segments)
    pytorch_end = max(e for _,e,_ in pytorch_segments)
    print(f"  MLX max time:     {mlx_end:.1f}s")
    print(f"  PyTorch max time: {pytorch_end:.1f}s")

# Sample comparison (first 10 segments)
print(f"\n📋 First 10 Segments Comparison:")
print(f"\n  MLX:")
for i, (start, end, speaker) in enumerate(mlx_segments[:10]):
    print(f"    {i+1}. [{start:.2f}-{end:.2f}] {speaker} ({end-start:.2f}s)")

print(f"\n  PyTorch:")
for i, (start, end, speaker) in enumerate(pytorch_segments[:10]):
    print(f"    {i+1}. [{start:.2f}-{end:.2f}] {speaker} ({end-start:.2f}s)")

print("\n" + "=" * 80)
print("4. Frame-Level Agreement")
print("=" * 80)

# Calculate frame-level agreement
def segments_to_frames(segments, max_time, frame_duration=0.01):
    """Convert segments to frame-level labels"""
    num_frames = int(max_time / frame_duration) + 1
    frames = [''] * num_frames
    
    for start, end, speaker in segments:
        start_frame = int(start / frame_duration)
        end_frame = int(end / frame_duration)
        for i in range(start_frame, min(end_frame, num_frames)):
            frames[i] = speaker
    
    return frames

if mlx_segments and pytorch_segments:
    max_time = max(
        max(e for _,e,_ in mlx_segments),
        max(e for _,e,_ in pytorch_segments)
    )
    
    mlx_frames = segments_to_frames(mlx_segments, max_time)
    pytorch_frames = segments_to_frames(pytorch_segments, max_time)
    
    # Calculate agreement
    min_len = min(len(mlx_frames), len(pytorch_frames))
    agreements = sum(1 for i in range(min_len) if mlx_frames[i] == pytorch_frames[i] and mlx_frames[i] != '')
    non_empty = sum(1 for i in range(min_len) if mlx_frames[i] != '' or pytorch_frames[i] != '')
    
    if non_empty > 0:
        agreement_rate = agreements / non_empty * 100
        print(f"\n  Frame-level agreement: {agreement_rate:.1f}% ({agreements}/{non_empty} frames)")
    else:
        print("\n  ⚠️  No overlapping frames to compare")

print("\n" + "=" * 80)
print("✅ Comparison Complete")
print("=" * 80)
print(f"\n📄 Output files:")
print(f"  MLX:     {mlx_output}")
print(f"  PyTorch: {pytorch_output}")
