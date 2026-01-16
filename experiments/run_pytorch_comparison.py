#!/usr/bin/env python3
"""
Run PyTorch pyannote model for comparison with MLX implementation.
"""

import torch
import torchaudio
from pathlib import Path
import sys
import mlx.core as mx

# Try to import pyannote
try:
    from pyannote.audio import Model
except ImportError:
    print("❌ Error: pyannote.audio not installed")
    print("Install with: pip install pyannote.audio")
    sys.exit(1)

def run_pytorch_inference(audio_path: str, output_rttm: str):
    """Run PyTorch pyannote model on audio file."""
    
    print("\n" + "="*70)
    print("PyTorch Pyannote Model Inference")
    print("="*70)
    
    # Load model
    print("\n[1] Loading PyTorch model...")
    model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=True)
    model.eval()
    print("✓ Model loaded")
    
    # Load audio
    print(f"\n[2] Loading audio: {audio_path}")
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    duration = waveform.shape[1] / sr
    print(f"✓ Loaded audio: {duration:.1f}s at {sr}Hz")
    
    # Process in chunks (10 seconds at a time)
    print(f"\n[3] Running PyTorch inference...")
    chunk_size = 160000  # 10 seconds at 16kHz
    all_predictions = []
    
    num_chunks = (waveform.shape[1] + chunk_size - 1) // chunk_size
    
    with torch.no_grad():
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, waveform.shape[1])
            chunk = waveform[:, start_idx:end_idx]
            
            # Pad if needed
            if chunk.shape[1] < chunk_size:
                pad_size = chunk_size - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_size))
            
            # Run inference
            logits = model(chunk.unsqueeze(0))  # Add batch dimension
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            
            all_predictions.append(predictions[0])
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{num_chunks} chunks...")
    
    # Concatenate predictions
    all_predictions = torch.cat(all_predictions, dim=0)
    print(f"✓ Generated {all_predictions.shape[0]} predictions")
    
    # Convert to segments
    print(f"\n[4] Converting to speaker segments...")
    segments = []
    
    frame_duration = 0.017  # ~17ms per frame
    current_speaker = int(all_predictions[0])
    start_time = 0.0
    
    for i in range(1, len(all_predictions)):
        if all_predictions[i] != current_speaker:
            end_time = i * frame_duration
            if end_time - start_time >= 0.5:  # Min duration 0.5s
                segments.append((start_time, end_time, current_speaker))
            
            current_speaker = int(all_predictions[i])
            start_time = i * frame_duration
    
    # Add final segment
    end_time = len(all_predictions) * frame_duration
    if end_time - start_time >= 0.5:
        segments.append((start_time, end_time, current_speaker))
    
    print(f"✓ Found {len(segments)} segments")
    
    # Save RTTM
    print(f"\n[5] Saving RTTM to: {output_rttm}")
    audio_id = Path(audio_path).stem
    
    with open(output_rttm, 'w') as f:
        for start, end, speaker in segments:
            duration = end - start
            line = f"SPEAKER {audio_id} 1 {start:.3f} {duration:.3f} <NA> <NA> speaker_{speaker} <NA> <NA>\n"
            f.write(line)
    
    print(f"✓ Saved {len(segments)} segments")
    
    # Show speaker statistics
    from collections import defaultdict
    speaker_times = defaultdict(float)
    for start, end, speaker in segments:
        speaker_times[f"speaker_{speaker}"] += (end - start)
    
    print(f"\n[6] Speaker activity:")
    for speaker in sorted(speaker_times.keys()):
        time = speaker_times[speaker]
        percentage = (time / duration) * 100
        print(f"  {speaker}: {time:6.1f}s ({percentage:5.1f}%)")
    
    print("\n" + "="*70)
    print("✅ PyTorch inference complete!")
    print("="*70 + "\n")
    
    return segments

if __name__ == "__main__":
    audio_path = "/Users/billymolina/Github/3D-Speaker/evaluation/wavs/Use-Case_Support-20260112_094222-Besprechungsaufzeichnung.wav"
    output_rttm = "pytorch_output.rttm"
    
    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)
    
    try:
        segments = run_pytorch_inference(audio_path, output_rttm)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
