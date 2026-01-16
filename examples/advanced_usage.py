"""
Example: Speaker diarization with custom parameters
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import SpeakerDiarizationPipeline


def main():
    # Initialize pipeline with custom parameters
    print("Initializing customized speaker diarization pipeline...")
    pipeline = SpeakerDiarizationPipeline(
        device="gpu",
        segmentation_threshold=0.5,  # Threshold for voice activity detection
        clustering_threshold=0.7,     # Threshold for speaker clustering
        min_duration=0.5,             # Minimum segment duration (seconds)
        use_vbx=False,                # Use VBx clustering (vs hierarchical)
    )
    
    # Audio file path
    audio_path = "path/to/your/audio.wav"
    
    if not Path(audio_path).exists():
        print(f"Audio file not found: {audio_path}")
        print("Creating a demo with synthetic results...")
        demo_results()
        return
    
    # Run diarization with speaker constraints
    print(f"\nProcessing audio: {audio_path}")
    
    # Option 1: Let the system determine the number of speakers
    diarization = pipeline(audio_path)
    
    # Option 2: Specify exact number of speakers
    # diarization = pipeline(audio_path, num_speakers=3)
    
    # Option 3: Specify range of speakers
    # diarization = pipeline(audio_path, min_speakers=2, max_speakers=5)
    
    # Print detailed results
    print("\nDiarization Results:")
    print("=" * 70)
    
    # Group by speaker
    speakers = {}
    for turn, speaker in diarization.speaker_diarization:
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append(turn)
    
    # Print summary
    print(f"\nDetected {len(speakers)} speaker(s)\n")
    
    for speaker, turns in speakers.items():
        total_duration = sum(turn.duration for turn in turns)
        print(f"{speaker}:")
        print(f"  Total speaking time: {total_duration:.2f}s")
        print(f"  Number of turns: {len(turns)}")
        print(f"  Segments:")
        for turn in turns:
            print(f"    {turn.start:.3f}s - {turn.end:.3f}s ({turn.duration:.2f}s)")
        print()
    
    print("=" * 70)


def demo_results():
    """Show demo results with synthetic data"""
    from diarization_types import SpeakerDiarization
    
    diarization = SpeakerDiarization()
    diarization.add_segment(0.0, 5.5, "SPEAKER_00")
    diarization.add_segment(5.5, 12.3, "SPEAKER_01")
    diarization.add_segment(12.3, 18.7, "SPEAKER_00")
    diarization.add_segment(18.7, 25.1, "SPEAKER_02")
    
    print("\nDemo Diarization Results:")
    print("=" * 70)
    for turn, speaker in diarization.speaker_diarization:
        print(f"{speaker} speaks between t={turn.start:.3f}s and t={turn.end:.3f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
