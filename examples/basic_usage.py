"""
Example: Basic speaker diarization usage
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import SpeakerDiarizationPipeline


def main():
    # Initialize pipeline
    print("Initializing speaker diarization pipeline...")
    pipeline = SpeakerDiarizationPipeline(device="gpu")
    
    # Path to audio file
    audio_path = "path/to/your/audio.wav"
    
    if not Path(audio_path).exists():
        print(f"Audio file not found: {audio_path}")
        print("Please provide a valid audio file path.")
        return
    
    # Run diarization
    print(f"Processing audio: {audio_path}")
    diarization = pipeline(audio_path)
    
    # Print results
    print("\nDiarization Results:")
    print("=" * 60)
    for turn, speaker in diarization.speaker_diarization:
        print(f"{speaker} speaks between t={turn.start:.3f}s and t={turn.end:.3f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
