#!/usr/bin/env python3
"""Analyze RTTM diarization output."""

from collections import defaultdict

# Parse RTTM file
segments = defaultdict(list)
with open('test_full_diarization.rttm', 'r') as f:
    for line in f:
        parts = line.strip().split()
        speaker = parts[7]
        start = float(parts[3])
        duration = float(parts[4])
        end = start + duration
        segments[speaker].append((start, end))

# Print timeline summary
print('\n' + '='*70)
print('Speaker Timeline Analysis')
print('='*70)

total_duration = 4666.2

for speaker in sorted(segments.keys()):
    segs = segments[speaker]
    total_time = sum(end - start for start, end in segs)
    percentage = (total_time / total_duration) * 100
    
    print(f'\n{speaker}:')
    print(f'  Total speaking time: {total_time:.1f}s ({percentage:.1f}%)')
    print(f'  Number of turns: {len(segs)}')
    print(f'  Average turn length: {total_time/len(segs):.1f}s')
    print(f'  First appearance: {segs[0][0]:.1f}s')
    print(f'  Last appearance: {segs[-1][1]:.1f}s')

# Show timeline in 10-minute blocks
print(f'\n' + '='*70)
print('Speaker Activity Timeline (10-minute blocks)')
print('='*70)

num_blocks = 47  # ~78 minutes
block_duration = 600  # 10 minutes in seconds

for block in range(num_blocks):
    start_time = block * block_duration
    end_time = start_time + block_duration
    
    # Count speaking time per speaker in this block
    block_times = defaultdict(float)
    for speaker, segs in segments.items():
        for seg_start, seg_end in segs:
            # Calculate overlap with this block
            overlap_start = max(seg_start, start_time)
            overlap_end = min(seg_end, end_time)
            if overlap_start < overlap_end:
                block_times[speaker] += (overlap_end - overlap_start)
    
    # Find dominant speaker
    if block_times:
        dominant = max(block_times.items(), key=lambda x: x[1])
        percentage = (dominant[1] / block_duration) * 100
        
        print(f'{start_time//60:3.0f}m-{end_time//60:3.0f}m: {dominant[0]:11s} ({percentage:4.1f}% active) ' + 
              '█' * int(percentage / 2))

print('\n' + '='*70)
print('✅ Analysis complete!')
print('='*70 + '\n')
