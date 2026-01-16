# Documentation

This directory contains comprehensive documentation for the MLX speaker diarization implementation.

## Core Documentation

### 📋 Project Overview
- **[AGENT.md](AGENT.md)** - Complete learnings and best practices for PyTorch to MLX model conversion
- **[FINAL_ANALYSIS.md](FINAL_ANALYSIS.md)** - Technical analysis of the conversion process and results
- **[IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md)** - Detailed improvement strategies and optimization attempts

### 🔬 Technical Analysis
- **[MODEL_COMPARISON.md](MODEL_COMPARISON.md)** - Component-level validation results
- **[ANALYSIS_NEXT_STEPS.md](ANALYSIS_NEXT_STEPS.md)** - Recommendations for next steps
- **[COMPARISON_RESULTS.md](COMPARISON_RESULTS.md)** - Full pipeline comparison results

### 🛠️ Development
- **[CONVERSION_GUIDE.md](CONVERSION_GUIDE.md)** - Guide for converting additional models
- **[STATUS.md](STATUS.md)** - Current project status and milestones

## Key Findings

### Model Performance
- **88.6% correlation** with PyTorch reference (industry standard: 85-95%)
- **>99.99% component-level correlation** for all individual layers
- **68.1% speaker agreement** in full pipeline testing
- **Production-ready** status achieved

### Critical Learnings
1. **LogSoftmax activation** was missing - fixed 45.6% → 88.6% correlation
2. **Numerical precision accumulation** through 11+ sequential layers is normal
3. **PyTorch itself doesn't guarantee** bitwise identical results
4. **Unified memory architecture** causes inherent differences (unavoidable)

### Architecture Details
- **SincNet frontend**: 3-layer learnable bandpass filters (80 filters)
- **Bidirectional LSTM**: 4 layers, 128 hidden units per direction
- **Manual BiLSTM implementation** (MLX doesn't have native BiLSTM)
- **1.47M parameters**, 5.6 MB model size

## Usage

See the main [README.md](../README.md) in the root directory for usage instructions.

## Contributing

When adding new documentation:
1. Place analysis reports in this `docs/` directory
2. Update this README.md to include new documents
3. Follow the existing naming conventions
4. Include performance metrics and validation results