# Embedding Model Comparison Status

## Summary

✅ **Successfully converted** `pyannote/wespeaker-voxceleb-resnet34-LM` to MLX
❌ **Bug found:** MLX model outputs all zeros (needs fixing)
✅ **PyTorch comparison framework** created and working

## What We Accomplished

### 1. Model Conversion ✅
- Successfully downloaded pyannote/wespeaker-voxceleb-resnet34-LM
- Converted 182 tensors from PyTorch to MLX (6.6M parameters, 25MB)
- Created proper Conv2d weight transposition (PyTorch → MLX format)
- Implemented ResNet34 architecture in MLX

### 2. Comparison Framework ✅
- Created `compare_resnet_only.py` to compare PyTorch vs MLX
- Successfully loads both models
- Identified correct input format: PyTorch expects (batch, time, freq), MLX expects (batch, time, freq, 1)
- PyTorch ResNet returns tuple: (loss?, embeddings) - use element[1]

### 3. Architecture Understanding ✅
- **Input:** Mel spectrogram (batch, time, freq=80)
- **After Layer4:** (batch, channels=256, freq=10, time=13) in PyTorch format
- **After TSTP Pooling:** (batch, 5120) = 256 channels × 10 freq × 2 (mean+std)
- **Final Output:** (batch, 256) L2-normalized embeddings

### 4. Bug Identified ❌

**Critical Issue:** MLX model outputs all zeros!

**Symptoms:**
```
Test 1: Small input (1, 50, 80, 1)
  Output: all zeros
  Norm: 0.0

Test 2: Larger input (1, 100, 80, 1)
  Output: all zeros
  Norm: 0.0
```

**What Works:**
- Conv1 → outputs reasonable values (mean=-0.00017, std=0.71)
- BatchNorm1 → outputs reasonable values (mean=0.15, std=0.35)
- ReLU → outputs reasonable values (mean=0.19, std=0.32)

**What Breaks:**
- Somewhere in the ResNet layers (layer1-4), output goes to zero
- By the time we reach final embeddings, everything is zero

## Potential Causes

### 1. Weight Loading Issue (Most Likely)
- Weights show "38 missing keys" (conv biases that don't exist - this is OK)
- But weights may not be properly applied to the layer lists
- The weight loading uses `setattr()` which might not work with list structures

```python
# Current approach in load_weights():
for attr in param_path[:-1]:
    if attr.isdigit():
        module = module[int(attr)]  # ← May not work correctly
    else:
        module = getattr(module, attr)
```

### 2. Layer Structure Issue
- Using Python lists instead of nn.Sequential
- Lists aren't proper nn.Module children, so weights might not transfer

```python
# Current structure:
self.layer1 = self._make_layer_list(...)  # Returns Python list
self.layer2 = self._make_layer_list(...)  # Returns Python list

# Should maybe be:
self.layer1 = nn.Sequential(...)  # Proper nn.Module
```

### 3. BatchNorm Evaluation Mode
- BatchNorm layers might need to be set to eval mode
- Or running_mean/running_var might not be loaded properly

## Files Created

✅ **Working Files:**
- `src/resnet_embedding.py` - MLX ResNet34 implementation (needs bug fix)
- `convert_embedding_model.py` - Weight conversion script (works!)
- `compare_resnet_only.py` - Comparison framework (works!)
- `test_embedding_model.py` - Basic tests (pass but don't catch zero bug)
- `models/embedding_mlx/weights.npz` - Converted weights (25MB)

## Next Steps to Fix

### Priority 1: Fix Zero Output Bug

**Option A:** Fix weight loading for list-based layers
```python
# Need to properly handle layer1[0], layer1[1], etc.
# Current setattr() approach doesn't work with lists
```

**Option B:** Change architecture to use nn.Sequential
```python
self.layer1 = nn.Sequential(*self._make_layer_list(...))
```

**Option C:** Manual weight assignment
```python
# Explicitly assign weights to each block
self.layer1[0].conv1.weight = weights['layer1.0.conv1.weight']
```

### Priority 2: Test with Real Mel Spectrograms
- Once zero bug is fixed, compare with real audio
- Use pyannote's mel spectrogram extraction
- Verify numerical accuracy

### Priority 3: End-to-End Pipeline
- Integrate with existing segmentation model
- Test full diarization workflow

## Key Learnings

1. **PyTorch vs MLX Input Formats:**
   - PyTorch ResNet: (batch, time, freq) - 3D
   - MLX ResNet: (batch, time, freq, 1) - 4D with channel dim

2. **TSTP Pooling:**
   - Pools over TIME dimension, not frequency
   - Output: channels × freq × 2 (mean + std)
   - Must flatten to single vector

3. **PyTorch Model Output:**
   - Returns tuple: (loss?, embeddings)
   - Use `output[1]` for embeddings
   - Embeddings have norm ~0.8 before L2 normalization

4. **Weight Conversion:**
   - Conv2d needs transposition: (out,in,h,w) → (out,h,w,in)
   - Shortcut convolutions also need transposition
   - Conv biases don't exist (bias=False in PyTorch)

## Comparison Results (Current)

```
PyTorch:
  Output shape: (2, 256)
  Output norm: [0.8157, 0.7901]
  Values: [0.0046, 0.0374, 0.0043, -0.0119, 0.0071]

MLX:
  Output shape: (2, 256)
  Output norm: [1.0, 1.0]  ← L2 normalized
  Values: ALL ZEROS ← BUG!
```

## Status: 🟡 In Progress

- ✅ Architecture implemented
- ✅ Weights converted
- ✅ Comparison framework created
- ❌ **Bug:** Zero outputs (needs immediate fix)
- ⏸️ Numerical accuracy validation (blocked by bug)
- ⏸️ Real audio testing (blocked by bug)

---

**Last Updated:** After discovering zero-output bug
**Next Action:** Debug and fix weight loading for layer lists
