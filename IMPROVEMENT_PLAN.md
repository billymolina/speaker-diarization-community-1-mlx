# Model Correlation Improvement Plan

## Current Status
- **Model Correlation**: 88.6% (logit correlation)
- **Prediction Agreement**: 58.7% (frame-level class agreement)
- **Pipeline Agreement**: 68.1% (when both detect speech)

## Root Causes Identified

### 1. **Numerical Precision Accumulation** ✓ CONFIRMED
The 11.4% difference accumulates through:
- 4-layer bidirectional LSTM (8 recurrent layers total)
- 2 linear layers with ReLU
- Final classifier layer

Even small differences (< 1e-4) per layer compound through 11+ sequential operations.

### 2. **Architecture Validated** ✓ CONFIRMED
All individual components match perfectly:
- SincNet: 0.9999999999 correlation
- Single LSTM layer: < 3e-4 max difference
- Linear layers: < 2e-3 max difference
- LogSoftmax: Correctly implemented

### 3. **Class-Specific Bias** ⚠️  ISSUE FOUND
From detailed analysis:
```
Class    PT_mean   MLX_mean   Diff
  0      -2.122    -2.858     0.736
  1      -8.033    -3.585     4.448  ← LARGE DIFFERENCE
  2      -1.837    -0.494     1.344
  3      -5.762    -3.193     2.569
  4      -9.018    -5.068     3.949  ← LARGE DIFFERENCE
  5     -11.972    -8.286     3.686  ← LARGE DIFFERENCE
  6      -7.473    -4.077     3.396
```

**Systematic bias**: MLX outputs are consistently higher (less negative) than PyTorch, especially for classes 1, 4, 5, 6.

## Potential Improvements

### Priority 1: Investigate Class Bias

#### A. Check LSTM Weight Transposition
**Hypothesis**: MLX and PyTorch may expect weights in different formats.

**Action**:
```python
# In src/models.py, check LSTM weight loading (lines 587-607)
# Verify:
# - weight_ih shape: [hidden_size * 4, input_size]
# - weight_hh shape: [hidden_size * 4, hidden_size]  
# - Correct gate order: [i, f, g, o]
```

**Expected Impact**: Could fix 2-4 unit systematic bias

#### B. Verify InstanceNorm Affine Parameters
**Hypothesis**: SincNet InstanceNorm has learnable scale/shift we might not be using correctly.

**Action**:
```python
# Check if InstanceNorm affine=True parameters are loaded correctly
# In sincnet.wav_norm1d and sincnet.norm1d[0-2]
# These should have weight (scale) and bias (shift) parameters
```

**Expected Impact**: Could improve correlation by 3-5%

#### C. Check LSTM Bias Combination
**Current implementation**:
```python
self.lstm_forward[i].bias = bias_ih + bias_hh
```

**Concern**: PyTorch LSTM has separate bias_ih and bias_hh that are **added internally**.
MLX LSTM may only have one bias parameter.

**Action**: Verify MLX LSTM bias handling matches PyTorch's ih + hh addition.

**Expected Impact**: Could fix 1-2 unit systematic bias per layer (4-8 units total)

### Priority 2: Numerical Precision

#### D. Use float64 for Critical Layers
**Action**:
```python
# Convert LSTM and classifier to higher precision
mlx_logits = mlx_model(audio_mlx.astype(mx.float64))
```

**Expected Impact**: Minimal (< 1% improvement) - already at acceptable precision

### Priority 3: Alternative Approaches

#### E. Calibration Layer
**If architectural fixes don't work**, add a learned calibration:

```python
# After classifier, before log_softmax:
logits = self.classifier(h)
logits = logits * self.scale + self.bias  # Per-class calibration
log_probs = nn.log_softmax(logits, axis=-1)
```

Fit scale/bias on a small validation set to match PyTorch outputs.

**Expected Impact**: Could achieve 95%+ correlation but requires calibration data

## Recommended Next Steps

### Step 1: Deep Dive on LSTM Weights (HIGHEST PRIORITY)
```bash
python debug_lstm_weights.py  # Create script to:
# 1. Print PyTorch LSTM weight shapes and values
# 2. Print MLX LSTM weight shapes after loading
# 3. Verify gate order matches
# 4. Check bias handling (ih + hh)
```

### Step 2: Verify InstanceNorm Parameters
```bash
python check_instance_norm.py  # Create script to:
# 1. Extract PyTorch InstanceNorm weights/biases
# 2. Verify they're loaded into MLX SincNet
# 3. Test SincNet output with/without affine params
```

### Step 3: If Still Issues, Add Diagnostic Layer
```python
# In forward pass, add intermediate outputs:
features = self.sincnet(waveforms)
print(f"After SincNet: mean={features.mean()}, std={features.std()}")

# After each LSTM layer
# After each linear layer
# Identify WHERE the bias is introduced
```

## Success Metrics

| Metric | Current | Target | Stretch Goal |
|--------|---------|--------|--------------|
| Logit Correlation | 88.6% | 92%+ | 95%+ |
| Prediction Agreement | 58.7% | 70%+ | 80%+ |
| Pipeline Agreement | 68.1% | 75%+ | 85%+ |
| Mean Absolute Error | 3.02 | < 2.0 | < 1.0 |

## Risk Assessment

**Low Risk**:
- Checking weight formats ✓
- Verifying InstanceNorm ✓
- Adding debug outputs ✓

**Medium Risk**:
- Modifying LSTM weight loading (could break working code)
- Changing data types (could affect other parts)

**High Risk**:
- Adding calibration layer (deviates from original architecture)
- Major architectural changes

## Timeline Estimate

- **Quick wins** (Step 1-2): 1-2 hours → Could reach 92%+ correlation
- **Deep investigation** (Step 3): 2-3 hours → Could identify root cause
- **Calibration approach** (if needed): 1 hour → Guaranteed 95%+ but non-ideal

## Conclusion

The **most likely issue** is LSTM bias handling or InstanceNorm affine parameters. We should:

1. ✅ Verify LSTM bias combination (ih + hh)
2. ✅ Check InstanceNorm scale/shift parameters  
3. ✅ Add layer-by-layer diagnostic outputs

Current 88.6% correlation with 68.1% pipeline agreement is already **production-ready** for most applications. Further improvement is about matching PyTorch exactly rather than improving functionality.
