# Mobile Deployment Guide for Handwriting Synthesis

This guide explains how to convert your handwriting synthesis model to mobile-friendly formats for deployment on edge devices like smartphones and tablets.

## 🎯 Overview

Your current model consists of:
- **Original Format**: TensorFlow v1 checkpoint (3 files, ~44MB total)
- **Architecture**: LSTM-based RNN with attention mechanism
- **Parameters**: ~3.7M parameters (LSTM: 400 units, Output: 20 mixtures, Attention: 10 components)

## 📱 Mobile-Friendly Solutions

### 1. Single File Format + Quantization

Convert your multi-file checkpoint to a single compressed file with reduced precision:

```python
from export_mobile_model import export_model

# Export with float16 quantization (50% size reduction)
export_model(
    output_dir='mobile_models',
    format_type='hdf5',       # Single HDF5 file
    quantize=True,            # Enable quantization
    precision='float16'       # Half precision
)
```

**Benefits:**
- ✅ Single file deployment
- ✅ ~50% size reduction (44MB → ~22MB)
- ✅ Maintains good accuracy
- ✅ Compatible with most mobile frameworks

### 2. Ultra-Compressed Format

For extreme size constraints:

```python
# Export with int8 quantization (75% size reduction)
export_model(
    format_type='npz',        # Compressed NumPy format
    quantize=True,
    precision='int8'          # 8-bit integers
)
```

**Benefits:**
- ✅ ~75% size reduction (44MB → ~11MB)
- ✅ Fastest inference
- ⚠️ Some accuracy loss

### 3. Simplified Mobile Inference

For resource-constrained devices, use pre-computed styles:

```python
from mobile_inference import SimplifiedMobileInference

# Lightweight inference with style vectors
mobile = SimplifiedMobileInference('styles')
strokes = mobile.synthesize_with_style("Hello World", style=2)
```

**Benefits:**
- ✅ Ultra-lightweight (~2-5MB total)
- ✅ No TensorFlow dependency
- ✅ Instant inference
- ⚠️ Limited to pre-computed styles

## 🚀 Quick Start

### Step 1: Export Your Model

```bash
# Export all variants
python export_mobile_model.py --export-all

# Or export specific format
python export_mobile_model.py --format hdf5 --precision float16
```

### Step 2: Test Mobile Inference

```bash
# Run the demo
python demo_mobile_export.py
```

### Step 3: Integrate into Your App

```python
# Option A: Full model (recommended for dynamic generation)
from mobile_model_utils import MobileModelLoader

loader = MobileModelLoader('mobile_models/handwriting_model_float16.hdf5')
weights = loader.get_all_weights()

# Option B: Simplified inference (recommended for style transfer)
from mobile_inference import SimplifiedMobileInference

mobile = SimplifiedMobileInference('styles')
strokes = mobile.synthesize_with_style("Your text", style=1)
```

## 📊 Size Comparison

| Format | Size | Reduction | Use Case |
|--------|------|-----------|----------|
| Original TF Checkpoint | ~44MB | 0% | Training/Development |
| HDF5 + Float16 | ~22MB | 50% | Mobile Apps |
| NPZ + Int8 | ~11MB | 75% | Edge Devices |
| Style Vectors Only | ~2-5MB | 90%+ | Simple Apps |

## 🛠️ Advanced Optimizations

### Weight Pruning

Remove unnecessary connections to reduce model size:

```python
export_model(
    format_type='hdf5',
    quantize=True,
    precision='float16',
    apply_pruning=True,
    sparsity=0.2              # Remove 20% of weights
)
```

### Format Selection Guide

- **HDF5**: Best for Python/TensorFlow mobile apps
- **NPZ**: Best for NumPy-based inference
- **JSON**: Best for web/JavaScript deployment (small models only)

## 📋 Mobile Integration Checklist

### For iOS/Android Apps:

1. **Choose Format:**
   - HDF5 for TensorFlow Lite integration
   - NPZ for custom inference engines

2. **Memory Management:**
   - Load weights on-demand
   - Use memory mapping for large models
   - Consider model caching strategies

3. **Performance:**
   - Test on target devices
   - Profile memory usage
   - Optimize for battery life

### For Web Applications:

1. **Use WebAssembly** for NumPy operations
2. **Consider model splitting** for progressive loading
3. **Implement client-side caching**

## 🔧 Troubleshooting

### Common Issues:

**Q: Model accuracy drops after quantization**
- Try float16 instead of int8
- Use post-training quantization calibration
- Consider quantization-aware training

**Q: Inference is too slow on mobile**
- Use simplified inference with pre-computed styles
- Reduce sequence length
- Apply more aggressive pruning

**Q: Memory usage is too high**
- Use streaming inference for long sequences
- Implement weight compression
- Consider model distillation

## 📱 Platform-Specific Notes

### Android
- Use TensorFlow Lite for HDF5 models
- NDK for custom NumPy inference
- Consider Android Neural Networks API

### iOS
- Use Core ML for model deployment
- Metal Performance Shaders for acceleration
- Consider creating a custom .mlmodel

### Web
- Use TensorFlow.js for HDF5 models
- WebAssembly for NPZ models
- Consider WebGL acceleration

## 🎨 Style Management

Your model includes pre-computed styles in the `styles/` directory:

```
styles/
├── style-0-strokes.npy    # Style vectors
├── style-0-chars.npy      # Character encodings
├── style-1-strokes.npy
└── ...
```

**For mobile deployment:**
1. Include only required styles
2. Compress style files
3. Consider style interpolation for variety

## 📈 Performance Benchmarks

Expected performance on common devices:

| Device Type | Model Size | Inference Time | Memory Usage |
|-------------|------------|----------------|--------------|
| High-end Phone | 22MB | 50-100ms | 50-100MB |
| Mid-range Phone | 11MB | 100-200ms | 30-60MB |
| Budget Phone | 2-5MB | 10-50ms | 10-30MB |

*Times for generating ~50 stroke points*

## 📞 Support

For mobile deployment questions:
1. Check the demo files for working examples
2. Test on your target devices
3. Consider your accuracy vs. size requirements
4. Profile memory and CPU usage

## 🔄 Future Improvements

Potential optimizations:
- **Model Distillation**: Train a smaller student model
- **Dynamic Quantization**: Runtime precision adjustment
- **Federated Learning**: On-device model updates
- **Neural Architecture Search**: Find optimal mobile architectures