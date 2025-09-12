# Troubleshooting Guide

This comprehensive guide helps resolve common issues encountered when using napari-mAIcrobe, from installation problems to analysis errors.

## 🚨 Installation Issues

### Python Version Problems

**Error:** `This package requires Python 3.10 or 3.11`

**Solution:**
```bash
# Check your Python version
python --version

# Install compatible Python version with conda
conda create -n mAIcrobe python=3.10
conda activate mAIcrobe
pip install napari-mAIcrobe
```

### TensorFlow Installation Issues

**Error:** `No module named 'tensorflow'` or CUDA conflicts

**Solution:**
```bash
# Install CPU-only TensorFlow (recommended)
pip install tensorflow-cpu<=2.15.0

# If you need GPU support (advanced users)
pip install tensorflow<=2.15.0
```

**GPU Conflicts:**
napari-mAIcrobe automatically disables GPU to avoid conflicts:
```python
# This is handled automatically, but you can verify:
import os
print(os.environ.get('CUDA_VISIBLE_DEVICES'))  # Should show '-1'
```

### napari Installation Problems

**Error:** `Could not find Qt platform plugin`

**Solution:**
```bash
# Install napari with conda (most reliable)
conda install -c conda-forge napari pyqt

# Alternative: specify Qt backend
pip install napari[pyqt5]
# or
pip install napari[pyside2]
```

**Error:** `napari won't start` or GUI issues

**Solution:**
```bash
# Check napari installation
napari --info

# Reinstall with specific Qt version
pip uninstall napari pyqt5 pyqt6
conda install -c conda-forge napari pyqt=5.15
```

### Dependency Conflicts

**Error:** Package version conflicts during installation

**Solution:**
```bash
# Create fresh environment
conda create -n mAIcrobe-clean python=3.10
conda activate mAIcrobe-clean

# Install napari first
conda install -c conda-forge napari

# Then install napari-mAIcrobe
pip install napari-mAIcrobe
```

## 🔬 Segmentation Problems

### Poor Cell Detection

**Symptoms:**
- Many cells not detected
- Background objects detected as cells
- Cell boundaries inaccurate

**Solutions:**

**For StarDist2D:**
```python
# Too few cells detected - lower thresholds
probability_threshold = 0.3  # from default 0.5
nms_threshold = 0.3         # from default 0.4

# Too many false positives - raise thresholds  
probability_threshold = 0.7
nms_threshold = 0.5

# Poor boundaries - enable normalization
normalize_input = True
```

**For Cellpose:**
```python
# Adjust diameter to match your cells
diameter = 15  # for small bacteria
diameter = 30  # for larger cells

# Adjust sensitivity
flow_threshold = 0.3      # more sensitive
cellprob_threshold = -2.0  # lower threshold
```

### Over-segmentation (Cells Split)

**Problem:** Single cells divided into multiple segments

**Solutions:**
1. **Increase probability threshold**:
   ```python
   probability_threshold = 0.6  # Higher = fewer detections
   ```

2. **Increase cell size limits**:
   ```python
   # For Cellpose
   diameter = 25  # Increase expected size
   ```

3. **Post-processing filters**:
   ```python
   # Filter by size after segmentation
   min_area = 50   # pixels
   max_area = 1000 # pixels
   ```

### Under-segmentation (Cells Merged)

**Problem:** Multiple cells appear as single segments

**Solutions:**
1. **Lower NMS threshold** (StarDist):
   ```python
   nms_threshold = 0.2  # Better separation
   ```

2. **Improve image contrast**:
   ```python
   # Preprocessing example
   from skimage import exposure
   enhanced = exposure.equalize_adapthist(image)
   ```

3. **Try different models**:
   - Switch from StarDist2D to Cellpose
   - Try different Cellpose models (cyto vs cyto2)

### Memory Errors During Segmentation

**Error:** `MemoryError` or system becomes unresponsive

**Solutions:**
1. **Process smaller regions**:
   ```python
   # Crop image to region of interest
   roi = image[1000:2000, 1000:2000]
   ```

2. **Reduce image resolution** (if scientifically appropriate):
   ```python
   from skimage.transform import rescale
   downsampled = rescale(image, 0.5, preserve_range=True)
   ```

3. **Close other applications** and restart napari

4. **Increase virtual memory** (swap space) on your system

## 🧬 Cell Analysis Issues

### No Cells Detected in Analysis

**Problem:** Analysis widget reports 0 cells

**Check:**
1. **Correct layer selection**:
   - Ensure Labels layer is selected (not Image layer)
   - Labels should contain integer values, not all zeros

2. **Verify segmentation**:
   ```python
   # Check if labels contain cells
   import numpy as np
   unique_labels = np.unique(labels_layer.data)
   print(f"Found {len(unique_labels)-1} cells")  # -1 for background
   ```

3. **Pixel size setting**:
   - Must be > 0
   - Typical values: 0.05-0.2 μm/pixel

### Cell Cycle Classification Fails

**Error:** Classification returns all "Unknown" or fails completely

**Solutions:**

1. **Check input channels**:
   ```python
   # Ensure both channels are loaded for dual-channel models
   membrane_layer = viewer.layers['Membrane']
   dna_layer = viewer.layers['DNA']
   print(f"Membrane shape: {membrane_layer.data.shape}")
   print(f"DNA shape: {dna_layer.data.shape}")
   ```

2. **Verify model selection**:
   - Match model to your imaging conditions
   - DNA+Membrane models need both channels
   - Single-channel models for individual stains

3. **Check cell sizes**:
   ```python
   # Cells too large for model (>50 pixels diameter)
   custom_model_maxsize = 100  # Increase limit
   ```

4. **Pixel size calibration**:
   ```python
   # Incorrect calibration affects cell size calculations
   pixel_size = 0.065  # Set accurate value in μm/pixel
   ```

### Intensity Measurements Seem Wrong

**Problem:** Unrealistic intensity values or all zeros

**Solutions:**

1. **Check image data type**:
   ```python
   print(f"Image dtype: {image.dtype}")
   print(f"Image range: {image.min()} to {image.max()}")
   
   # Convert if needed
   if image.dtype != np.float32:
       image = image.astype(np.float32)
   ```

2. **Verify channel assignment**:
   - Ensure correct image is selected for each channel
   - Check that images are properly loaded

3. **Background subtraction issues**:
   ```python
   # Adjust background region size
   baseline_margin = 50  # Increase for better background estimation
   ```

4. **Saturation detection**:
   ```python
   # Check for saturated pixels
   saturated_pixels = np.sum(image >= image.max())
   print(f"Saturated pixels: {saturated_pixels}")
   ```

### Colocalization Analysis Problems

**Error:** Colocalization metrics are NaN or unrealistic

**Solutions:**

1. **Check channel alignment**:
   ```python
   # Verify images are properly registered
   import matplotlib.pyplot as plt
   plt.figure(figsize=(10, 5))
   plt.subplot(1, 2, 1)
   plt.imshow(membrane_image, alpha=0.7, cmap='red')
   plt.imshow(dna_image, alpha=0.7, cmap='green')
   plt.title('Channel Overlay')
   ```

2. **Intensity range matching**:
   ```python
   # Normalize channels to similar ranges
   from skimage import exposure
   membrane_norm = exposure.equalize_hist(membrane_image)
   dna_norm = exposure.equalize_hist(dna_image)
   ```

3. **Background correction**:
   - Ensure proper background subtraction
   - Check for negative values after correction

## 🎛️ Widget and GUI Issues

### Widgets Not Appearing

**Problem:** Can't find napari-mAIcrobe widgets in menu

**Solutions:**

1. **Verify plugin installation**:
   ```python
   import napari_mAIcrobe
   print("Plugin installed successfully")
   ```

2. **Check napari plugin manager**:
   - Open napari
   - Go to `Plugins > Plugin Manager`
   - Ensure napari-mAIcrobe is listed and enabled

3. **Restart napari** completely:
   ```bash
   # Close napari completely, then restart
   python -m napari
   ```

4. **Clear napari cache**:
   ```bash
   # Remove napari settings (last resort)
   rm -rf ~/.napari  # Linux/Mac
   # On Windows: Delete %USERPROFILE%\.napari
   ```

### Widget Parameters Not Working

**Problem:** Changing parameters has no effect

**Solutions:**

1. **Check parameter ranges**:
   - Probability threshold: 0.0-1.0
   - Pixel size: Must be > 0
   - Integer parameters: Whole numbers only

2. **Layer selection issues**:
   - Ensure correct layers are selected in dropdowns
   - Refresh layer list if newly added

3. **Widget state reset**:
   - Close and reopen the widget
   - Restart napari if persistent

### Performance Issues

**Problem:** Analysis is very slow or system becomes unresponsive

**Solutions:**

1. **Reduce image size**:
   ```python
   # Process region of interest
   roi = image[500:1500, 500:1500]
   ```

2. **Adjust analysis parameters**:
   ```python
   # Reduce background calculation region
   baseline_margin = 20  # Smaller region
   
   # Skip optional analyses
   find_septum = False
   compute_colocalization = False
   compute_heatmap = False
   ```

3. **System optimization**:
   - Close unnecessary applications
   - Increase available RAM
   - Use SSD storage for better I/O performance

## 📊 Data Export and Reporting Issues

### Report Generation Fails

**Error:** HTML report not created or incomplete

**Solutions:**

1. **Check output directory**:
   ```python
   import os
   output_dir = "/path/to/output"
   print(f"Directory exists: {os.path.exists(output_dir)}")
   print(f"Write permissions: {os.access(output_dir, os.W_OK)}")
   ```

2. **File permissions**:
   ```bash
   # Ensure write permissions
   chmod 755 /path/to/output/directory
   ```

3. **Disk space**:
   ```bash
   # Check available space
   df -h /path/to/output
   ```

4. **Dependencies**:
   ```python
   # Verify plotting libraries
   import matplotlib.pyplot as plt
   import seaborn as sns
   print("Plotting libraries available")
   ```

### CSV Export Problems

**Problem:** CSV files empty, corrupted, or missing columns

**Solutions:**

1. **Check analysis completion**:
   - Ensure analysis finished without errors
   - Verify cells were detected and analyzed

2. **File path issues**:
   ```python
   import pandas as pd
   
   # Check if file exists and is readable
   csv_path = "/path/to/results.csv"
   if os.path.exists(csv_path):
       df = pd.read_csv(csv_path)
       print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
   ```

3. **Character encoding**:
   ```python
   # Try different encodings if file appears corrupted
   df = pd.read_csv(csv_path, encoding='utf-8')
   # or
   df = pd.read_csv(csv_path, encoding='latin1')
   ```

## 🐛 Common Error Messages

### `tensorflow.python.framework.errors_impl.InternalError`

**Cause:** TensorFlow GPU/CPU conflicts

**Solution:**
```bash
# Reinstall CPU-only version
pip uninstall tensorflow
pip install tensorflow-cpu<=2.15.0
```

### `ValueError: shapes not aligned`

**Cause:** Image dimensions don't match between channels

**Solution:**
```python
# Check and fix image dimensions
print(f"Phase shape: {phase_image.shape}")
print(f"Membrane shape: {membrane_image.shape}")
print(f"DNA shape: {dna_image.shape}")

# Resize if needed
from skimage.transform import resize
membrane_resized = resize(membrane_image, phase_image.shape)
```

### `MemoryError` or system freeze

**Cause:** Insufficient memory for large images

**Solution:**
1. Process smaller image regions
2. Reduce image resolution
3. Increase system RAM or virtual memory
4. Use 64-bit Python installation

### `ModuleNotFoundError: No module named 'napari_mAIcrobe'`

**Cause:** Plugin not properly installed

**Solution:**
```bash
# Reinstall plugin
pip uninstall napari-mAIcrobe
pip install napari-mAIcrobe

# Verify installation
python -c "import napari_mAIcrobe; print('OK')"
```

## 🔧 Advanced Troubleshooting

### Debug Mode

Enable verbose logging for detailed error information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run your analysis with detailed output
```

### System Information

Collect system information for bug reports:

```python
import napari
import numpy as np
import tensorflow as tf
import sys
import platform

print("=== System Information ===")
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {sys.version}")
print(f"napari: {napari.__version__}")
print(f"NumPy: {np.__version__}")
print(f"TensorFlow: {tf.__version__}")

# GPU information
if tf.config.list_physical_devices('GPU'):
    print("GPU available")
else:
    print("CPU only")
```

### Memory Profiling

Monitor memory usage during analysis:

```python
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")

# Run analysis here

print(f"Memory after: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

### Performance Profiling

Identify bottlenecks in analysis pipeline:

```python
import cProfile
import pstats

# Profile your analysis code
profiler = cProfile.Profile()
profiler.enable()

# Run your analysis here
# analyze_cells(...)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(10)
```

## 📞 Getting Help

### Before Asking for Help

1. **Check this troubleshooting guide**
2. **Search existing GitHub Issues**
3. **Try with sample data** to isolate the problem
4. **Collect system information** (see above)
5. **Create minimal example** that reproduces the issue

### Reporting Issues

When reporting bugs, please include:

**System Information:**
- Operating system and version
- Python version
- napari version
- napari-mAIcrobe version
- Hardware specs (RAM, CPU)

**Error Information:**
- Complete error message with traceback
- Steps to reproduce the issue
- Input data characteristics (image size, type, etc.)
- Parameter values used

**Example Issue Template:**
```markdown
## Bug Description
Brief description of the problem

## Steps to Reproduce
1. Load image with dimensions X x Y
2. Run segmentation with parameters A, B, C
3. Error occurs at step...

## Error Message
```
Complete error traceback here
```

## System Information
- OS: Windows 10 / macOS 12 / Ubuntu 20.04
- Python: 3.10.5
- napari: 0.4.18
- napari-mAIcrobe: 0.0.1

## Additional Context
Any additional information that might be relevant
```

### Community Resources

- **GitHub Discussions**: [Ask questions and share solutions](https://github.com/HenriquesLab/napari-mAIcrobe/discussions)
- **GitHub Issues**: [Report bugs and request features](https://github.com/HenriquesLab/napari-mAIcrobe/issues)
- **Documentation**: [Complete user guides](../user-guide/getting-started.md)

### Emergency Workarounds

**If napari-mAIcrobe is completely broken:**

1. **Use core functionality directly**:
   ```python
   # Direct StarDist usage
   from stardist.models import StarDist2D
   model = StarDist2D.from_pretrained('2D_versatile_bacteria')
   labels, _ = model.predict_instances(image)
   ```

2. **Alternative segmentation**:
   ```python
   # Use Cellpose directly
   from cellpose import models
   model = models.Cellpose(gpu=False, model_type='cyto')
   labels, _, _, _ = model.eval(image, diameter=None)
   ```

3. **Manual analysis** with scikit-image:
   ```python
   from skimage.measure import regionprops_table
   props = regionprops_table(labels, image, 
                            properties=['area', 'perimeter', 'eccentricity'])
   ```

---

**Still having issues?** Don't hesitate to ask for help in our [GitHub Discussions](https://github.com/HenriquesLab/napari-mAIcrobe/discussions). The community is here to help! 🤝