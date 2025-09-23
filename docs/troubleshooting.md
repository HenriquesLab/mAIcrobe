# Troubleshooting Guide

This comprehensive guide helps resolve common issues encountered when using mAIcrobe, from installation problems to analysis errors.

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
mAIcrobe automatically disables GPU to avoid conflicts:
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

# Then install mAIcrobe
pip install napari-mAIcrobe
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
- mAIcrobe version
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
- mAIcrobe: 0.0.1

## Additional Context
Any additional information that might be relevant
```

### Community Resources

- **GitHub Issues**: [Report bugs and request features](https://github.com/HenriquesLab/mAIcrobe/issues)
- **Documentation**: [Complete user guides](../user-guide/getting-started.md)
