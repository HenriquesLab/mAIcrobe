# Contributing to napari-mAIcrobe

We welcome contributions to napari-mAIcrobe! Whether it's bug reports, feature requests, documentation improvements, or code contributions, your help makes this project better for the entire scientific community.

## 🎯 Ways to Contribute

### 🐛 **Bug Reports**
- Report issues via [GitHub Issues](https://github.com/HenriquesLab/napari-mAIcrobe/issues)
- Include detailed descriptions, error messages, and steps to reproduce
- Provide system information (OS, Python version, napari version)

### ✨ **Feature Requests**
- Suggest new analysis features or improvements
- Describe your use case and expected behavior
- Consider contributing the implementation if possible

### 📖 **Documentation**
- Improve existing documentation
- Add tutorials and examples
- Fix typos and clarify explanations

### 🧪 **Code Contributions**
- New segmentation algorithms
- Additional AI models for classification
- Performance improvements
- Test coverage expansion

## 🚀 Development Setup

### Prerequisites

- **Python 3.10 or 3.11**
- **Git** for version control
- **napari** development environment

### Quick Start

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/napari-mAIcrobe.git
cd napari-mAIcrobe

# 3. Create development environment
conda create -n mAIcrobe-dev python=3.10
conda activate mAIcrobe-dev

# 4. Install napari with conda (recommended)
conda install -c conda-forge napari pyqt

# 5. Install in development mode
pip install -e .[testing]

# 6. Install pre-commit hooks
pip install pre-commit
pre-commit install

# 7. Verify installation
python -c "import napari_mAIcrobe; print('Installation successful!')"
```

### Development Dependencies

```bash
pip install -e .[testing]
```

**Includes:**
- pytest and pytest-cov for testing
- pytest-qt for GUI testing
- black for code formatting
- isort for import sorting
- flake8 for linting
- pre-commit for automated checks

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=napari_mAIcrobe --cov-report=html

# Run specific test file
pytest src/napari_mAIcrobe/_tests/sampledata_test.py -v

# Run tests across multiple Python versions
tox
```

### Writing Tests

**Test Structure:**
```
src/napari_mAIcrobe/_tests/
├── conftest.py                # Test configuration and fixtures
├── test_widgets.py           # Widget functionality tests
├── test_segmentation.py      # Segmentation algorithm tests
├── test_analysis.py          # Cell analysis tests
├── test_classification.py    # Classification model tests
└── test_sample_data.py       # Sample data tests
```

**Example Test:**
```python
import numpy as np
import pytest
from napari_mAIcrobe._computelabel import compute_label

def test_stardist_segmentation(phase_image_fixture):
    """Test StarDist2D segmentation functionality."""
    # Create mock napari viewer
    viewer = MockViewer()
    image_layer = viewer.add_image(phase_image_fixture, name="Phase")

    # Run segmentation
    labels_layer = compute_label(
        viewer=viewer,
        image=image_layer,
        model="StarDist2D",
        probability_threshold=0.5
    )

    # Validate results
    assert labels_layer is not None
    assert labels_layer.data.max() > 0  # Should detect some cells
    assert len(np.unique(labels_layer.data)) > 1  # Multiple cell labels
```

## 🎨 Code Style

We follow standard Python conventions with project-specific guidelines.

### Formatting Standards

**Black Configuration (pyproject.toml):**
```toml
[tool.black]
line-length = 79
target-version = ['py310', 'py311']
```

**Import Organization (isort):**
```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
from magicgui import magic_factory

# napari imports
import napari
from napari.layers import Image, Labels

# Local imports
from .mAIcrobe.cells import CellManager
```

### Code Quality Checks

**Pre-commit hooks automatically run:**
- Black code formatting
- isort import sorting
- flake8 linting

**Manual checks:**
```bash
# Format code
black src/

# Sort imports
isort src/

# Lint code
flake8 src/

# Run all pre-commit checks
pre-commit run --all-files
```

### Documentation Standards

**Docstring Format (NumPy style):**
```python
def analyze_cell_morphology(image, labels, pixel_size=1.0):
    """Compute morphological measurements for segmented cells.

    Parameters
    ----------
    image : np.ndarray
        Input image for analysis
    labels : np.ndarray
        Segmented cell labels
    pixel_size : float, optional
        Spatial calibration in μm/pixel (default: 1.0)

    Returns
    -------
    dict
        Dictionary containing morphological measurements

    Raises
    ------
    ValueError
        If image and labels dimensions don't match

    Examples
    --------
    >>> measurements = analyze_cell_morphology(phase_image, cell_labels, 0.065)
    >>> print(f"Found {len(measurements)} cells")
    """
```

## 🏗️ Contribution Workflow

### Standard Process

1. **Create an Issue** (for significant changes)
   - Describe the problem or enhancement
   - Discuss approach with maintainers
   - Get approval before starting work

2. **Fork and Branch**
   ```bash
   # Create feature branch
   git checkout -b feature/descriptive-name

   # Or bug fix branch
   git checkout -b fix/issue-number-description
   ```

3. **Make Changes**
   - Write code following style guidelines
   - Add tests for new functionality
   - Update documentation as needed

4. **Test Locally**
   ```bash
   # Run tests
   pytest -v

   # Check code quality
   pre-commit run --all-files

   # Test with real data
   python scripts/test_new_feature.py
   ```

5. **Commit Changes**
   ```bash
   # Stage changes
   git add .

   # Commit with descriptive message
   git commit -m "feat: add custom model loading functionality

   - Add support for loading custom TensorFlow models
   - Include input validation and error handling
   - Add tests for model loading edge cases
   - Update documentation with usage examples"
   ```

6. **Push and Create PR**
   ```bash
   git push origin feature/descriptive-name
   ```

   Then create a Pull Request on GitHub.


## 📋 Release Process

### Version Numbering

**Semantic Versioning (semver):**
- `MAJOR.MINOR.PATCH` (e.g., 1.2.3)
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes


## 👥 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Focus on technical discussions
- Help newcomers learn
- Credit contributions appropriately
- Follow scientific integrity principles

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **Pull Requests**: Code contributions and review


---

Thank you for contributing to napari-mAIcrobe! Your efforts help advance automated microscopy analysis for the entire research community.

*Developed collaboratively by the [Henriques](https://henriqueslab.org) and [Pinho](https://www.itqb.unl.pt/research/biology/bacterial-cell-biology) Labs.* 🔬✨
