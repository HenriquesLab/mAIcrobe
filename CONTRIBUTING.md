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

- Place tests in `src/napari_mAIcrobe/_tests/`
- Filenames should end with `_test.py`

**Example Test:**
```python
def test_segmentation_isodata(phase_example):
    """Test segmentation using isodata method."""
    # Test data
    phase_image = phase_example

    # Run isodata
    mask = mask_computation(phase_image, method='isodata')

    # Assert results
    assert mask is not None
    assert np.any(mask)  # Ensure that the mask is not empty

    # Run watershed
    pars = {"peak_min_distance_from_edge":10, "peak_min_distance":5, "peak_min_height" :5, "max_peaks" :100000}
    seg_man = SegmentsManager()
    seg_man.compute_segments(pars, mask)
    labels = seg_man.labels

    # Assert labels
    assert labels is not None
    assert labels.data.max() > 0  # Should detect some cells
    assert len(np.unique(labels.data)) > 1  # Multiple cell labels
```

## 🎨 Code Style

We follow standard Python conventions with project-specific guidelines.

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
def rotation_matrices(step):
    """Generate rotation matrices from 0 to <180 degrees.

    Matrices are transposed to use with 2 column point arrays (x, y).

    Parameters
    ----------
    step : int
        Angular step in degrees.

    Returns
    -------
    list[numpy.matrix]
        List of 2x2 rotation matrices (transposed).
    """

    result = []
    ang = 0

    while ang < 180:
        sa = np.sin(ang / 180.0 * np.pi)
        ca = np.cos(ang / 180.0 * np.pi)
        # note .T, for column points
        result.append(np.matrix([[ca, -sa], [sa, ca]]).T)
        ang = ang + step

    return result
```

## 🏗️ Contribution Workflow

### Standard Process

1. **Create an Issue** (for significant changes)

2. **Fork and Branch**
   ```bash
   # Create feature branch
   git checkout -b feature/descriptive-name

   # Or bug fix branch
   git checkout -b fix/issue-number-description
   ```

3. **Make Changes**
   - Write code following style guidelines
   - Add tests for new /old functionality
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
