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

- **Python 3.10 or 3.11** (required for TensorFlow compatibility)
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
pytest src/napari_mAIcrobe/_tests/test_widgets.py -v

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
├── test_classification.py    # AI model tests
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

### Test Coverage Goals

- **Widget functions**: 100% coverage
- **Core analysis functions**: >95% coverage
- **Utility functions**: >90% coverage
- **Integration tests**: Key workflows covered

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
- pyupgrade syntax modernization

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

### Commit Message Guidelines

**Format:**
```
type(scope): brief description

Detailed explanation if needed
- List of changes
- Impact on users
- Breaking changes (if any)
```

**Types:**
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `test`: Test additions/improvements
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

### Pull Request Guidelines

**PR Title:** Clear, descriptive summary of changes

**PR Description Template:**
```markdown
## Summary
Brief description of changes and motivation.

## Changes Made
- [ ] Added new feature X
- [ ] Fixed bug Y
- [ ] Updated documentation Z

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Documentation
- [ ] Updated relevant documentation
- [ ] Added docstrings to new functions
- [ ] Updated CHANGELOG.md

## Breaking Changes
List any breaking changes and migration path.

## Related Issues
Fixes #123, Related to #456
```

## 🤖 AI Model Contributions

### Pre-trained Model Standards

**Model Requirements:**
- TensorFlow 2.15.0 compatible
- Input: standardized cell patches
- Output: cell cycle phase probabilities
- Documented training dataset and methodology

**Validation Requirements:**
- Cross-validation on independent test set
- Accuracy >80% on validation data
- Robustness across imaging conditions
- Clear documentation of limitations

**Submission Process:**
1. Train and validate model following standards
2. Create model documentation (dataset, architecture, performance)
3. Submit via PR with model file and documentation
4. Code review and testing by maintainers
5. Integration into pre-trained model collection

### Custom Architecture Contributions

**Guidelines for new architectures:**
- Clear improvement over existing methods
- Comprehensive benchmarking
- Memory and speed considerations
- Documentation and examples

## 📊 Dataset Contributions

### Sample Data Guidelines

**Format Requirements:**
- TIFF format for microscopy images
- Calibrated pixel sizes
- Multiple imaging conditions represented
- Balanced cell cycle phase representation

**Metadata Requirements:**
```python
dataset_metadata = {
    'organism': 'Staphylococcus aureus',
    'strain': 'strain_identifier',
    'imaging_method': 'epifluorescence',
    'pixel_size_um': 0.065,
    'channels': ['phase_contrast', 'membrane_FM464', 'dna_DAPI'],
    'growth_condition': 'exponential_phase',
    'temperature': 37,
    'dataset_size': 500,  # number of cells
    'annotation_method': 'manual_expert'
}
```

### Data Privacy and Ethics

- Ensure data sharing permissions
- Remove any identifying information
- Follow institutional data policies
- Credit original data sources

## 🐛 Bug Fix Guidelines

### Bug Report Analysis

1. **Reproduce the Issue**
   - Follow reported steps exactly
   - Test on multiple systems if possible
   - Document reproduction conditions

2. **Root Cause Analysis**
   - Identify the source of the problem
   - Consider edge cases and error propagation
   - Check for similar issues elsewhere

3. **Fix Implementation**
   - Minimal, targeted fix
   - Preserve existing functionality
   - Add regression tests

4. **Testing and Validation**
   - Verify fix resolves the issue
   - Ensure no new issues introduced
   - Test edge cases

### Common Bug Categories

**Segmentation Issues:**
- Parameter sensitivity
- Edge case handling
- Memory management

**Analysis Errors:**
- Numerical stability
- Data type compatibility
- Missing value handling

**GUI Problems:**
- Widget state management
- Layer synchronization
- User input validation

## 📈 Performance Contributions

### Optimization Areas

**Computation Speed:**
- Vectorized operations with NumPy
- Efficient algorithms for large datasets
- Parallel processing where appropriate
- Memory-efficient data structures

**Memory Usage:**
- Lazy loading for large images
- Chunked processing strategies
- Garbage collection optimization
- Memory profiling and optimization

**User Experience:**
- Progress indicators for long operations
- Responsive GUI during processing
- Informative error messages
- Intuitive parameter defaults

### Benchmarking

**Performance Tests:**
```python
import time
import pytest
from napari_mAIcrobe._computecells import compute_cells

def test_analysis_performance(large_dataset):
    """Test analysis performance on large dataset."""
    start_time = time.time()
    
    results = compute_cells(
        labels=large_dataset['labels'],
        membrane_image=large_dataset['membrane'],
        dna_image=large_dataset['dna']
    )
    
    elapsed_time = time.time() - start_time
    
    # Performance requirements
    assert elapsed_time < 60  # Should complete within 1 minute
    assert len(results) > 1000  # Should process >1000 cells
```

## 📋 Release Process

### Version Numbering

**Semantic Versioning (semver):**
- `MAJOR.MINOR.PATCH` (e.g., 1.2.3)
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Checklist

**Pre-release:**
- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version numbers updated
- [ ] Performance benchmarks run

**Release:**
- [ ] Create release tag
- [ ] Build and test distribution
- [ ] Upload to PyPI
- [ ] Update GitHub release notes
- [ ] Announce in community channels

## 👥 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Focus on technical discussions
- Help newcomers learn
- Credit contributions appropriately
- Follow scientific integrity principles

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, usage help
- **Pull Requests**: Code contributions and review

### Recognition

Contributors are recognized through:
- Contributor list in README
- Release notes acknowledgments  
- Academic paper co-authorship (for significant contributions)
- Conference presentation opportunities

## 📚 Resources for Contributors

### Learning Resources

**napari Development:**
- [napari Plugin Development Guide](https://napari.org/stable/plugins/index.html)
- [napari Architecture](https://napari.org/stable/developers/architecture.html)

**Scientific Python:**
- [NumPy Documentation](https://numpy.org/doc/)
- [scikit-image Tutorials](https://scikit-image.org/docs/stable/user_guide.html)
- [TensorFlow Guides](https://www.tensorflow.org/guide)

**Testing and Quality:**
- [pytest Documentation](https://docs.pytest.org/)
- [Python Testing Best Practices](https://realpython.com/python-testing/)

### Getting Help

**For Development Questions:**
1. Check existing documentation and issues
2. Ask in GitHub Discussions
3. Contact maintainers directly for complex issues

**For Scientific Questions:**
- Consult relevant literature on bacterial cell cycle
- Discuss methodology in GitHub Discussions
- Consider collaboration with domain experts

## ✅ Contributor Checklist

Before submitting your contribution:

- [ ] Code follows style guidelines (black, isort, flake8)
- [ ] Tests added for new functionality
- [ ] All tests pass locally
- [ ] Documentation updated (docstrings, user guides)
- [ ] CHANGELOG.md updated with changes
- [ ] No breaking changes without discussion
- [ ] Performance impact considered
- [ ] Scientific accuracy verified
- [ ] Commit messages follow conventions
- [ ] PR description is complete and clear

---

Thank you for contributing to napari-mAIcrobe! Your efforts help advance automated microscopy analysis for the entire research community. 

*Developed collaboratively by the [Henriques](https://henriqueslab.org) and [Pinho](https://www.itqb.unl.pt/research/biology/bacterial-cell-biology) Labs.* 🔬✨

**Questions?** Feel free to ask in [GitHub Discussions](https://github.com/HenriquesLab/napari-mAIcrobe/discussions).