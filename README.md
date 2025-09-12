[![License BSD-3](https://img.shields.io/pypi/l/napari-mAIcrobe.svg?color=green)](https://github.com/HenriquesLab/napari-mAIcrobe/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-mAIcrobe.svg?color=green)](https://pypi.org/project/napari-mAIcrobe)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-mAIcrobe.svg?color=green)](https://python.org)
[![tests](https://github.com/HenriquesLab/napari-mAIcrobe/actions/workflows/test_oncall.yml/badge.svg)](https://github.com/HenriquesLab/napari-mAIcrobe/actions/workflows/test_oncall.yml)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-mAIcrobe)](https://napari-hub.org/plugins/napari-mAIcrobe)

# napari-mAIcrobe

<img src="docs/assets/logo.png" align="right" width="200" style="margin-left: 20px;"/>

**AI-powered microbial cell analysis for _S. aureus_ bacteria research.**

napari-mAIcrobe is a comprehensive napari plugin that automates bacterial cell analysis workflows, specifically optimized for _Staphylococcus aureus_. Combining state-of-the-art segmentation algorithms with AI-powered cell cycle classification, it transforms manual microscopy analysis into streamlined, reproducible scientific workflows.

## ✨ Why napari-mAIcrobe?

### 🔬 **For Microbiologists**
- **Automated Cell Segmentation**: StarDist2D, Cellpose, and custom U-Net models
- **AI Cell Cycle Classification**: 6 pre-trained TensorFlow models for accurate phase determination
- **Morphological Analysis**: Comprehensive measurements using scikit-image regionprops
- **Interactive Filtering**: Real-time cell selection based on computed statistics

### 📊 **For Quantitative Research**
- **Colocalization Analysis**: Multi-channel fluorescence quantification
- **Automated Reports**: Professional HTML reports with embedded plots and statistics
- **Batch Processing**: Analyze multiple images with consistent parameters
- **Data Export**: CSV export for downstream statistical analysis

### 🎯 **For Reproducible Science**
- **Consistent Workflows**: Standardized analysis pipelines
- **Version Control Friendly**: All parameters and settings are configurable and trackable
- **Sample Data Included**: Test datasets for method validation
- **Pre-trained Models**: Ready-to-use AI models for immediate deployment

## 🚀 Installation

**Standard Installation:**

```bash
pip install napari-mAIcrobe
```

**Development Installation:**

```bash
git clone https://github.com/HenriquesLab/napari-mAIcrobe.git
cd napari-mAIcrobe
pip install -e .
```

**Requirements:**
- Python 3.10-3.11
- napari with Qt backend
- TensorFlow ≤2.15.0 (CPU-optimized)
- Dependencies: scikit-image, pandas, cellpose, stardist-napari

## 🔥 Quick Start

**Get your first analysis in under 5 minutes:**

```python
import napari
from napari_mAIcrobe import napari_get_reader

# Launch napari with the plugin
viewer = napari.Viewer()

# Load sample S. aureus data
from napari_mAIcrobe._sample_data import phase_example, membrane_example, dna_example
viewer.add_image(phase_example()[0][0], name="Phase")
viewer.add_image(membrane_example()[0][0], name="Membrane") 
viewer.add_image(dna_example()[0][0], name="DNA")

# Access mAIcrobe widgets:
# 1. Plugins > napari-mAIcrobe > Compute label (segmentation)
# 2. Plugins > napari-mAIcrobe > Compute cells (analysis)  
# 3. Plugins > napari-mAIcrobe > Filter cells (interactive filtering)
```

**🎯 [Complete Tutorial →](docs/tutorials/basic-workflow.md)**

## 🏆 Key Features

### 🎨 **Advanced Cell Segmentation**
- **StarDist2D**: Optimized for dense bacterial populations
- **Cellpose**: ML-based universal cell segmentation  
- **Custom U-Net Models**: Train your own segmentation networks
- **Multi-channel Support**: Phase contrast, fluorescence, and brightfield

### 🧠 **AI-Powered Cell Cycle Analysis**
- **Pre-trained Models**: 6 specialized models for different imaging conditions:
  - DNA+Membrane (Epifluorescence & SIM)
  - DNA-only (Epifluorescence & SIM)  
  - Membrane-only (Epifluorescence & SIM)
- **Custom Model Support**: Load your own TensorFlow models
- **Automated Classification**: G1, S, G2, and division phase detection

### 📊 **Comprehensive Morphometry**
- **Shape Analysis**: Area, perimeter, circularity, aspect ratio
- **Intensity Measurements**: Mean, median, standard deviation per channel
- **Spatial Analysis**: Centroid tracking, neighbor analysis
- **Custom Measurements**: Inner mask analysis, septum detection

### 🔧 **Professional Reporting**
- **Interactive HTML Reports**: Publication-ready visualizations
- **Statistical Summaries**: Population-level analysis with plots
- **Data Export**: CSV files for external analysis
- **Figure Generation**: High-quality plots for presentations

## 📖 Documentation

| Guide | Purpose |
|-------|---------|
| **[🚀 Getting Started](docs/user-guide/getting-started.md)** | Installation to first analysis |
| **[🔬 Segmentation Guide](docs/user-guide/segmentation-guide.md)** | Choose the right segmentation method |
| **[📊 Cell Analysis](docs/user-guide/cell-analysis.md)** | Complete analysis workflows |
| **[🧠 AI Models Guide](docs/user-guide/ai-models.md)** | Cell cycle classification setup |
| **[⚙️ API Reference](docs/api/api-reference.md)** | Programmatic usage |

## 🎯 Analysis Workflow

### 📄 **Single Image Analysis**
1. **Load Images**: Phase contrast, membrane, and/or DNA channels
2. **Segment Cells**: Choose segmentation algorithm and parameters
3. **Analyze Cells**: Extract morphological and intensity features
4. **Classify Cell Cycle**: Apply AI models for phase determination
5. **Generate Report**: Create comprehensive analysis report

### 📊 **Batch Processing**
- Process multiple images with consistent parameters
- Compare conditions across experiments
- Export aggregated statistics for meta-analysis

### 🎓 **Research Applications**
- Antimicrobial susceptibility testing
- Cell cycle dynamics studies
- Morphological phenotyping
- Drug mechanism of action studies

## 🔧 Advanced Features

### 🎨 **Custom Model Integration**
```python
# Load custom TensorFlow model
from napari_mAIcrobe.mAIcrobe.cells import CellCycleClassifier

classifier = CellCycleClassifier()
classifier.load_custom_model("/path/to/your/model.keras")
```

### 📊 **Colocalization Analysis**
- Pearson correlation coefficients
- Manders overlap coefficients  
- Channel-specific intensity analysis
- Spatial correlation mapping

### ⚡ **Optimization Features**
- GPU acceleration disabled by default (CPU-only for stability)
- Memory-efficient processing for large datasets
- Configurable batch sizes and processing parameters

## 🧪 Sample Data

The plugin includes test datasets for method validation:

- **Phase Contrast**: _S. aureus_ cells in exponential growth
- **Membrane Stain**: FM 4-64 fluorescence imaging
- **DNA Stain**: DAPI nuclear labeling

Access via napari: `File > Open Sample > napari-mAIcrobe`

## 🏃‍♀️ Example Analysis

**Input Data:**
- Phase contrast image (cell morphology)
- Membrane fluorescence (cell boundaries)  
- DNA fluorescence (nucleoid organization)

**Analysis Pipeline:**
1. **Segmentation**: StarDist2D identifies individual cells
2. **Feature Extraction**: 50+ morphological and intensity measurements
3. **AI Classification**: Cell cycle phase determination
4. **Quality Control**: Interactive filtering of analysis results
5. **Report Generation**: Professional HTML output with statistics

**Output:**
- Cell count and population statistics
- Cell cycle phase distribution
- Morphological parameter distributions  
- Colocalization analysis (if multi-channel)
- Exportable data tables (CSV format)

## 📚 Available Jupyter Notebooks

Explore advanced functionality with included notebooks:

- **[Cell Cycle Model Training](notebooks/napari_mAIcrobe_cellcyclemodel.ipynb)**: Train custom classification models
- **[StarDist Segmentation](notebooks/StarDistSegmentationTraining.ipynb)**: Optimize segmentation parameters

## 🤝 Community

- **💬 [GitHub Discussions](https://github.com/HenriquesLab/napari-mAIcrobe/discussions)** - Ask questions, share results
- **🐛 [Issues](https://github.com/HenriquesLab/napari-mAIcrobe/issues)** - Report bugs, request features
- **📚 [napari hub](https://napari-hub.org/plugins/napari-mAIcrobe)** - Plugin ecosystem
- **🧪 [Sample Datasets](docs/user-guide/getting-started.md#sample-data)** - Test data for validation

## 🏗️ Contributing

We welcome contributions! Whether it's:

- 🐛 Bug reports and fixes
- ✨ New segmentation algorithms
- 📖 Documentation improvements  
- 🧪 Additional test datasets
- 🤖 New AI models for classification

**Quick contributor setup:**
```bash
git clone https://github.com/HenriquesLab/napari-mAIcrobe.git
cd napari-mAIcrobe
pip install -e .[testing]
pre-commit install
```

**Testing:**
```bash
# Run tests
pytest -v

# Run tests with coverage  
pytest --cov=napari_mAIcrobe

# Run tests across Python versions
tox
```

**[📋 Full Contributing Guide →](CONTRIBUTING.md)**


## 🛠️ Troubleshooting

**Common Issues:**

<details>
<summary><strong>🐍 TensorFlow/GPU Issues</strong></summary>

napari-mAIcrobe disables GPU acceleration by default to avoid CUDA conflicts:

```python
# This is handled automatically, but you can verify:
import os
print(os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'))  # Should be '-1'
```

If you encounter TensorFlow errors, ensure you have CPU-only installation:
```bash
pip install tensorflow-cpu<=2.15.0
```

</details>

<details>
<summary><strong>📦 Installation Problems</strong></summary>

**napari installation issues:**
```bash
# Use conda for napari dependencies
conda install -c conda-forge napari
pip install napari-mAIcrobe
```

**Qt backend issues:**
```bash
# Install specific Qt backend
pip install pyqt5  # or pyqt6
```

</details>

<details>
<summary><strong>🖼️ Segmentation Quality</strong></summary>

**Poor segmentation results:**
1. Try different algorithms (StarDist2D vs Cellpose)
2. Adjust preprocessing parameters  
3. Check image contrast and resolution
4. Validate with sample data first

</details>

**[📋 Complete Troubleshooting Guide →](docs/troubleshooting.md)**

## 📜 License

Distributed under the terms of the [BSD-3](http://opensource.org/licenses/BSD-3-Clause) license, napari-mAIcrobe is free and open source software.

## 🙏 Acknowledgments

napari-mAIcrobe is developed in the [Henriques](https://henriqueslab.org) and [Pinho](https://www.itqb.unl.pt/research/biology/bacterial-cell-biology) Labs with contributions from the napari and scientific Python communities.

**Built with:**
- [napari](https://napari.org/) - Multi-dimensional image viewer
- [TensorFlow](https://tensorflow.org/) - Machine learning framework  
- [StarDist](https://github.com/stardist/stardist) - Object detection with star-convex shapes
- [Cellpose](https://github.com/MouseLand/cellpose) - Generalist cell segmentation
- [scikit-image](https://scikit-image.org/) - Image processing library

---

<div align="center">

**🔬 From the [Henriques](https://henriqueslab.org) and [Pinho](https://www.itqb.unl.pt/research/biology/bacterial-cell-biology) Labs**

*"Advancing microbiology through AI-powered image analysis."*

**[🚀 Get Started →](docs/user-guide/getting-started.md)** | **[📚 Learn More →](docs/user-guide/segmentation-guide.md)** | **[⚙️ API Docs →](docs/api/api-reference.md)**

</div>