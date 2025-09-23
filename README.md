[![License BSD-3](https://img.shields.io/pypi/l/napari-mAIcrobe.svg?color=green)](https://github.com/HenriquesLab/mAIcrobe/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-mAIcrobe.svg?color=green)](https://pypi.org/project/napari-mAIcrobe)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-mAIcrobe.svg?color=green)](https://python.org)
[![tests](https://github.com/HenriquesLab/mAIcrobe/actions/workflows/test_oncall.yml/badge.svg)](https://github.com/HenriquesLab/mAIcrobe/actions/workflows/test_oncall.yml)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-mAIcrobe)](https://napari-hub.org/plugins/napari-mAIcrobe)

# mAIcrobe

<img src="docs/logowhitebg.png" align="right" width="200" style="margin-left: 20px;"/>

**mAIcrobe: a napari plugin for microbial image analysis.**

mAIcrobe is a comprehensive napari plugin that facilitates image analysis workflows of bacterial cells. Combining state-of-the-art segmentation approaches, morphological analysis and adaptable classification models into a napari-plugin, mAIcrobe aims to deliver a user-friendly interface that helps inexperienced users perform image analysis tasks regardless of the bacterial species and microscopy modality.

## ✨ Why mAIcrobe?

### 🔬 **For Microbiologists**
- **Automated Cell Segmentation**: StarDist2D, Cellpose, and custom U-Net models
- **Deep learning classification**: 6 pre-trained CNN models for *S. aureus* cell cycle determination plus support for custom models
- **Morphological Analysis**: Comprehensive measurements using scikit-image regionprops
- **Interactive Filtering**: Real-time cell selection based on computed statistics

### 📊 **For Quantitative Research**
- **Colocalization Analysis**: Multi-channel fluorescence quantification
- **Automated Reports**: HTML reports with visualizations and statistics
- **Data Export**: CSV export for downstream statistical analysis


## 🚀 Installation

**Standard Installation:**

```bash
pip install napari-mAIcrobe
```

**Development Installation:**

```bash
git clone https://github.com/HenriquesLab/mAIcrobe.git
cd mAIcrobe
pip install -e .
```


**🎯 [Complete Tutorial →](docs/tutorials/basic-workflow.md)**

## 🏆 Key Features

### 🎨 **Cell Segmentation**
- **Thresholding**: Isodata and Local Average methods with watershed
- **StarDist2D**: custom models
- **Cellpose**: cyto3 model
- **Custom U-Net Models**: custom models

### 🧠 **Single cell Classification**
- **Pre-trained Models**: 6 specialized models for cell cycle determination in *S. aureus*:
  - DNA+Membrane (Epifluorescence & SIM)
  - DNA-only (Epifluorescence & SIM)
  - Membrane-only (Epifluorescence & SIM)
- **Custom Model Support**: Load your own TensorFlow models

### 📊 **Comprehensive Morphometry**
- **Shape Analysis**: Area, perimeter, eccentricity
- **Intensity Measurements**: Fluorescence statistics
- **Custom Measurements**: Septum detection

## 📖 Documentation

| Guide | Purpose |
|-------|---------|
| **[🚀 Getting Started](docs/user-guide/getting-started.md)** | Installation to first analysis |
| **[🔬 Segmentation Guide](docs/user-guide/segmentation-guide.md)** | Choose the right segmentation method |
| **[📊 Cell Analysis](docs/user-guide/cell-analysis.md)** | Complete analysis workflows |
| **[🧠 Cell Classification Guide](docs/user-guide/cell-classification.md)** | Cell cycle classification setup |
| **[⚙️ API Reference](docs/api/api-reference.md)** | Programmatic usage |

## 🎯 Analysis Workflow

### 📄 **Single Image Analysis**
1. **Load Images**: Phase contrast and/or fluorescence
2. **Segment Cells**: Choose segmentation algorithm and parameters
3. **Analyze Cells**: Extract morphological and intensity features and choose classification model
4. **Filter Results**: Interactive filtering of cell populations
5. **Generate Report**: Create comprehensive analysis report


## 🧪 Sample Data

The plugin includes test datasets for method validation:

- **Phase Contrast**: _S. aureus_ cells in exponential growth
- **Membrane Stain**: NileRed fluorescence imaging
- **DNA Stain**: Hoechst nuclear labeling

Access via napari: `File > Open Sample > napari-mAIcrobe`

## 🏃‍♀️ Example Analysis

**Input Data:**
- Phase contrast image
- Membrane fluorescence
- DNA fluorescence

**Analysis Pipeline:**
1. **Segmentation**: Isodata or CellPose's cyto3 identifies individual cells in the phase contrast image
2. **Morphology**: Calculate morphological and intensity measurements
3. **Classification**: Cell cycle phase determination using pre-trained CNN model
4. **Quality Control**: Interactive filtering of analysis results. Select subpopulations based on size, intensity, or classification
5. **Report Generation**: HTML output


## 📚 Available Jupyter Notebooks

Explore advanced functionality with included notebooks:

- **[Cell Cycle Model Training](notebooks/napari_mAIcrobe_cellcyclemodel.ipynb)**: Train custom classification models
- **[StarDist Segmentation](notebooks/StarDistSegmentationTraining.ipynb)**: Retrain a StarDist segmentation model

## 🤝 Community

- **🐛 [Issues](https://github.com/HenriquesLab/mAIcrobe/issues)** - Report bugs, request features
- **📚 [napari hub](https://napari-hub.org/plugins/napari-mAIcrobe)** - Plugin ecosystem

## 🏗️ Contributing

We welcome contributions! Whether it's:

- 🐛 Bug reports and fixes
- ✨ New segmentation algorithms
- 📖 Documentation improvements
- 🧪 Additional test datasets
- 🤖 New AI models for classification

**Quick contributor setup:**
```bash
git clone https://github.com/HenriquesLab/mAIcrobe.git
cd mAIcrobe
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


## 📜 License

Distributed under the terms of the [BSD-3](http://opensource.org/licenses/BSD-3-Clause) license, mAIcrobe is free and open source software.

## 🙏 Acknowledgments

mAIcrobe is developed in the [Henriques](https://henriqueslab.org) and [Pinho](https://www.itqb.unl.pt/research/biology/bacterial-cell-biology) Labs with contributions from the napari and scientific Python communities.

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
