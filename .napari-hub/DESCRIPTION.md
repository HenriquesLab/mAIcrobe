# napari-mAIcrobe

**AI-powered microbial cell analysis for _S. aureus_ bacteria research.**

## Overview

napari-mAIcrobe is a comprehensive napari plugin that automates bacterial cell analysis workflows, specifically optimized for _Staphylococcus aureus_. The plugin combines state-of-the-art segmentation algorithms with AI-powered cell cycle classification to transform manual microscopy analysis into streamlined, reproducible scientific workflows.

## Key Features

### 🔬 **Advanced Cell Segmentation**
- **StarDist2D**: Optimized for dense bacterial populations
- **Cellpose**: ML-based universal cell segmentation  
- **Custom U-Net Models**: Support for user-trained segmentation networks
- **Multi-channel Support**: Phase contrast, fluorescence, and brightfield imaging

### 🧠 **AI-Powered Cell Cycle Classification**
- **6 Pre-trained TensorFlow Models**: Specialized for different imaging conditions
  - DNA+Membrane (Epifluorescence & SIM)
  - DNA-only (Epifluorescence & SIM)  
  - Membrane-only (Epifluorescence & SIM)
- **Custom Model Support**: Load your own TensorFlow models
- **Automated Classification**: G1, S, G2, and division phase detection

### 📊 **Comprehensive Analysis**
- **Morphological Measurements**: 50+ features using scikit-image regionprops
- **Colocalization Analysis**: Multi-channel fluorescence quantification
- **Interactive Filtering**: Real-time cell selection based on computed statistics
- **Professional Reports**: HTML reports with embedded plots and statistics

## Plugin Widgets

The plugin provides three main widgets accessible through the napari interface:

1. **Compute label**: Cell segmentation using StarDist2D, Cellpose, or custom models
2. **Compute cells**: Comprehensive cell analysis with morphology and AI classification  
3. **Filter cells**: Interactive filtering based on computed statistics

## Sample Data

Includes test datasets for method validation:
- Phase contrast _S. aureus_ images
- Membrane fluorescence (FM 4-64)
- DNA fluorescence (DAPI)

Access via: `File > Open Sample > napari-mAIcrobe`

## Installation

```bash
pip install napari-mAIcrobe
```

**Requirements**: Python 3.10-3.11, napari with Qt backend, TensorFlow ≤2.15.0

## Research Applications

- Antimicrobial susceptibility testing
- Cell cycle dynamics studies  
- Morphological phenotyping
- Drug mechanism of action studies
- Automated bacterial cell counting and analysis

## Documentation

Complete documentation and tutorials available at the [GitHub repository](https://github.com/HenriquesLab/napari-mAIcrobe).

---

**Developed by the [Henriques](https://henriqueslab.org) and [Pinho](https://www.itqb.unl.pt/research/biology/bacterial-cell-biology) Labs** - Advancing microbiology through AI-powered image analysis.