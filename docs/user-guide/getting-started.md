# Getting Started with napari-mAIcrobe

Welcome to napari-mAIcrobe! This guide will get you up and running with bacterial cell analysis in under 10 minutes.

## 🚀 Installation

### Prerequisites

Before installing napari-mAIcrobe, ensure you have:

- **Python 3.10 or 3.11** (required for TensorFlow compatibility)
- **napari** installed (see below)

### Standard Installation

Install napari-mAIcrobe from PyPI:

```bash
pip install napari-mAIcrobe
```

### Conda Installation (Recommended)

For the most stable installation, use conda for napari dependencies:

```bash
# Install napari with conda
conda install -c conda-forge napari pyqt

# Install napari-mAIcrobe with pip
pip install napari-mAIcrobe
```

### Development Installation

For contributors or advanced users:

```bash
git clone https://github.com/HenriquesLab/napari-mAIcrobe.git
cd napari-mAIcrobe
pip install -e .[testing]
```

## ✅ Verify Installation

Test your installation by launching napari with the plugin:

```python
import napari
viewer = napari.Viewer()
```

You should see `napari-mAIcrobe` in the plugins list.

## 🔥 First Analysis

Let's perform your first bacterial cell analysis using the included sample data.

### Step 1: Launch napari

Go to your terminal and run:

```bash
napari
```

or programmatically:

```python
import napari
viewer = napari.Viewer()
```

### Step 2: Load Sample Data

napari-mAIcrobe includes _S. aureus_ test images. Load them via:

**Option A: GUI Method**
1. Go to `File > Open Sample > mAIcrobe`
2. Select:
   - "Phase contrast S. aureus"
   - "Membrane dye S.aureus"
   - "DNA dye S.aureus"

**Option B: Programmatic Method**

```python
import napari
import numpy as np

# Launch napari viewer
viewer = napari.Viewer()

# Load sample data
from napari_mAIcrobe._sample_data import phase_example, membrane_example, dna_example

# Get the sample images
phase_data = phase_example()[0]
membrane_data = membrane_example()[0]
dna_data = dna_example()[0]

# Add images to viewer
phase_layer = viewer.add_image(phase_data[0], **phase_data[1])
membrane_layer = viewer.add_image(membrane_data[0], **membrane_data[1])
dna_layer = viewer.add_image(dna_data[0], **dna_data[1])
```

### Step 3: Segment Cells

1. Go to `Plugins > mAIcrobe > Compute label`
2. **Configure segmentation parameters:**
   ```
   Base Image: Phase (select the phase contrast layer)
   Fluor 1: Membrane (select the membrane layer)
   Fluor 2: DNA (select the DNA layer)
   Model: Isodata (or CellPose cyto3)
   ```
   For the purpose of this tutorial, leave all other parameters as default.
3. Click **Run**

The segmentation will create a new "Labels" layer with individual cells outlined.

### Step 4: Analyze Cells

1. Go to `Plugins > mAIcrobe > Compute cells`
2. Configure analysis parameters:
   - **Label Image**: Select the labels layer from Step 3
   - **Membrane Image**: Select membrane layer
   - **DNA Image**: Select DNA layer
   - **Pixel size**: Enter your pixel size (e.g., 0.065 μm/pixel) (optional)
   - **Classify cell cycle**: Check this box
   - **Model**: Select "S.aureus DNA+Membrane Epi"
3. Click **Run**

This will add statistical measurements to the Labels layer properties. For the purpose of this tutorial, leave all other parameters as default.

### Step 5: View Results

After analysis completes, you can:

- **View cell statistics**: Check the layer properties panel
- **Filter cells**: Use `Plugins > napari-mAIcrobe > Filter cells`
- **Generate reports**: Enable "Generate Report" in Step 4


## 📚 Next Steps

Now that you've completed your first analysis:

1. **[Segmentation Guide](segmentation-guide.md)** - Choose the optimal segmentation method
2. **[Cell Analysis Guide](cell-analysis.md)** - Comprehensive analysis workflows
3. **[Cell Classification Guide](cell-classification.md)** - Cell classification in detail
4. **[API Reference](../api/api-reference.md)** - Programmatic usage

## 🤝 Getting Help

If you encounter issues:

1. Check this documentation and troubleshooting guides
2. Search [GitHub Issues](https://github.com/HenriquesLab/napari-mAIcrobe/issues)
4. Include error messages, napari version, and system details

Welcome to the napari-mAIcrobe community! 🔬
