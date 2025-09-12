# Getting Started with napari-mAIcrobe

Welcome to napari-mAIcrobe! This guide will get you up and running with AI-powered bacterial cell analysis in under 10 minutes.

## 🚀 Installation

### Prerequisites

Before installing napari-mAIcrobe, ensure you have:

- **Python 3.10 or 3.11** (required for TensorFlow compatibility)
- **napari** installed with a Qt backend
- At least **4GB RAM** (8GB+ recommended for large datasets)

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

# Check if plugin is available
print("Available plugins:", viewer.plugins._plugin_manager.list_name_plugin())
```

You should see `napari-mAIcrobe` in the plugins list.

## 🔥 First Analysis

Let's perform your first bacterial cell analysis using the included sample data.

### Step 1: Launch napari

```python
import napari
viewer = napari.Viewer()
```

### Step 2: Load Sample Data

napari-mAIcrobe includes _S. aureus_ test images. Load them via:

**Option A: GUI Method**
1. Go to `File > Open Sample > napari-mAIcrobe`
2. Select:
   - "Phase contrast S. aureus"
   - "Membrane dye S.aureus"  
   - "DNA dye S.aureus"

**Option B: Programmatic Method**

```python
from napari_mAIcrobe._sample_data import phase_example, membrane_example, dna_example

# Load sample images
phase_data = phase_example()
membrane_data = membrane_example()
dna_data = dna_example()

# Add to viewer
viewer.add_image(phase_data[0][0], name="Phase")
viewer.add_image(membrane_data[0][0], name="Membrane")
viewer.add_image(dna_data[0][0], name="DNA")
```

### Step 3: Segment Cells

1. Go to `Plugins > napari-mAIcrobe > Compute label`
2. Configure segmentation parameters:
   - **Image**: Select "Phase" layer
   - **Model**: Choose "StarDist2D" (recommended for bacteria)
   - **Probability threshold**: 0.5 (default)
   - **NMS threshold**: 0.4 (default)
3. Click **Run**

The segmentation will create a new "Labels" layer with individual cells outlined.

### Step 4: Analyze Cells

1. Go to `Plugins > napari-mAIcrobe > Compute cells`
2. Configure analysis parameters:
   - **Label Image**: Select the labels layer from Step 3
   - **Membrane Image**: Select "Membrane" layer
   - **DNA Image**: Select "DNA" layer  
   - **Pixel size**: Enter your pixel size (e.g., 0.065 μm/pixel)
   - **Classify cell cycle**: Check this box
   - **Model**: Select "S.aureus DNA+Membrane Epi"
3. Click **Run**

This will add statistical measurements to the Labels layer properties.

### Step 5: View Results

After analysis completes, you can:

- **View cell statistics**: Check the layer properties panel
- **Filter cells**: Use `Plugins > napari-mAIcrobe > Filter cells`
- **Generate reports**: Enable "Generate Report" in Step 4

## 📊 Understanding Your Results

### Cell Statistics

napari-mAIcrobe computes 50+ measurements per cell:

**Morphological Features:**
- Area, perimeter, circularity
- Aspect ratio, solidity, extent
- Major/minor axis lengths

**Intensity Features (per channel):**
- Mean, median, standard deviation
- Min/max intensities
- Integrated density

**Cell Cycle Classification:**
- Predicted phase: G1, S, G2, Division
- Confidence scores per phase
- Model used for classification

### Interactive Filtering

Use the Filter cells widget to:
- Set minimum/maximum thresholds for any measurement
- Visualize filtered cell populations
- Export filtered results

## 🔧 Common Workflows

### Single Image Analysis

1. Load your images (phase contrast + fluorescence channels)
2. Segment cells with appropriate model
3. Analyze cells with desired measurements
4. Apply filters to clean up results
5. Generate report for documentation

### Batch Processing

For multiple images:

1. Process each image individually with consistent parameters
2. Export CSV files for each analysis
3. Combine CSV files for meta-analysis
4. Use statistical software for population comparisons

## 🛠️ Troubleshooting

### Common Issues

**"No module named 'tensorflow'" Error:**
```bash
pip install tensorflow<=2.15.0
```

**Segmentation produces poor results:**
- Try different probability thresholds (0.3-0.7)
- Switch between StarDist2D and Cellpose models
- Ensure good image contrast and focus

**Memory errors with large images:**
- Process smaller regions of interest
- Reduce image resolution if appropriate
- Close unnecessary applications to free RAM

**Cell cycle classification fails:**
- Ensure both membrane and DNA channels are loaded
- Check that pixel size is set correctly
- Verify channel assignments match model requirements

## 📚 Next Steps

Now that you've completed your first analysis:

1. **[Segmentation Guide](segmentation-guide.md)** - Choose the optimal segmentation method
2. **[Cell Analysis Guide](cell-analysis.md)** - Comprehensive analysis workflows  
3. **[AI Models Guide](ai-models.md)** - Cell cycle classification in detail
4. **[API Reference](../api/api-reference.md)** - Programmatic usage

## 💡 Tips for Success

### Best Practices

- **Image Quality**: Ensure good contrast and proper focus
- **Pixel Size**: Always set accurate pixel size for meaningful measurements  
- **Sample Data**: Test with provided samples before analyzing your data
- **Parameter Consistency**: Use identical parameters for comparative studies
- **Validation**: Manually verify results on subset of cells initially

### Optimal Imaging Conditions

- **Phase Contrast**: Clear cell boundaries, minimal artifacts
- **Membrane Staining**: Even labeling, low background
- **DNA Staining**: Strong nuclear signal, minimal cytoplasmic staining
- **Resolution**: Sufficient to resolve individual bacteria (typically >10 pixels per cell diameter)

## 🤝 Getting Help

If you encounter issues:

1. Check this documentation and troubleshooting guides
2. Search [GitHub Issues](https://github.com/HenriquesLab/napari-mAIcrobe/issues)
3. Post questions in [GitHub Discussions](https://github.com/HenriquesLab/napari-mAIcrobe/discussions)
4. Include error messages, napari version, and system details

Welcome to the napari-mAIcrobe community! 🔬