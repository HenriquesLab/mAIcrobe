# Basic Workflow Tutorial

This step-by-step tutorial will guide you through a complete bacterial cell analysis workflow using napari-mAIcrobe, from image loading to final report generation.

## 🎯 Tutorial Goals

By the end of this tutorial, you will:
- Load and display multi-channel bacterial images
- Segment individual cells using StarDist2D
- Perform comprehensive morphological analysis
- Apply AI-powered cell cycle classification
- Generate a professional analysis report
- Export data for further statistical analysis

## 📊 Dataset Overview

We'll use the included _S. aureus_ sample data featuring:
- **Phase contrast**: Cell morphology and boundaries
- **Membrane fluorescence**: FM 4-64 staining of cell membranes
- **DNA fluorescence**: DAPI staining of nucleoids

**Imaging parameters:**
- Pixel size: 0.065 μm/pixel
- Imaging method: Epifluorescence microscopy
- Cell population: Exponential growth phase

## 🚀 Step-by-Step Workflow

### Step 1: Launch napari and Load Data

First, let's start napari and load the sample data:

```python
import napari
import numpy as np

# Launch napari viewer
viewer = napari.Viewer()

# Load sample data
from napari_mAIcrobe._sample_data import phase_example, membrane_example, dna_example

# Get the sample images
phase_data, phase_kwargs = phase_example()
membrane_data, membrane_kwargs = membrane_example()  
dna_data, dna_kwargs = dna_example()

# Add images to viewer
phase_layer = viewer.add_image(phase_data[0], name="Phase", **phase_kwargs)
membrane_layer = viewer.add_image(membrane_data[0], name="Membrane", **membrane_kwargs)
dna_layer = viewer.add_image(dna_data[0], name="DNA", **dna_kwargs)
```

**Expected result:** You should see three image layers in napari with bacterial cells visible in each channel.

### Step 2: Examine the Images

Let's explore the data before analysis:

**Phase contrast layer:**
- Adjust contrast using the layer controls
- Identify individual bacterial cells
- Note cell density and morphology

**Membrane layer:**
- Switch to this layer to see membrane staining
- Observe cell boundaries and membrane organization
- Check for staining quality and background

**DNA layer:**
- Examine nucleoid organization
- Look for cells in different cell cycle phases
- Identify potential dividing cells

**Pro tip:** Use the layer visibility toggles to overlay channels and examine colocalization.

### Step 3: Cell Segmentation

Now we'll segment individual cells using the Compute label widget:

1. **Access the widget:**
   - Go to `Plugins > napari-mAIcrobe > Compute label`

2. **Configure segmentation parameters:**
   ```
   Image: Phase (select the phase contrast layer)
   Model: StarDist2D (recommended for rod-shaped bacteria)
   Probability threshold: 0.5
   NMS threshold: 0.4
   Normalize input: ✓ (checked)
   ```

3. **Run segmentation:**
   - Click the **Run** button
   - Wait for processing to complete (typically 10-30 seconds)

**Expected result:** A new "Labels" layer appears with individual cells outlined in different colors.

#### Evaluating Segmentation Quality

**Good segmentation indicators:**
- Each bacterial cell has a unique label/color
- Cell boundaries align well with phase contrast edges
- Minimal over-segmentation (cells split incorrectly)
- Minimal under-segmentation (cells merged incorrectly)

**If segmentation needs improvement:**
- Adjust probability threshold (lower = more cells detected)
- Modify NMS threshold (lower = better separation of touching cells)
- Try different preprocessing options

### Step 4: Comprehensive Cell Analysis

With segmented cells, we can now perform detailed analysis:

1. **Open the analysis widget:**
   - Go to `Plugins > napari-mAIcrobe > Compute cells`

2. **Configure analysis parameters:**
   ```
   Label Image: Labels (from segmentation step)
   Membrane Image: Membrane
   DNA Image: DNA
   Pixel size: 0.065 (μm/pixel)
   Inner mask thickness: 4
   Septum algorithm: Isodata
   Baseline margin: 30
   Find septum: ✓ (optional - for division detection)
   Classify cell cycle: ✓ (enable AI classification)
   Model: S.aureus DNA+Membrane Epi
   Compute Colocalization: ✓ (analyze channel relationships)  
   Generate Report: ✓ (create HTML report)
   Report path: [choose output directory]
   ```

3. **Run analysis:**
   - Click **Run**
   - Analysis may take 1-3 minutes depending on cell count

**Expected result:** The Labels layer now contains comprehensive measurements for each cell, including morphology, intensity, and cell cycle predictions.

### Step 5: Explore Analysis Results

#### View Cell Statistics

1. **Access the properties panel:**
   - Select the Labels layer
   - Open the layer properties (right panel in napari)
   - View the "properties" section

2. **Available measurements:**
   - **Morphology:** area, perimeter, circularity, aspect ratio
   - **Intensity:** mean/median values for membrane and DNA channels
   - **Cell cycle:** predicted phase and confidence scores
   - **Colocalization:** correlation coefficients between channels

#### Interactive Data Exploration

```python
# Access the analysis results programmatically
labels_layer = viewer.layers['Labels']
properties = labels_layer.properties

# View available measurements
print("Available properties:", properties.keys())

# Examine cell cycle distribution
cell_phases = properties['cell_cycle_phase']
unique_phases, counts = np.unique(cell_phases, return_counts=True)
print("Cell cycle distribution:")
for phase, count in zip(unique_phases, counts):
    print(f"  {phase}: {count} cells")

# Basic statistics
areas = properties['area_um2']
print(f"Average cell area: {np.mean(areas):.2f} ± {np.std(areas):.2f} μm²")
```

### Step 6: Quality Control and Filtering

Let's filter the results to focus on high-quality measurements:

1. **Open the filtering widget:**
   - Go to `Plugins > napari-mAIcrobe > Filter cells`

2. **Apply quality filters:**
   ```
   Area (μm²): Min: 1.0, Max: 8.0 (exclude debris and clusters)
   Circularity: Min: 0.4, Max: 1.0 (exclude damaged cells)
   Cell cycle confidence: Min: 0.6 (high-confidence classifications only)
   ```

3. **Preview filtered results:**
   - The viewer updates to show only cells meeting criteria
   - Statistics update in real-time

4. **Export filtered data:**
   - Click **Export** to save filtered results as CSV

### Step 7: Generate and Review Report

If you enabled report generation, examine the HTML output:

**Report contents:**
- **Summary statistics:** Total cells, phase distribution
- **Morphological analysis:** Size and shape distributions
- **Intensity analysis:** Fluorescence measurements and correlations
- **Cell cycle analysis:** Phase proportions and confidence metrics
- **Quality control:** Filtering effects and data quality metrics

**Viewing the report:**
```python
import webbrowser
report_path = "/path/to/your/report.html"
webbrowser.open(report_path)
```

## 📊 Data Interpretation

### Cell Cycle Analysis Results

**Expected phase distribution for exponential growth:**
- G1: ~40-50% (largest population)
- S: ~20-30% (DNA replication)
- G2: ~15-25% (pre-division)
- Division: ~5-15% (active division)

**Confidence score interpretation:**
- >0.8: High confidence, reliable classification
- 0.6-0.8: Medium confidence, generally reliable
- <0.6: Low confidence, consider manual verification

### Morphological Measurements

**Typical _S. aureus_ values:**
- Cell area: 1-4 μm²
- Length: 1-3 μm
- Width: 0.8-1.2 μm
- Circularity: 0.6-0.9 (rod-shaped)

### Intensity Analysis

**Colocalization metrics:**
- DNA-Membrane correlation: Typically 0.3-0.7
- High correlation suggests co-localized structures
- Low correlation indicates distinct compartmentalization

## 🔧 Troubleshooting Common Issues

### Segmentation Problems

**Too many false positives (background detected as cells):**
- Increase probability threshold to 0.6-0.7
- Check image quality and contrast

**Missing cells (under-detection):**
- Decrease probability threshold to 0.3-0.4
- Verify image preprocessing settings

**Merged cells (under-segmentation):**
- Decrease NMS threshold to 0.3
- Improve image contrast if possible

### Analysis Issues

**Cell cycle classification fails:**
- Verify both membrane and DNA channels are properly loaded
- Check pixel size calibration
- Ensure model selection matches imaging conditions

**Memory errors:**
- Process smaller regions of interest
- Close other applications to free memory
- Consider using a computer with more RAM

### Quality Control

**Inconsistent results across images:**
- Use identical parameters for all images
- Verify consistent imaging conditions
- Check for systematic batch effects

## 🎯 Advanced Workflow Extensions

### Batch Processing Multiple Images

```python
import glob
from pathlib import Path

# Process all images in a directory
image_paths = glob.glob("/path/to/images/*.tif")

for image_path in image_paths:
    # Load image
    image = imread(image_path)
    
    # Add to napari and process
    viewer.add_image(image)
    
    # Run segmentation and analysis with saved parameters
    # Save results with systematic naming
    
    # Clear viewer for next image
    viewer.layers.clear()
```

### Time-lapse Analysis

```python
# For time-series data
time_points = [0, 30, 60, 90, 120]  # minutes
results = {}

for t, image_path in zip(time_points, image_paths):
    # Process each time point
    # Store results with time stamp
    results[t] = analyze_timepoint(image_path)

# Analyze temporal trends
plot_cell_cycle_dynamics(results)
```

### Statistical Comparisons

```python
import pandas as pd
from scipy import stats

# Load results from different conditions
control_data = pd.read_csv("control_results.csv")
treatment_data = pd.read_csv("treatment_results.csv")

# Compare cell cycle distributions
control_phases = control_data['cell_cycle_phase'].value_counts()
treatment_phases = treatment_data['cell_cycle_phase'].value_counts()

# Statistical testing
chi2_stat, p_value = stats.chi2_contingency([control_phases, treatment_phases])
print(f"Cell cycle distribution difference: p = {p_value:.3f}")

# Compare morphological parameters
area_comparison = stats.ttest_ind(control_data['area_um2'], 
                                 treatment_data['area_um2'])
print(f"Cell area difference: p = {area_comparison.pvalue:.3f}")
```

## ✅ Tutorial Completion Checklist

- [ ] Successfully loaded multi-channel bacterial images
- [ ] Performed accurate cell segmentation
- [ ] Completed comprehensive morphological analysis
- [ ] Applied AI-powered cell cycle classification
- [ ] Generated quality control filters
- [ ] Exported data for further analysis
- [ ] Reviewed and interpreted HTML report
- [ ] Understood common troubleshooting approaches

## 📚 Next Steps

**Continue learning:**
1. **[Advanced Features Tutorial](advanced-features.md)** - Explore specialized analysis options
2. **[Segmentation Guide](../user-guide/segmentation-guide.md)** - Optimize segmentation for your data
3. **[AI Models Guide](../user-guide/ai-models.md)** - Understand cell cycle classification in detail
4. **[API Reference](../api/api-reference.md)** - Programmatic control and automation

**Apply to your research:**
- Adapt parameters for your specific bacterial strain
- Optimize imaging conditions based on this tutorial
- Develop standard protocols for your laboratory
- Consider batch processing for high-throughput studies

**Join the community:**
- Share results and ask questions in [GitHub Discussions](https://github.com/HenriquesLab/napari-mAIcrobe/discussions)
- Report issues or request features via [GitHub Issues](https://github.com/HenriquesLab/napari-mAIcrobe/issues)
- Contribute improvements to the documentation or code

Congratulations on completing your first napari-mAIcrobe analysis! 🎉