# API Reference

This comprehensive reference covers programmatic usage of napari-mAIcrobe, enabling automation, batch processing, and integration into custom analysis pipelines.

## 🎯 Overview

napari-mAIcrobe provides both GUI widgets and programmatic APIs:

- **Widget APIs**: Direct access to plugin functionality
- **Core Libraries**: Low-level analysis components
- **Utility Functions**: Helper functions for common tasks
- **Data Structures**: Standard formats for analysis results

## 📦 Module Structure

```
napari_mAIcrobe/
├── _computelabel.py      # Segmentation widget and functions
├── _computecells.py      # Cell analysis widget and functions  
├── _filtercells.py       # Filtering widget and functions
├── _sample_data.py       # Sample data providers
├── napari.yaml           # Plugin manifest
└── mAIcrobe/            # Core analysis library
    ├── cells.py         # Cell analysis and classification
    ├── segments.py      # Segmentation management
    ├── coloc.py         # Colocalization analysis
    └── reports.py       # Report generation
```

## 🔧 Widget APIs

### Segmentation API

```python
from napari_mAIcrobe._computelabel import compute_label

def compute_label(
    viewer: "napari.Viewer",
    image: "napari.layers.Image",
    model: str = "StarDist2D",
    probability_threshold: float = 0.5,
    nms_threshold: float = 0.4,
    normalize_input: bool = True,
    custom_model_path: str = ""
) -> "napari.layers.Labels"
```

**Parameters:**
- `viewer`: napari viewer instance
- `image`: Input image layer for segmentation
- `model`: Segmentation model ("StarDist2D", "Cellpose", "Custom")
- `probability_threshold`: Detection sensitivity (0.0-1.0)
- `nms_threshold`: Non-maximum suppression threshold
- `normalize_input`: Apply input normalization
- `custom_model_path`: Path to custom model file

**Returns:** Labels layer with segmented cells

**Example:**
```python
import napari
from napari_mAIcrobe._computelabel import compute_label

viewer = napari.Viewer()
image_layer = viewer.add_image(phase_image, name="Phase")

# Perform segmentation
labels_layer = compute_label(
    viewer=viewer,
    image=image_layer,
    model="StarDist2D",
    probability_threshold=0.5,
    nms_threshold=0.4
)
```

### Cell Analysis API

```python
from napari_mAIcrobe._computecells import compute_cells

def compute_cells(
    Viewer: "napari.Viewer",
    Label_Image: "napari.layers.Labels",
    Membrane_Image: "napari.layers.Image" = None,
    DNA_Image: "napari.layers.Image" = None,
    Pixel_size: float = 1.0,
    Inner_mask_thickness: int = 4,
    Septum_algorithm: str = "Isodata",
    Baseline_margin: int = 30,
    Find_septum: bool = False,
    Find_open_septum: bool = False,
    Classify_cell_cycle: bool = False,
    Model: str = "S.aureus DNA+Membrane Epi",
    Custom_model_path: str = "",
    Custom_model_input: str = "Membrane",
    Custom_model_MaxSize: int = 50,
    Compute_Colocalization: bool = False,
    Generate_Report: bool = False,
    Report_path: str = "",
    Compute_Heatmap: bool = False
) -> None
```

**Parameters:**
- `Viewer`: napari viewer instance
- `Label_Image`: Segmented cells layer
- `Membrane_Image`: Membrane fluorescence channel (optional)
- `DNA_Image`: DNA fluorescence channel (optional)
- `Pixel_size`: Spatial calibration (μm/pixel)
- Analysis parameters: Various morphological and AI options

**Example:**
```python
from napari_mAIcrobe._computecells import compute_cells

# Perform comprehensive cell analysis
compute_cells(
    Viewer=viewer,
    Label_Image=labels_layer,
    Membrane_Image=membrane_layer,
    DNA_Image=dna_layer,
    Pixel_size=0.065,
    Classify_cell_cycle=True,
    Model="S.aureus DNA+Membrane Epi",
    Generate_Report=True,
    Report_path="./analysis_results/"
)
```

### Filtering API

```python
from napari_mAIcrobe._filtercells import filter_cells

def filter_cells(
    viewer: "napari.Viewer",
    labels_layer: "napari.layers.Labels"
) -> None
```

**Opens interactive filtering widget for quality control.**

## 🧬 Core Library APIs

### CellManager Class

The main orchestrator for cell analysis workflows:

```python
from napari_mAIcrobe.mAIcrobe.cells import CellManager

class CellManager:
    def __init__(self, 
                 label_image: np.ndarray,
                 membrane_image: np.ndarray = None,
                 dna_image: np.ndarray = None,
                 pixel_size: float = 1.0):
        """Initialize cell analysis manager."""
        
    def compute_morphology(self) -> None:
        """Compute morphological measurements for all cells."""
        
    def compute_intensity(self) -> None:
        """Compute intensity measurements for all channels."""
        
    def classify_cell_cycle(self, 
                           model_name: str = "S.aureus DNA+Membrane Epi",
                           custom_model_path: str = None) -> None:
        """Apply AI-powered cell cycle classification."""
        
    def compute_colocalization(self) -> None:
        """Analyze colocalization between channels."""
        
    def export_results(self, filepath: str) -> None:
        """Export analysis results to CSV file."""
        
    def generate_report(self, output_path: str) -> None:
        """Generate comprehensive HTML report."""
```

**Example Usage:**
```python
from napari_mAIcrobe.mAIcrobe.cells import CellManager
import numpy as np

# Initialize cell manager
manager = CellManager(
    label_image=segmented_labels,
    membrane_image=membrane_data,
    dna_image=dna_data,
    pixel_size=0.065
)

# Run complete analysis
manager.compute_morphology()
manager.compute_intensity()
manager.classify_cell_cycle("S.aureus DNA+Membrane Epi")
manager.compute_colocalization()

# Export results
manager.export_results("cell_analysis_results.csv")
manager.generate_report("./reports/")
```

### Cell Class

Individual cell object with measurements and properties:

```python
from napari_mAIcrobe.mAIcrobe.cells import Cell

class Cell:
    def __init__(self, 
                 label: int,
                 mask: np.ndarray,
                 properties: dict = None):
        """Initialize individual cell object."""
        
    @property
    def morphology(self) -> dict:
        """Morphological measurements."""
        
    @property
    def intensity(self) -> dict:
        """Intensity measurements per channel."""
        
    @property
    def cell_cycle(self) -> dict:
        """Cell cycle classification results."""
        
    def compute_features(self, 
                        membrane_image: np.ndarray = None,
                        dna_image: np.ndarray = None) -> None:
        """Compute all cell features."""
```

### CellCycleClassifier Class

AI-powered cell cycle classification:

```python
from napari_mAIcrobe.mAIcrobe.cells import CellCycleClassifier

class CellCycleClassifier:
    def __init__(self):
        """Initialize classifier with pre-trained models."""
        
    def load_model(self, model_name: str) -> None:
        """Load pre-trained model by name."""
        
    def load_custom_model(self, model_path: str) -> None:
        """Load custom TensorFlow model."""
        
    def predict(self, 
               cell_images: np.ndarray,
               input_type: str = "Membrane+DNA") -> dict:
        """Classify cell cycle phases."""
        
    def get_available_models(self) -> list:
        """List available pre-trained models."""
```

**Example:**
```python
from napari_mAIcrobe.mAIcrobe.cells import CellCycleClassifier

# Initialize classifier
classifier = CellCycleClassifier()

# Load specific model
classifier.load_model("S.aureus DNA+Membrane Epi")

# Classify cell cycle phases
results = classifier.predict(cell_patches, input_type="Membrane+DNA")

# Results contain phase predictions and confidence scores
phases = results['predicted_phases']
confidences = results['confidence_scores']
```

### SegmentsManager Class

Advanced segmentation management:

```python
from napari_mAIcrobe.mAIcrobe.segments import SegmentsManager

class SegmentsManager:
    def __init__(self, image: np.ndarray):
        """Initialize segmentation manager."""
        
    def stardist_segmentation(self,
                            probability_threshold: float = 0.5,
                            nms_threshold: float = 0.4) -> np.ndarray:
        """Perform StarDist2D segmentation."""
        
    def cellpose_segmentation(self,
                            model_type: str = "cyto",
                            diameter: float = None) -> np.ndarray:
        """Perform Cellpose segmentation."""
        
    def custom_segmentation(self, model_path: str) -> np.ndarray:
        """Apply custom segmentation model."""
        
    def post_process(self, 
                    labels: np.ndarray,
                    min_size: int = 50,
                    max_size: int = 10000) -> np.ndarray:
        """Post-process segmentation results."""
```

## 📊 Data Structures

### Analysis Results Format

Cell analysis results are stored in standardized dictionaries:

```python
# Cell measurement structure
cell_measurements = {
    'label': int,                    # Cell ID
    'area_um2': float,              # Area in μm²
    'perimeter_um': float,          # Perimeter in μm
    'circularity': float,           # Shape circularity (0-1)
    'aspect_ratio': float,          # Major/minor axis ratio
    'solidity': float,              # Area/convex area
    'mean_membrane': float,         # Mean membrane intensity
    'mean_dna': float,              # Mean DNA intensity
    'cell_cycle_phase': str,        # Predicted phase
    'cycle_confidence': float,      # Classification confidence
    'colocalization_pearson': float # Channel correlation
}
```

### Configuration Parameters

Standard parameter dictionaries for reproducible analysis:

```python
# Segmentation parameters
segmentation_params = {
    'model': 'StarDist2D',
    'probability_threshold': 0.5,
    'nms_threshold': 0.4,
    'normalize_input': True
}

# Analysis parameters
analysis_params = {
    'pixel_size': 0.065,
    'inner_mask_thickness': 4,
    'baseline_margin': 30,
    'find_septum': False,
    'classify_cell_cycle': True,
    'model': 'S.aureus DNA+Membrane Epi',
    'compute_colocalization': True
}
```

## 🔄 Batch Processing APIs

### Batch Analysis Functions

```python
def batch_analyze_images(
    image_paths: list,
    output_directory: str,
    segmentation_params: dict,
    analysis_params: dict
) -> None:
    """Analyze multiple images with consistent parameters."""
    
    for image_path in image_paths:
        # Load image
        image_data = load_image(image_path)
        
        # Segment cells
        labels = segment_cells(image_data, **segmentation_params)
        
        # Analyze cells
        results = analyze_cells(labels, image_data, **analysis_params)
        
        # Save results
        output_path = Path(output_directory) / f"{Path(image_path).stem}_results.csv"
        save_results(results, output_path)

def merge_batch_results(
    result_files: list,
    output_file: str,
    add_metadata: dict = None
) -> pd.DataFrame:
    """Combine multiple analysis result files."""
    
    combined_data = []
    for file_path in result_files:
        df = pd.read_csv(file_path)
        if add_metadata:
            for key, value in add_metadata.items():
                df[key] = value
        combined_data.append(df)
    
    merged_df = pd.concat(combined_data, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    return merged_df
```

### Example Batch Processing Script

```python
import glob
from pathlib import Path
from napari_mAIcrobe.batch import batch_analyze_images, merge_batch_results

# Define analysis parameters
seg_params = {
    'model': 'StarDist2D',
    'probability_threshold': 0.5,
    'nms_threshold': 0.4
}

analysis_params = {
    'pixel_size': 0.065,
    'classify_cell_cycle': True,
    'model': 'S.aureus DNA+Membrane Epi',
    'generate_report': True
}

# Process all images in directory
image_paths = glob.glob("/data/experiment_1/*.tif")
batch_analyze_images(
    image_paths=image_paths,
    output_directory="/results/experiment_1/",
    segmentation_params=seg_params,
    analysis_params=analysis_params
)

# Combine results from all images
result_files = glob.glob("/results/experiment_1/*_results.csv")
combined_results = merge_batch_results(
    result_files=result_files,
    output_file="/results/experiment_1_combined.csv",
    add_metadata={'experiment': 'exp1', 'condition': 'control'}
)
```

## 📚 Utility Functions

### Sample Data Access

```python
from napari_mAIcrobe._sample_data import phase_example, membrane_example, dna_example

# Load sample data programmatically
phase_data, phase_kwargs = phase_example()
membrane_data, membrane_kwargs = membrane_example()
dna_data, dna_kwargs = dna_example()

# Extract image arrays
phase_image = phase_data[0]
membrane_image = membrane_data[0] 
dna_image = dna_data[0]
```

### Image Processing Utilities

```python
from napari_mAIcrobe.utils import preprocessing

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to 0-1 range."""
    
def enhance_contrast(image: np.ndarray, percentiles: tuple = (2, 98)) -> np.ndarray:
    """Enhance image contrast using percentile normalization."""
    
def remove_background(image: np.ndarray, radius: int = 50) -> np.ndarray:
    """Remove background using rolling ball subtraction."""
    
def resize_image(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resize image while preserving aspect ratio."""
```

### Data Export Utilities

```python
from napari_mAIcrobe.utils import io

def export_to_csv(data: dict, filepath: str) -> None:
    """Export analysis results to CSV format."""
    
def export_to_excel(data: dict, filepath: str, sheet_name: str = "Results") -> None:
    """Export results to Excel format."""
    
def load_results(filepath: str) -> dict:
    """Load previously saved analysis results."""
    
def export_figures(data: dict, output_dir: str) -> None:
    """Generate and export analysis figures."""
```

## 🔧 Advanced Usage Examples

### Custom Analysis Pipeline

```python
import numpy as np
import pandas as pd
from napari_mAIcrobe.mAIcrobe import CellManager, CellCycleClassifier
from napari_mAIcrobe.utils import preprocessing

def custom_analysis_pipeline(
    phase_image: np.ndarray,
    membrane_image: np.ndarray,
    dna_image: np.ndarray,
    pixel_size: float = 0.065
) -> pd.DataFrame:
    """Custom analysis pipeline with preprocessing."""
    
    # Preprocess images
    phase_enhanced = preprocessing.enhance_contrast(phase_image)
    membrane_normalized = preprocessing.normalize_image(membrane_image)
    dna_normalized = preprocessing.normalize_image(dna_image)
    
    # Segment cells
    from napari_mAIcrobe.mAIcrobe.segments import SegmentsManager
    segmenter = SegmentsManager(phase_enhanced)
    labels = segmenter.stardist_segmentation(
        probability_threshold=0.6,
        nms_threshold=0.3
    )
    
    # Analyze cells
    manager = CellManager(
        label_image=labels,
        membrane_image=membrane_normalized,
        dna_image=dna_normalized,
        pixel_size=pixel_size
    )
    
    # Run analysis steps
    manager.compute_morphology()
    manager.compute_intensity()
    manager.classify_cell_cycle("S.aureus DNA+Membrane Epi")
    manager.compute_colocalization()
    
    # Convert to DataFrame for easy manipulation
    results_df = pd.DataFrame(manager.get_results())
    
    # Apply quality filters
    filtered_df = results_df[
        (results_df['area_um2'] > 1.0) &
        (results_df['area_um2'] < 8.0) &
        (results_df['circularity'] > 0.4) &
        (results_df['cycle_confidence'] > 0.6)
    ]
    
    return filtered_df
```

### Integration with Scientific Computing Stack

```python
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def advanced_statistical_analysis(results_df: pd.DataFrame) -> None:
    """Advanced statistical analysis of cell measurements."""
    
    # Dimensionality reduction
    feature_columns = ['area_um2', 'perimeter_um', 'circularity', 
                      'aspect_ratio', 'mean_membrane', 'mean_dna']
    features = results_df[feature_columns].values
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    
    # Clustering analysis
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(features)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Cell cycle distribution
    axes[0,0].pie(results_df['cell_cycle_phase'].value_counts().values,
                  labels=results_df['cell_cycle_phase'].value_counts().index,
                  autopct='%1.1f%%')
    axes[0,0].set_title('Cell Cycle Distribution')
    
    # Morphology scatter plot
    axes[0,1].scatter(results_df['area_um2'], results_df['aspect_ratio'],
                      c=clusters, alpha=0.6)
    axes[0,1].set_xlabel('Area (μm²)')
    axes[0,1].set_ylabel('Aspect Ratio')
    axes[0,1].set_title('Morphological Clusters')
    
    # PCA visualization
    axes[1,0].scatter(pca_result[:, 0], pca_result[:, 1], 
                      c=clusters, alpha=0.6)
    axes[1,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[1,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[1,0].set_title('PCA Analysis')
    
    # Correlation heatmap
    correlation_matrix = results_df[feature_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, ax=axes[1,1])
    axes[1,1].set_title('Feature Correlations')
    
    plt.tight_layout()
    plt.show()
```

### Custom Model Integration

```python
import tensorflow as tf
from napari_mAIcrobe.mAIcrobe.cells import CellCycleClassifier

def train_custom_classifier(
    training_data: np.ndarray,
    training_labels: np.ndarray,
    validation_split: float = 0.2
) -> tf.keras.Model:
    """Train a custom cell cycle classification model."""
    
    # Define model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                              input_shape=training_data.shape[1:]),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation='softmax')  # 4 cell cycle phases
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        training_data, training_labels,
        epochs=50,
        batch_size=32,
        validation_split=validation_split,
        verbose=1
    )
    
    return model

def use_custom_model(model_path: str, cell_images: np.ndarray) -> dict:
    """Use custom trained model for classification."""
    
    classifier = CellCycleClassifier()
    classifier.load_custom_model(model_path)
    
    results = classifier.predict(cell_images, input_type="Custom")
    
    return results
```

## 🐛 Error Handling

### Common Exception Types

```python
from napari_mAIcrobe.exceptions import (
    SegmentationError,
    ClassificationError,
    ModelLoadError,
    DataFormatError
)

try:
    # Perform analysis
    results = analyze_cells(image_data, parameters)
    
except SegmentationError as e:
    print(f"Segmentation failed: {e}")
    # Fallback to alternative segmentation method
    
except ClassificationError as e:
    print(f"Cell cycle classification failed: {e}")
    # Continue with morphological analysis only
    
except ModelLoadError as e:
    print(f"Model loading failed: {e}")
    # Use default model or skip classification
    
except DataFormatError as e:
    print(f"Data format error: {e}")
    # Check input data format and preprocessing
```

### Best Practices for Error Handling

```python
def robust_analysis_pipeline(image_path: str, parameters: dict) -> dict:
    """Robust analysis pipeline with comprehensive error handling."""
    
    try:
        # Load and validate image
        image_data = load_image(image_path)
        validate_image_format(image_data)
        
        # Attempt segmentation
        try:
            labels = segment_cells(image_data, parameters['segmentation'])
        except SegmentationError:
            # Try alternative segmentation method
            labels = fallback_segmentation(image_data)
        
        # Perform analysis
        results = analyze_cells(labels, image_data, parameters['analysis'])
        
        # Validate results
        if len(results) == 0:
            raise ValueError("No cells detected in analysis")
            
        return results
        
    except Exception as e:
        # Log error and return empty results
        logging.error(f"Analysis failed for {image_path}: {e}")
        return {}
```

## 📖 Further Resources

### Documentation Links

- **[User Guide](../user-guide/getting-started.md)** - Complete user documentation
- **[Tutorials](../tutorials/basic-workflow.md)** - Step-by-step examples
- **GitHub Repository**: [napari-mAIcrobe](https://github.com/HenriquesLab/napari-mAIcrobe)

### External Dependencies

- **napari**: [napari.org](https://napari.org/) - Multi-dimensional image viewer
- **TensorFlow**: [tensorflow.org](https://www.tensorflow.org/) - Machine learning framework
- **scikit-image**: [scikit-image.org](https://scikit-image.org/) - Image processing
- **StarDist**: [StarDist GitHub](https://github.com/stardist/stardist) - Object detection
- **Cellpose**: [Cellpose GitHub](https://github.com/MouseLand/cellpose) - Cell segmentation

---

**Support**: For API questions and issues, visit our [GitHub Discussions](https://github.com/HenriquesLab/napari-mAIcrobe/discussions).