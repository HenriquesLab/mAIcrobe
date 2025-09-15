# API Reference

This comprehensive reference covers programmatic usage of napari-mAIcrobe, enabling automation, batch processing, and integration into custom analysis pipelines.

## 🎯 Overview

napari-mAIcrobe provides both GUI widgets and programmatic APIs:

- **Widget APIs**: Direct access to plugin functionality
- **Core Libraries**: Low-level analysis components
- **Utility Functions**: Helper functions for common tasks

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



## 📖 Further Resources

### Documentation Links

- **[User Guide](../user-guide/getting-started.md)** - Complete user documentation
- **[Tutorials](../tutorials/basic-workflow.md)** - Step-by-step examples
- **GitHub Repository**: [napari-mAIcrobe](https://github.com/HenriquesLab/napari-mAIcrobe)

### External Dependencies

- **napari**: [napari.org](https://napari.org/) - Multi-dimensional image viewer
- **TensorFlow**: [tensorflow.org](https://www.tensorflow.org/) - Machine learning framework
- **scikit-image**: [scikit-image.org](https://scikit-image.org/) - Image processing
- **StarDist**: [StarDist GitHub](https://github.com/stardist/stardist) - Cell segmentation
- **Cellpose**: [Cellpose GitHub](https://github.com/MouseLand/cellpose) - Cell segmentation

---

**Support**: For API questions and issues, visit our [GitHub Discussions](https://github.com/HenriquesLab/napari-mAIcrobe/discussions).