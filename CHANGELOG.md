# Changelog

All notable changes to napari-mAIcrobe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation overhaul and updated README
- User guide documentation
- Tutorials and API reference documentation
- Contributing guidelines and development setup instructions
- Compute_label widget enhancements - pretrained models for StarDist2D and U-Net
- Compute_cells widget enhancements - new pretrained classification model
- Other minor improvements and bug fixes

### Changed
- README.md completely rewritten
- Documentation structure reorganized with user-friendly navigation
- Enhanced project description and value proposition

### Documentation
- Added getting started guide with step-by-step tutorial
- Added generate training data documentation with screenshots
- Added notebooks for StarDist segmentation and cell classification
- Created comprehensive segmentation guide
- Detailed cell analysis documentation
- Cell classification guide with model descriptions
- API reference
- Basic workflow tutorial for new users

## [0.0.1] - Initial Release

### Added
- Core napari plugin functionality
- Three main widgets:
  - `compute_label`: Cell segmentation using StarDist2D, Cellpose, or custom U-Net models
  - `compute_cells`: Comprehensive cell analysis with morphological measurements and optionally deep learning classification
  - `filter_cells`: Interactive cell filtering based on computed statistics
- Deep learning cell classification with 6 pre-trained TensorFlow models for cell cycle determination in *S. aureus*:
  - S.aureus DNA+Membrane (Epi/SIM)
  - S.aureus DNA-only (Epi/SIM)
  - S.aureus Membrane-only (Epi/SIM)
- Sample _S. aureus_ datasets for testing:
  - Phase contrast images
  - Membrane fluorescence (Nile Red)
  - DNA fluorescence (Hoechst)
- Morphological analysis with scikit-image regionprops
- Multi-channel colocalization analysis
- HTML report generation with statistics and visualizations
- CSV data export for further analysis
- Support for custom Keras models for cell classification and U-Net segmentation

### Features
- **Cell Segmentation**: StarDist2D, Cellpose, custom U-Net and traditional thresholding methods with watershed
- **Morphometry**: Comprehensive shape and size analysis
- **Classification**: Single cell classification using pre-trained models (S. aureus cell cycle) or custom trained ones
- **Filtering**: Interactive filtering of cell populations
- **Reporting**: HTML reports and CSV exports
- **Custom Models**: Support for user-trained classification models

---


## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md)

## Support

- **Documentation**: Complete guides at [napari-mAIcrobe docs](docs/)
- **Issues**: Report bugs via [GitHub Issues](https://github.com/HenriquesLab/napari-mAIcrobe/issues)
- **Email**: Contact maintainers for security issues or collaboration

---

*This changelog is updated with each release to help users understand what has changed and how to adapt their workflows accordingly.*
