# Changelog

All notable changes to napari-mAIcrobe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation overhaul with professional README
- Complete user guide documentation (getting-started, segmentation-guide, cell-analysis, ai-models)
- Tutorials and API reference documentation
- Contributing guidelines and development setup instructions
- Citation file format (CITATION.cff) for academic use
- Enhanced napari hub description

### Changed
- README.md completely rewritten with professional styling and comprehensive feature overview
- Documentation structure reorganized with user-friendly navigation
- Enhanced project description and value proposition

### Documentation
- Added getting started guide with step-by-step tutorial
- Created comprehensive segmentation guide with troubleshooting
- Detailed cell analysis documentation with advanced features
- AI models guide with custom model integration instructions
- API reference with programmatic usage examples
- Basic workflow tutorial for new users

## [0.0.1] - Initial Release

### Added
- Core napari plugin functionality
- Three main widgets:
  - `compute_label`: Cell segmentation using StarDist2D, Cellpose, or custom U-Net models
  - `compute_cells`: Comprehensive cell analysis with morphological measurements and AI classification
  - `filter_cells`: Interactive cell filtering based on computed statistics
- AI-powered cell cycle classification with 6 pre-trained TensorFlow models:
  - S.aureus DNA+Membrane (Epi/SIM)
  - S.aureus DNA-only (Epi/SIM) 
  - S.aureus Membrane-only (Epi/SIM)
- Sample _S. aureus_ datasets for testing:
  - Phase contrast images
  - Membrane fluorescence (FM 4-64)
  - DNA fluorescence (DAPI)
- Morphological analysis with 50+ measurements using scikit-image regionprops
- Multi-channel colocalization analysis
- Professional HTML report generation with statistics and visualizations
- CSV data export for further analysis
- Support for custom TensorFlow models
- CPU-optimized TensorFlow configuration to avoid CUDA conflicts

### Features
- **Cell Segmentation**: StarDist2D and Cellpose integration with parameter optimization
- **Morphometry**: Comprehensive shape and size analysis
- **AI Classification**: Automated cell cycle phase determination (G1, S, G2, Division)
- **Quality Control**: Interactive filtering and validation tools
- **Reporting**: Publication-ready HTML reports with embedded plots
- **Batch Processing**: Consistent analysis across multiple images
- **Custom Models**: Support for user-trained classification models

### Technical
- Python 3.10-3.11 compatibility
- TensorFlow ≤2.15.0 integration with CPU-only configuration
- napari plugin architecture following best practices
- Comprehensive test suite with pytest
- Pre-commit hooks for code quality (black, isort, flake8)
- Cross-platform compatibility (Windows, macOS, Linux)

### Dependencies
- napari[all] for image visualization
- TensorFlow ≤2.15.0 for AI models
- scikit-image for image processing and morphometry
- pandas for data management
- cellpose==3.1.1.1 for alternative segmentation
- stardist-napari==2022.12.6 for bacterial segmentation
- napari-skimage-regionprops for measurements integration

---

## Future Releases

### Planned Features (Roadmap)

#### Version 0.1.0 - Enhanced Analysis
- [ ] Time-lapse analysis capabilities with cell tracking
- [ ] Additional bacterial species support beyond _S. aureus_
- [ ] Advanced colocalization metrics (Manders coefficients, etc.)
- [ ] Batch processing GUI for high-throughput analysis
- [ ] Statistical analysis integration (t-tests, ANOVA)

#### Version 0.2.0 - Advanced AI
- [ ] Transfer learning framework for custom species adaptation
- [ ] Uncertainty quantification for AI predictions
- [ ] Active learning for model improvement
- [ ] Multi-class morphological classification beyond cell cycle
- [ ] Integration with additional deep learning frameworks

#### Version 0.3.0 - Workflow Enhancement
- [ ] Automated quality control metrics
- [ ] Advanced visualization options (3D rendering, animations)
- [ ] Integration with laboratory information management systems (LIMS)
- [ ] Cloud processing capabilities
- [ ] Multi-user collaboration features

#### Version 1.0.0 - Production Ready
- [ ] Complete API stability
- [ ] Comprehensive performance optimizations
- [ ] Full documentation coverage
- [ ] Extensive validation on diverse datasets
- [ ] Publication-ready feature set

---

## Release Process

### Version Release Criteria

**Patch Releases (0.0.x):**
- Bug fixes
- Documentation improvements
- Performance optimizations
- No breaking changes

**Minor Releases (0.x.0):**
- New features
- New AI models
- Enhanced workflows
- Backward compatible changes

**Major Releases (x.0.0):**
- Breaking API changes
- Significant architectural changes
- Major new functionality
- Requires user migration

### Quality Gates

All releases must pass:
- [ ] Complete test suite (>95% coverage)
- [ ] Documentation review and updates
- [ ] Performance benchmarks
- [ ] Cross-platform compatibility testing
- [ ] User acceptance testing with sample data
- [ ] Security vulnerability scanning

### Release Checklist

**Pre-release:**
- [ ] Update version numbers in setup.cfg and __init__.py
- [ ] Update CHANGELOG.md with all changes
- [ ] Run full test suite across supported Python versions
- [ ] Update documentation and verify links
- [ ] Tag release in git with version number

**Post-release:**
- [ ] Deploy to PyPI
- [ ] Update conda-forge recipe
- [ ] Create GitHub release with notes
- [ ] Update napari hub listing
- [ ] Announce in community channels

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup instructions
- Code style guidelines  
- Testing requirements
- Pull request process
- Community guidelines

## Support

- **Documentation**: Complete guides at [napari-mAIcrobe docs](docs/)
- **Issues**: Report bugs via [GitHub Issues](https://github.com/HenriquesLab/napari-mAIcrobe/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/HenriquesLab/napari-mAIcrobe/discussions)
- **Email**: Contact maintainers for security issues or collaboration

---

*This changelog is updated with each release to help users understand what has changed and how to adapt their workflows accordingly.*