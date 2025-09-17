# Cell Segmentation Guide

This guide helps you choose the optimal segmentation method for your bacterial images and achieve the best possible cell detection results.

## 🎯 Overview

napari-mAIcrobe offers four main segmentation approaches:

1. **StarDist models**
2. **Cellpose cyto3 model**
3. **Custom U-Net Models**
4. **Thresholding-based methods** - Classical image processing using [Isodata](https://scikit-image.org/docs/0.25.x/api/skimage.filters.html#skimage.filters.threshold_isodata) or [Local Average](https://scikit-image.org/docs/0.25.x/api/skimage.filters.html#skimage.filters.threshold_local) tresholding followed by euclidean distance transform and watershed segmentation. This method is fast and does not require training data, but may be less accurate for complex images.

## Stardist

1. Check the original [StarDist paper](https://arxiv.org/abs/1806.03535) and [repository](https://github.com/stardist/stardist) for details on how StarDist works!
2. Deep learning-based segmentation based on star-convex shapes detection.
3. mAIcrobe does not include pre-trained StarDist models, you have to provide your own model. To train your own model, check out the [StarDist examples](https://github.com/stardist/stardist/tree/main/examples/2D). We also provide an example notebook to train your own StarDist2D model in [notebooks/StarDistSegmentationTraining.ipynb](../../notebooks/StarDistSegmentationTraining.ipynb).

## Cellpose
1. Check the original [Cellpose paper](https://www.nature.com/articles/s41592-020-01018-x) and [repository](https://github.com/MouseLand/cellpose) for details on how Cellpose works!
2. Deep learning-based universal segmentation model trained on a large variety of cell types and imaging modalities.
3. mAIcrobe includes the Cellpose cyto3 model, which is pre-trained and ready to use.
4. The first time you run Cellpose, it will download the model weights from the Cellpose repository.

## U-Net Models
1. Check the original [U-Net paper](https://arxiv.org/abs/1505.04597) for details on how U-Net works!
2. Deep learning-based segmentation using a convolutional neural network architecture.
3. mAIcrobe does not include pre-trained U-Net models, you have to provide your own model in Keras format (.keras).
4. You can train your own U-Net segmentation models using [ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic).
5. The U-Net model is assumed to output a label image with 0 for background, 1 for cell interior, and 2 for cell boundary. mAIcrobe will convert this to a label image with unique integer IDs for each cell via a watershed algorithm. Check [skimage.segmentation.watershed](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed) for more info.

## Thresholding-based Methods
1. Classical image processing methods that do not require training data.
2. Fast and easy to use, but may be less accurate for complex images.
3. Two methods available:
   - **Isodata**: Global thresholding method that automatically determines the optimal threshold value based on image histogram. Check [skimage.filters.threshold_isodata](https://scikit-image.org/docs/0.25.x/api/skimage.filters.html#skimage.filters.threshold_isodata) for more info.
   - **Local Average**: Adaptive thresholding method that computes a local threshold for each pixel based on the average intensity in its neighborhood. Check [skimage.filters.threshold_local](https://scikit-image.org/docs/0.25.x/api/skimage.filters.html#skimage.filters.threshold_local) for more info.
4. After thresholding, a distance transform and watershed algorithm is applied to separate touching cells. Check [skimage.filters](https://scikit-image.org/docs/0.25.x/api/skimage.filters.html) and [skimage.segmentation.watershed](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed) for more info.


## 📏 Validation and Quality Control

### Manual Validation

Always validate segmentation on a representative subset:

1. **Random sampling**: Check 50-100 cells
2. **Visual inspection**: Look for common errors

### Automated Quality Metrics

**Segmentation Quality Indicators:**
- Cell count consistency
- Size distribution reasonableness (look for outliers on the size distribution)


## 📚 Further Reading

- **[Cell Analysis Guide](cell-analysis.md)** - What to do after segmentation
- **[Cell Classification Guide](cell-classification.md)** - Cell classification
- **[API Reference](../api/api-reference.md)** - Programmatic control
- **StarDist Paper**: [Schmidt et al., MICCAI 2018](https://arxiv.org/abs/1806.03535)
- **Cellpose Paper**: [Stringer et al., Nature Methods 2021](https://doi.org/10.1038/s41592-020-01018-x)
- **U-Net Paper**: [Ronneberger et al., MICCAI 2015](https://arxiv.org/abs/1505.04597)
- **scikit-image watershed**: [Documentation](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed)
- **scikit-image filters**: [Documentation](https://scikit-image.org/docs/stable/api/skimage.filters.html)

---

**Next:** Learn how to analyze your segmented cells in the [Cell Analysis Guide](cell-analysis.md).
