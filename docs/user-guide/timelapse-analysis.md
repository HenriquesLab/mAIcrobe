# Timelapse Analysis Guide

This guide covers how to analyze 2D+t datasets in mAIcrobe, including optional label reassignment (tracking-like stable IDs) and image registration.

## Overview

Timelapse support is available in the **Compute label** and **Compute cells** workflows.

- Input data shape for timelapse mode is `(T, Y, X)`
- Segmentation is performed frame by frame
- Optional label reassignment can keep stable IDs across consecutive frames
- Optional registration can correct frame drift before segmentation

## Step 1: Prepare timelapse data

Use one base timelapse image and up to two fluorescence timelapse channels.

- Base image: phase or transmitted light stack `(T, Y, X)`
- Fluor 1 and Fluor 2: fluorescence stacks with the same number of frames as Base Image

Important checks:

- All stacks must represent the same field of view
- Frame count must match between channels
- Avoid mixing 2D single-frame images with timelapse stacks in the same run

## Step 2: Compute labels in timelapse mode

Open `Plugins > mAIcrobe > Compute label` and configure your segmentation model as usual.

mAIcrobe will automatically detect timelapse data and enable timelapse-specific options:

- Enable **Run analysis for all time points**
- Optionally enable **Perform image registration before segmentation**
- Optionally enable **Enable frame-to-frame cell tracking**

### Registration options

When registration is enabled, mAIcrobe estimates drift from the base stack and applies correction to fluorescence channels before downstream analysis.

- Use this when you observe stage/sample drift over time
- Keep it disabled when data is already drift-corrected or drift is negligible

Parameters:

- Reference frame: Choose a frame to serve as the reference for registration. Register all frames relative to the first frame or relative to the previous frame.

This registration algorithm is based upon the one implemented in NanoPyx. Check the [NanoPyx documentation](https://github.com/HenriquesLab/Nanopyx) for more details.

### Label reassignment behavior

When enabled, labels will be reassigned across frames to stabilize cell IDs through time.

- Matches are based on overlap between consecutive frames
- Splits generate new IDs for both daughter regions
- IDs from disappeared objects are not reused

This behavior is derived from logic based on [ReScale4DL](https://github.com/HenriquesLab/ReScale4DL) concepts.

## Step 3: Compute cells on timelapse labels

Open `Plugins > mAIcrobe > Compute cells` and select the timelapse label layer.

In timelapse mode:

- Frames are processed sequentially
- Output properties include a `frame` column
- Reports are generated in a timelapse-aware format

Recommended settings:

- Keep report generation enabled for QA and reproducibility
- Use filtering after compute cells to remove low-quality detections

## Common pitfalls

- **Mismatched frame count**: Make sure all channels have identical `T`
- **Unexpected ID jumps**: If reassignment is disabled, IDs are independent per frame
- **Slow runs on long movies**: Start with shorter clips to tune segmentation parameters

## Related pages

- [Batch Analysis Tutorial](batch-analysis.md)
- [Cell Analysis](cell-analysis.md)
- [Segmentation Guide](segmentation-guide.md)
- [Basic Workflow](../tutorials/basic-workflow.md)
