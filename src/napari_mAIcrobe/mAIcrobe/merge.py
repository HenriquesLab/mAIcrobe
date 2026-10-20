"""
Module responsible for optional pre/post-processing of segmentation:
Gaussian smoothing of the input image and merging of septated cell masks.
"""

import numpy as np
from scipy import ndimage
from skimage.filters import gaussian
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential


def gaussian_smooth(image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Gaussian blur so a septum no longer reads as a cell boundary.

    Works on 2D images or (T, Y, X) stacks (each frame is blurred
    independently). Used for segmentation input only.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:
        return np.stack([gaussian_smooth(f, sigma) for f in img])
    return gaussian(img, sigma=sigma, preserve_range=True).astype(np.float32)


def merge_septated_cells(
    labels: np.ndarray,
    pixel_um: float,
    min_solidity: float = 0.91,
    max_aspect: float = 1.55,
    contact_width_ratio: tuple[float, float] = (0.6, 1.2),
    max_union_um2: float = 3.0,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Merge pairs of touching masks that together look like one dividing cell.

    A pair is merged when the union is compact and near-round (high solidity,
    low aspect ratio), the shared boundary spans the union's width like a
    septum, and the union is cell-sized. Each mask joins at most one merge.
    Returns the relabelled image and the merged (label_a, label_b) pairs, in
    the input labelling.
    """
    px2 = pixel_um**2
    shared: dict[tuple[int, int], list] = {}
    for dr, dc in ((0, 1), (1, 0)):
        a = labels[: labels.shape[0] - dr, : labels.shape[1] - dc]
        b = labels[dr:, dc:]
        rows, cols = np.nonzero((a > 0) & (b > 0) & (a != b))
        for r, c in zip(rows, cols):
            key = (int(min(a[r, c], b[r, c])), int(max(a[r, c], b[r, c])))
            shared.setdefault(key, []).extend(([r, c], [r + dr, c + dc]))

    contacts = {}
    for key, points in shared.items():
        pts = np.unique(np.asarray(points, dtype=float), axis=0)
        if len(pts) < 2:
            contacts[key] = 1.0
            continue
        # Septum length = straight extent of the contact along its main
        # direction (counting pixel pairs overestimates diagonal septa).
        centered = pts - pts.mean(axis=0)
        direction = np.linalg.eigh(np.cov(centered.T))[1][:, -1]
        along = centered @ direction
        contacts[key] = float(along.max() - along.min()) + 1.0

    areas = np.bincount(labels.ravel())
    candidates = []
    for (x, y), length in contacts.items():
        if (areas[x] + areas[y]) * px2 > max_union_um2:
            continue
        rows, cols = np.nonzero((labels == x) | (labels == y))
        r0, c0 = rows.min(), cols.min()
        union = np.zeros(
            (rows.max() - r0 + 3, cols.max() - c0 + 3), dtype=np.uint8
        )
        union[rows - r0 + 1, cols - c0 + 1] = 1
        # close a possible 1 px gap between the two masks
        union = ndimage.binary_closing(union, iterations=1).astype(np.uint8)
        props = regionprops(union)[0]
        minor = max(props.minor_axis_length, 1e-6)
        aspect = props.major_axis_length / minor
        width = length / minor
        if (
            props.solidity >= min_solidity
            and aspect <= max_aspect
            and contact_width_ratio[0] <= width <= contact_width_ratio[1]
        ):
            candidates.append((props.solidity, x, y))

    merged: list[tuple[int, int]] = []
    used: set[int] = set()
    out = labels.copy()
    for _solidity, x, y in sorted(candidates, reverse=True):
        if x in used or y in used:
            continue
        used.update((x, y))
        out[out == y] = x
        merged.append((x, y))
    return relabel_sequential(out)[0].astype(labels.dtype), merged
