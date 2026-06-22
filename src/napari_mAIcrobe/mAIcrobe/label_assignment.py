import numpy as np

"""
THE FOLLOWING FUNCTIONS WERE COPIED FROM THE RESCALE4DL PACKAGE
https://github.com/HenriquesLab/ReScale4DL/blob/main/src/rescale4dl/utils.py
"""


def compute_labels_matching_scores(gt: np.array, pred: np.array):
    """
    Compute matching scores between ground truth and predicted labels.

    Parameters
    ----------
    gt : np.array
        Ground truth labels.
    pred : np.array
        Predicted labels.

    Returns
    -------
    dict
        Dictionary with gt_label as keys and a list of tuples (pred_label, score) as values.
    """
    scores = {}
    gt_labels = np.unique(gt)

    for lbl in gt_labels[1:]:  # skips the background label
        scores[lbl] = []
        rows_idx, cols_idx = np.nonzero(gt == lbl)
        min_row, max_row, min_col, max_col = (
            np.min(rows_idx),
            np.max(rows_idx),
            np.min(cols_idx),
            np.max(cols_idx),
        )
        pred_box = pred[min_row : max_row + 1, min_col : max_col + 1]
        pred_labels_in_box = np.unique(pred_box)
        for pred_lbl in pred_labels_in_box:
            score = score_label_overlap(gt, pred, lbl, pred_lbl)
            scores[lbl].append([pred_lbl, score])

        scores[lbl] = sorted(scores[lbl], key=lambda x: x[1], reverse=True)

    return scores


def score_label_overlap(gt: np.array, pred: np.array, gt_label, pred_label):
    """
    Calculate the score of label overlap between ground truth and prediction.

    Parameters
    ----------
    gt : np.array
        Ground truth labels.
    pred : np.array
        Predicted labels.
    gt_label : int
        Label in ground truth.
    pred_label : int
        Label in prediction.

    Returns
    -------
    float
        Score of label overlap.
    """
    gt_mask = gt == gt_label
    pred_mask = pred == pred_label

    intersection = np.sum(gt_mask & pred_mask)
    union = np.sum(gt_mask | pred_mask)

    if union == 0:
        score = 0.0
    else:
        score = intersection / union

    return score


def remove_duplicates(scores, pred_labels):
    """
    Resolve conflicts in the scores dictionary by ensuring each pred_label
    is assigned to the gt_label with the highest score. If a pred_label has no
    assignment in the ground truth, assign it to 0.

    Parameters
    ----------
    scores : dict
        Dictionary with gt_label as keys and a list of tuples (pred_label, score) as values.
    pred_labels : np.array
        Array of unique predicted labels.

    Returns
    -------
    list
        List of tuples (gt_label, pred_label, score) with resolved conflicts.
    """
    assigned_pred_labels = set()
    result = []

    # Sort gt_labels by their highest score to prioritize them
    sorted_gt_labels = sorted(
        scores.keys(),
        key=lambda lbl: scores[lbl][0][1] if scores[lbl] else 0,
        reverse=True,
    )

    for gt_label in sorted_gt_labels:
        for pred_label, score in scores[gt_label]:
            if pred_label not in assigned_pred_labels:
                result.append((gt_label, pred_label, score))
                assigned_pred_labels.add(pred_label)
                break

    # Add unmatched pred_labels with gt_label = 0
    for pred_label in pred_labels:
        if pred_label not in assigned_pred_labels:
            result.append((0, pred_label, 0.0))

    return result


def find_matching_labels(gt: np.array, pred: np.array):
    """
    Find the matching labels between ground truth and prediction. If a pred_label
    has no assignment in the ground truth, assign it to 0.

    Parameters
    ----------
    gt : np.array
        Ground truth labels.
    pred : np.array
        Predicted labels.

    Returns
    -------
    list
        List of tuples (gt_label, pred_label, score).
    """
    if np.unique(pred).shape[0] == 1:
        return ((gt_lbl, 0, 0) for gt_lbl in np.unique(gt))

    scores = compute_labels_matching_scores(gt, pred)
    pred_labels = np.unique(pred)

    # Process scores to resolve conflicts and get final matching labels
    matching_labels = remove_duplicates(scores, pred_labels)
    return matching_labels


def _remap_labels(frame: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Return a relabeled frame according to ``mapping`` for non-zero labels."""
    remapped = np.zeros(frame.shape, dtype=np.int32)
    for src, dst in mapping.items():
        remapped[frame == src] = dst
    return remapped


def relabel_timelapse_labels(labels: np.ndarray, iou_threshold: float = 0.1):
    """Relabel timelapse labels to keep stable IDs across consecutive frames.

    Rules
    -----
    - Operates only on 3D arrays with shape ``(T, H, W)``.
    - IDs are monotonic and never reused.
    - One-to-one matches keep the previous ID.
    - If one parent overlaps multiple children above threshold (split),
      all children receive new IDs.

    Parameters
    ----------
    labels : np.ndarray
        Input labels with shape ``(T, H, W)``.
    iou_threshold : float, optional
        Minimum IoU score to consider a label correspondence valid.

    Returns
    -------
    np.ndarray
        Relabeled array with dtype ``int32``.
    """
    if labels.ndim != 3:
        raise ValueError(
            "relabel_timelapse_labels expects a 3D array (T, H, W)"
        )

    tracked = np.zeros(labels.shape, dtype=np.int32)
    if labels.shape[0] == 0:
        return tracked

    first = labels[0].astype(np.int32, copy=False)
    tracked[0] = first
    first_ids = np.unique(first)
    first_ids = first_ids[first_ids != 0]
    next_id = int(np.max(first_ids)) + 1 if first_ids.size else 1

    for t in range(1, labels.shape[0]):
        prev_frame = tracked[t - 1]
        curr_frame = labels[t]
        curr_labels = np.unique(curr_frame)
        curr_labels = curr_labels[curr_labels != 0]

        # Fast path: empty frame
        if curr_labels.size == 0:
            continue

        # Compute scores once and reuse
        scores = compute_labels_matching_scores(prev_frame, curr_frame)

        # Detect splits: one parent → multiple children
        split_children = set()
        split_parents = set()
        for gt_label, gt_scores in scores.items():
            children_above_threshold = [
                pred_label
                for pred_label, score in gt_scores
                if pred_label != 0 and score >= iou_threshold
            ]
            if len(children_above_threshold) >= 2:
                split_parents.add(gt_label)
                split_children.update(children_above_threshold)

        # Build mapping from pred → gt using greedy best-match logic,
        # then filter for split events and threshold
        assigned_pred_labels = set()
        keep_mapping: dict[int, int] = {}
        sorted_gt_labels = sorted(
            scores.keys(),
            key=lambda lbl: scores[lbl][0][1] if scores[lbl] else 0,
            reverse=True,
        )

        for gt_label in sorted_gt_labels:
            for pred_label, score in scores[gt_label]:
                if pred_label == 0 or pred_label in assigned_pred_labels:
                    continue
                if score < iou_threshold:
                    continue
                if gt_label in split_parents or pred_label in split_children:
                    continue
                keep_mapping[int(pred_label)] = int(gt_label)
                assigned_pred_labels.add(pred_label)
                break

        frame_mapping: dict[int, int] = {}
        for pred_label in curr_labels:
            if int(pred_label) in keep_mapping:
                frame_mapping[int(pred_label)] = keep_mapping[int(pred_label)]
            else:
                frame_mapping[int(pred_label)] = next_id
                next_id += 1

        tracked[t] = _remap_labels(curr_frame, frame_mapping)

    return tracked
