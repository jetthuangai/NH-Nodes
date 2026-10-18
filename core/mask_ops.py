"""Mask helpers shared by the NH mask and image nodes.

NH-Nodes' own code (MIT). It is deliberately not part of core/utils_mask.py: that file is third-party
FitDiT code under CC BY-NC-SA 4.0 (see THIRD_PARTY_NOTICES.md), and plain MIT nodes must not depend on it.
"""

import numpy as np


def mask_bbox(mask, threshold=0.5):
    """Inclusive bounding box ``(x1, y1, x2, y2)`` of the pixels of a 2-D mask that are above ``threshold``.

    ``mask`` is an [H, W] array (a torch mask goes through ``.cpu().numpy()`` first). A pixel counts only when it is
    strictly greater than ``threshold``, so a value of exactly 0.5 is outside, as in every NH mask node. The result is
    four Python ints; (x2, y2) is the last pixel inside, so width = x2 - x1 + 1 and height = y2 - y1 + 1.
    Returns None when no pixel is above the threshold, i.e. for an empty mask. Anything that is not 2-D raises ValueError.
    """
    inside = np.asarray(mask) > threshold
    if inside.ndim != 2:
        raise ValueError(f"mask_bbox expects a 2-D [H, W] mask, got shape {inside.shape}")
    rows = np.flatnonzero(inside.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(inside.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])
