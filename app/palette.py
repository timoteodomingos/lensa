import glasbey
import numpy as np
from scipy.spatial import cKDTree

NOISE = {"Unlabelled", "", None}


def make_position_palette(labels, coords):
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    labels = np.asarray(labels, dtype=object)

    is_label = np.array(
        [not (v in NOISE or (isinstance(v, float) and np.isnan(v))) for v in labels]
    )

    filled = labels.copy()
    if (~is_label).any():
        tree = cKDTree(coords[is_label])
        _, idx = tree.query(coords[~is_label], k=1)
        filled[~is_label] = labels[is_label][idx]

    uniq = sorted(set(filled), key=str)
    palette = glasbey.create_palette(
        palette_size=len(uniq),
        lightness_bounds=(35, 75),
        chroma_bounds=(25, 40),
    )
    color_map = {u: palette[i] for i, u in enumerate(uniq)}

    return np.array([color_map[u] for u in filled])
