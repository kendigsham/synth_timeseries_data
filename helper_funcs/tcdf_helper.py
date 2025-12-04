from typing import Optional, Dict, Tuple
import numpy as np


def make_matrices(alldelays: Dict[Tuple[int, int], int], columns: list, allscores: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build graph_matrix (bool) and val_matrix (float) from alldelays and allscores.
    alldelays keys expected as (effect_idx, cause_idx) mapping to integer delay.
    """
    if len(alldelays) == 0:
        max_delay = 0
    else:
        max_delay = max(alldelays.values())
    n_vars = len(columns)
    D = int(max_delay) + 1

    graph_matrix = np.zeros((n_vars, n_vars, D), dtype=bool)
    val_matrix = np.zeros((n_vars, n_vars, D), dtype=float)

    for (effect_idx, cause_idx), delay in alldelays.items():
        src = int(cause_idx)
        tgt = int(effect_idx)
        lag_idx = int(delay)
        graph_matrix[src, tgt, lag_idx] = True

        score = 1.0
        sc = allscores.get(tgt)
        if sc is not None:
            try:
                # attempt dict-like lookup
                score = float(sc.get(cause_idx, 1.0))
            except Exception:
                # fallback
                score = 1.0
        val_matrix[src, tgt, lag_idx] = score

    return graph_matrix, val_matrix


def _to_bool_array(arr: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype == bool:
        return a.copy()
    if threshold is None:
        return a != 0
    return a.astype(float) > float(threshold)


def _ensure_3d_bool(arr: np.ndarray, threshold: Optional[float]) -> np.ndarray:
    b = _to_bool_array(arr, threshold)
    if b.ndim == 2:
        b = b[:, :, np.newaxis]
    if b.ndim != 3:
        raise ValueError(f"Adjacency must be 2D or 3D array; got shape {arr.shape}")
    return b




