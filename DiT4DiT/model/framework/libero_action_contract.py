"""LIBERO-specific inverse of training min/max normalization."""
import numpy as np

def unnormalize_libero_actions(normalized_actions, action_norm_stats):
    values = np.asarray(normalized_actions)
    if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError('INVALID_LIBERO_NORMALIZED_ACTION')
    lo = np.asarray(action_norm_stats['min'])
    hi = np.asarray(action_norm_stats['max'])
    mask = np.asarray(action_norm_stats.get('mask', np.ones_like(lo, dtype=bool)))
    if lo.shape != (7,) or hi.shape != (7,) or mask.shape != (7,) or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi < lo):
        raise ValueError('INVALID_LIBERO_MINMAX_STATS')
    values = np.clip(values, -1, 1)
    values[:, 6] = np.where(values[:, 6] < 0.5, 0, 1)
    return np.where(mask, 0.5 * (values + 1) * (hi - lo) + lo, values)
