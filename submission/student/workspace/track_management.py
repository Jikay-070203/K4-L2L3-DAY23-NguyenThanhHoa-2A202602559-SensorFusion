"""Track initialization, scoring, and deletion helpers.

Part H supplies lidar-driven existence decisions (docs/HUONG_DAN_KY_THUAT.md §2).
Use tracking parameters for the score window, thresholds, and covariance limit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusion_lab.workspace_support import get_tracking_params

params = get_tracking_params()


def init_track_state_from_meas(meas: Any) -> dict[str, Any]:
    """Initialize track state, covariance, lifecycle state, and score from a measurement.

    Args:
        meas: Lidar measurement with ``z``, ``R``, ``sensor``.

    Returns:
        Dict with keys ``x``, ``P``, ``state``, ``score`` (matrices as ``np.matrix``).
    """
    sens_to_veh = np.asarray(meas.sensor.sens_to_veh, dtype=float)
    z_h = np.vstack([np.asarray(meas.z, dtype=float).reshape(3, 1), [[1.0]]])
    pos_veh = (sens_to_veh @ z_h)[:3]
    x = np.asmatrix(np.vstack([pos_veh, np.zeros((3, 1))]))

    rot = sens_to_veh[:3, :3]
    P = np.zeros((params.dim_state, params.dim_state))
    P[:3, :3] = rot @ np.asarray(meas.R, dtype=float) @ rot.T
    P[3, 3] = params.sigma_p44**2
    P[4, 4] = params.sigma_p55**2
    P[5, 5] = params.sigma_p66**2
    return {
        "x": x,
        "P": np.asmatrix(P),
        "state": "initialized",
        "score": 1.0 / params.window,
    }


def update_track_score(track: dict[str, Any], associated: bool) -> dict[str, Any]:
    """Update existence once per lidar frame; camera passes never call this helper.

    A hit adds 1/window, capped at one; an in-FOV miss subtracts 1/window.
    Confirm above confirmed_threshold, and preserve confirmed state after misses.

    Args:
        track: Dict-like track with ``score``, ``state``.
        associated: True for a lidar hit; False for a lidar miss within the lidar FOV.

    Returns:
        Updated track dict.
    """
    step = 1.0 / params.window
    if associated:
        track["score"] = min(1.0, track["score"] + step)
    else:
        track["score"] = track["score"] - step
    if track["score"] > params.confirmed_threshold:
        track["state"] = "confirmed"
    elif associated and track["state"] != "confirmed":
        track["state"] = "tentative"
    return track


def should_delete_track(track: dict[str, Any]) -> bool:
    """Return whether a lidar lifecycle pass should remove this track.

    Delete if either horizontal variance exceeds max_P, or if a confirmed
    track has score < delete_threshold, or an unconfirmed track has score <= 0.
    Camera passes never trigger deletion.

    Args:
        track: Dict with ``score``, ``state``, ``P``.

    Returns:
        True if track should be removed.
    """
    P = track["P"]
    if P[0, 0] > params.max_P or P[1, 1] > params.max_P:
        return True
    if track["state"] == "confirmed":
        return bool(track["score"] < params.delete_threshold)
    return bool(track["score"] <= 0)
