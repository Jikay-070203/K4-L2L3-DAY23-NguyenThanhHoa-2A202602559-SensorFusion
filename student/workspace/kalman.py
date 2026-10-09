"""Extended Kalman filter helpers for 6D constant-velocity motion.

Part E supplies prediction and correction for docs/HUONG_DAN_KY_THUAT.md §2.
Read the shared time step and process-noise settings with get_tracking_params().
"""

from __future__ import annotations

from typing import Any
from typing import Optional

import numpy as np

from fusion_lab.workspace_support import get_tracking_params

Matrix = np.matrix | np.ndarray

params = get_tracking_params()


def build_F(dt: Optional[float] = None) -> Matrix:
    """Build the constant-velocity state transition matrix F.

    Args:
        dt: Time step in seconds; default from tracking params.

    Returns:
        6x6 state transition matrix as ``np.matrix``.
    """
    if dt is None:
        dt = params.dt
    F = np.asmatrix(np.eye(params.dim_state))
    F[0, 3] = F[1, 4] = F[2, 5] = dt  # position += velocity * dt
    return F


def build_Q(dt: Optional[float] = None, q: Optional[float] = None) -> Matrix:
    """Build the process noise covariance matrix Q.

    Args:
        dt: Time step; default from tracking params.
        q: Process noise scale; default from tracking params.

    Returns:
        6x6 process noise matrix.
    """
    if dt is None:
        dt = params.dt
    if q is None:
        q = params.q
    return np.asmatrix(np.diag([dt * q] * params.dim_state))


def ekf_predict(
    x: Matrix,
    P: Matrix,
    F: Optional[Matrix] = None,
    Q: Optional[Matrix] = None,
) -> tuple[Matrix, Matrix]:
    """Predict state and covariance one time step forward.

    Args:
        x: State vector (6x1).
        P: State covariance (6x6).
        F: Optional transition matrix; build via ``build_F`` if None.
        Q: Optional process noise; build via ``build_Q`` if None.

    Returns:
        Tuple ``(x_pred, P_pred)``.
    """
    if F is None:
        F = build_F()
    if Q is None:
        Q = build_Q()
    F, Q = np.asmatrix(F), np.asmatrix(Q)
    x, P = np.asmatrix(x), np.asmatrix(P)
    return F @ x, F @ P @ F.T + Q


def innovation(x: Matrix, meas: Any) -> Matrix:
    """Compute the measurement residual (innovation) gamma.

    Args:
        x: Predicted state.
        meas: Measurement with ``z`` and ``sensor.get_hx(x)``.

    Returns:
        Innovation vector ``z - h(x)``.
    """
    return meas.z - meas.sensor.get_hx(x)


def innovation_covariance(P: Matrix, meas: Any, H: Matrix) -> Matrix:
    """Compute the innovation covariance S = H P H' + R.

    Args:
        P: State covariance.
        meas: Measurement with ``R``.
        H: Measurement Jacobian.

    Returns:
        Innovation covariance matrix S.
    """
    H = np.asmatrix(H)
    return H @ np.asmatrix(P) @ H.T + meas.R


def ekf_update(x: Matrix, P: Matrix, meas: Any) -> tuple[Matrix, Matrix]:
    """Apply an EKF measurement update and return updated state and covariance.

    Args:
        x: Prior state.
        P: Prior covariance.
        meas: Associated measurement.

    Returns:
        Tuple ``(x_upd, P_upd)``.
    """
    x, P = np.asmatrix(x), np.asmatrix(P)
    H = np.asmatrix(meas.sensor.get_H(x))
    gamma = innovation(x, meas)
    S = innovation_covariance(P, meas, H)
    K = P @ H.T @ np.linalg.inv(S)
    identity = np.asmatrix(np.eye(P.shape[0]))
    return x + K @ gamma, (identity - K @ H) @ P
