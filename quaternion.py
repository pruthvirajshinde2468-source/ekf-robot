import numpy as np

def q_from_axis_angle(axis, angle_deg):

    axis = np.asarray(axis, dtype=float)

    length = np.linalg.norm(axis)
    if length < 1e-12:
        raise ValueError("axis must have non-zero length")

    axis = axis / length

    half = np.radians(angle_deg) / 2

    w = np.cos(half)
    v = np.sin(half) * axis

    return np.concatenate(([w], v))

def q_conjugate(q):
    q = np.asarray(q, dtype=float)

    w = q[0]
    v = q[1:]*-1

    return np.concatenate(([w], v))

def q_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Cannot normalize a zero-length quaternion")
    return q / n



