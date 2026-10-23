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


