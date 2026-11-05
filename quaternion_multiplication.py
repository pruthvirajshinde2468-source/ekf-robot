import numpy as np

def qmul(q, p):

    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)

    q0, qv = q[0], q[1:]
    p0, pv = p[0], p[1:]

    scalar = q0 * p0 - qv @ pv
    vector = q0 * pv + p0 * qv + np.cross(qv, pv)
    return np.concatenate(([scalar], vector))

