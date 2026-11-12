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



def qmul(q, p):

    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)

    q0, qv = q[0], q[1:]
    p0, pv = p[0], p[1:]

    scalar = q0 * p0 - qv @ pv
    vector = q0 * pv + p0 * qv + np.cross(qv, pv)
    return np.concatenate(([scalar], vector))

def q_rotate_vector(q, v):

    v = np.asarray(v, dtype=float)
    pure = np.concatenate(([0.0], v))
    out = qmul(qmul(q, pure), q_conjugate(q))

    return out[1:]

def q_to_dcm(q):

    q = np.asarray(q, dtype= float)

    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-w*z),    2*(x*z+w*y)],
        [2*(x*y+w*z),    1-2*(x*x+z*z),  2*(y*z-w*x)],
        [2*(x*z-w*y),    2*(y*z+w*x),    1-2*(x*x+y*y)]
    ])

def skew(a):
    a = np.asarray(a, dtype=float)
    return np.array([[    0, -a[2],  a[1]],
                     [ a[2],     0, -a[0]],
                     [-a[1],  a[0],     0]])

def q_dot(q, omega):
    omega = np.asarray(omega, dtype=float)
    pure = np.concatenate(([0.0], omega))
    return 0.5 * qmul(q, pure)

def rk4_step(f, t, s, h):
    k1 = f(t, s)
    k2 = f(t + h/2, s  + (h/2) * k1)
    k3 = f(t + h/2, s + (h/2) * k2)
    k4 = f(t + h, s + h * k3)
    return s + (h/6) * (k1 +2*k2 + 2*k3 + k4)

def omega_true(t):
    return np.array([0.35*np.sin(0.4*t), 0.25, 0.20*np.cos(0.3*t)])

def simulate_truth(omega_fn, T=20.0, dt=0.01):
    n = int(T/dt)
    ts = np.zeros(n+1)
    qs = np.zeros((n+1, 4))
    ws = np.zeros((n+1, 3))

    q = np.array([1.0, 0, 0, 0])
    qs[0] = q
    ws[0] = omega_fn(0.0)

    f = lambda t, qq: q_dot(qq, omega_fn(t))

    for i in range(n):
        t = i * dt
        q = q_normalize(rk4_step(f, t, q, dt))
        ts[i+1] = t + dt
        qs[i+1] = q
        ws[i+1] = omega_fn(t + dt)

    return ts, qs, ws

def simulate_sensors(qs, ws, dt, seed=42,
                     sigma_gyro=0.01, sigma_bias=0.0005, sigma_accel=0.02,
                     bias0=np.array([0.02, -0.015, 0.01])):

    rng = np.random.default_rng(seed)
    n = len(qs)
    gyro = np. zeros((n, 3))
    accel = np.zeros((n, 3))
    bias_true = np.zeros((n, 3))


    g_world = np.array([0, 0, -1.0])
    b = bias0.copy()
    for i in range(n):
         bias_true[i] = b
         gyro[i] = ws[i] +b + rng.normal(0, sigma_gyro, 3)
        
         accel[i] = q_rotate_vector(q_conjugate(qs[i]), g_world) + rng.normal(0, sigma_accel, 3)
        
         b = b + rng.normal(0, sigma_bias*np.sqrt(dt), 3)

    return gyro, accel, bias_true

   


def predict(x, gyro_reading, dt, t=0.0):
    q = x[:4]
    b = x[4:]

    omega_hat = gyro_reading - b
    f = lambda tt, qq: q_dot(qq, omega_hat)
    q_new = q_normalize(rk4_step(f, t, q, dt))
    return np.concatenate((q_new, b))