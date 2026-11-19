import numpy as np
from quaternion import q_dot, q_normalize, rk4_step, q_rotate_vector, q_conjugate

def omega_true(t):
    return np.array([0.35*np.sin(0.4*t), 0.25, 0.20*np.cos(0.3*t)])

def simulate_truth(omega_fn, T=20.0, dt=0.01):
    n = int(round(T/dt))
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