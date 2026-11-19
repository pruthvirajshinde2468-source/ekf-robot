import numpy as np
from quaternion import (q_dot, q_normalize, rk4_step, qmul, q_conjugate,
                        q_rotate_vector, q_to_dcm, skew)

def predict(x, gyro_reading, dt, t=0.0):
    q = x[:4]
    b = x[4:]

    omega_hat = gyro_reading - b
    f = lambda tt, qq: q_dot(qq, omega_hat)
    q_new = q_normalize(rk4_step(f, t, q, dt))
    return np.concatenate((q_new, b))


def propagate_P(P, x, gyro_reading, dt,
                sigma_gyro=0.01, sigma_bias=0.0005):

    b = x[4:]
    omega_hat = gyro_reading - b
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))
    F = np.block([[-skew(omega_hat), -I3],
                  [Z3,               Z3]])

    Phi = np.eye(6) + F * dt

    Q = np.block([[sigma_gyro**2 * dt**2 * I3, Z3],
                  [Z3, sigma_bias**2 * dt * I3]])

    P_new = Phi @ P @ Phi.T + Q
    return 0.5 * (P_new + P_new.T)

def update(x, P, accel, sigma_accel=0.02):
    q = x[:4]
    b = x[4:]

    g_world = np.array([0, 0, -1.0])
    a_hat = q_rotate_vector(q_conjugate(q), g_world)
    y = accel - a_hat

    H = np.block([[skew(a_hat), np.zeros((3, 3))]])
    R = sigma_accel**2 * np.eye(3)

    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    dx = K @ y
    dtheta = dx[:3]
    db = dx[3:]

    q_new = q_normalize(qmul(q, np.concatenate(([1.0], 0.5 * dtheta))))
    b_new = b + db

    P_new = (np.eye(6) - K @ H) @ P
    P_new = 0.5 * (P_new + P_new.T)

    return np.concatenate((q_new, b_new)), P_new

def q_angle_deg(qa, qb):
    d = qmul(q_conjugate(qa), qb)
    return np.degrees(2 * np.arccos(min(1.0, np.abs(d[0]))))

def run_filter(gyro, accel, dt, x0, P0, sigma_accel=0.02):
    n = len(gyro)
    xs = np.zeros((n, 7))
    Ps = np.zeros((n, 6, 6))
    x = x0.copy()
    P = P0.copy()
    xs[0] = x
    Ps[0] = P
    for k in range(n - 1):
        x_next = predict(x, gyro[k], dt)
        P = propagate_P(P, x, gyro[k], dt)
        x = x_next
        x, P = update(x, P, accel[k + 1], sigma_accel)
        xs[k + 1] = x
        Ps[k + 1] = P
    return xs, Ps


def attitude_errors(qs_hat, qs_true):
    n= len(qs_true)
    tilt = np.zeros(n)
    yaw = np. zeros(n)
    for i in range(n):
        d = qmul(q_conjugate(qs_hat[i]), qs_true[i])
        if d[0] < 0:
            d = -d
        v = d[1:]
        s = np.linalg.norm(v)
        if s < 1e-12:
            continue
        e_body = 2 * np.arctan2(s, d[0]) * v / s
        e_world = q_to_dcm(qs_hat[i]) @ e_body
        tilt[i] = np.degrees(np.linalg.norm(e_world[:2]))
        yaw[i] = np.degrees(abs(e_world[2]))
    return tilt, yaw
