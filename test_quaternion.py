import numpy as np
from quaternion import q_from_axis_angle, q_conjugate, q_normalize, q_rotate_vector, qmul, q_to_dcm, skew, q_dot, rk4_step, simulate_truth, omega_true, simulate_sensors, predict, propagate_P


result = q_from_axis_angle([0, 0, 1], 90)
expected = np.array([0.7071, 0, 0, 0.7071])
print("Test 1:", result, np.allclose(result, expected, atol= 1e-4))


result = q_from_axis_angle([0, 0, 1], 0)
expected = np.array([1, 0, 0, 0])
print("Test 2:", result, np.allclose(result, expected, atol=1e-4))

result = q_from_axis_angle([0, 0, 5], 90)
expected = np.array([0.7071, 0, 0, 0.7071])
print("Test 3:", result, np.allclose(result, expected, atol=1e-4))

result = q_from_axis_angle([1, 0, 0], 180)
expected = np.array([0, 1, 0, 0])
print("Test 4:", result, np.allclose(result, expected, atol=1e-4))

result = q_conjugate(q_from_axis_angle([0, 0, 1], 90))
expected = np.array([0.7071, 0, 0, -0.7071])
print("Test 5:", result, np.allclose(result, expected, atol=1e-4))

result = q_normalize([2, 0, 0, 0])
expected = np.array([1, 0, 0, 0])
print("Test 6:", result, np.allclose(result, expected, atol=1e-4))

result = qmul(q_from_axis_angle([0,0,1], 90), q_from_axis_angle([1,0,0], 90))
expected = np.array([0.5, 0.5, 0.5, 0.5])
print("Test 7:", result, np.allclose(result, expected, atol=1e-4))

result = q_rotate_vector(q_from_axis_angle([0, 0, 1], 30), [1, 0, 0])
expected = np.array([0.8660254, 0.5, 0])
print("Test 8:", result, np.allclose(result, expected, atol=1e-4))

result = q_to_dcm(q_from_axis_angle([0,0,1], 30))
expected = np.array([[0.8660254, -0.5, 0], [0.5, 0.8660254, 0], [0, 0, 1]])
print("Test 9:", np.allclose(result, expected, atol=1e-4))

q = q_from_axis_angle([1, 2, 3], 47)
v = np.array([0.3, -0.7, 1.1])
print("Test 10:", np.allclose(q_rotate_vector(q, v), q_to_dcm(q) @ v))

result = skew([1, 2, 3])@ [1,2,3]
print("Test 11:", np.allclose(result, [0,0,0]))

q = q_dot([1,0,0,0], [0,0,0])
print("Test 12:", np.allclose(q, [0,0,0,0]))

q = q_dot([1,0,0,0], np.radians([0,0,10]))
print("Test 13:", np.allclose(q, 0.5*np.radians(10)))

f = lambda t, s: np.array([2.0, -1.0])
result = rk4_step(f, 0, np.array([0.0, 0.0]), 0.5)
print("Test 14:", np.allclose(result, [1.0, -0.5]))


om = lambda t: np.array([0.35*np.sin(0.4*t), 0.25, 0.20*np.cos(0.3*t)])
ts, qs, ws = simulate_truth(om)
print("Test 19:", ts.shape == (2001,) and qs.shape == (2001,4) and ws.shape == (2001,3))

print("Test 20:", np.allclose(qs[0], [1,0,0,0]))

print("Test 21:", np.abs(np.linalg.norm(qs, axis=1) - 1).max() < 1e-12)

ts, qs, ws = simulate_truth(lambda t: np.zeros(3), T=5.0)
print("Test 22:", np.allclose(qs, [1,0,0,0]))

ts, qs, ws = simulate_truth(lambda t: np.array([0, 0, np.radians(10.0)]), T=9.0)
print("Test 23:", np.allclose(qs[-1], q_from_axis_angle([0,0,1], 90)))

ts, qs, ws = simulate_truth(om)
print("Test 24:", np.allclose(qs[1000], [0.0666, 0.2136, 0.8986, 0.3775], atol=1e-4))

om = lambda t: np.array([0.35*np.sin(0.4*t), 0.25, 0.20*np.cos(0.3*t)])
ts, qs, ws = simulate_truth(om)
dt = 0.01
gyro, accel, bias_true = simulate_sensors(qs, ws, dt)


print("Test 25:", gyro.shape == (2001,3) and accel.shape == (2001,3) and bias_true.shape == (2001,3))
print("Test 26:", np.allclose(bias_true[0], [0.02, -0.015, 0.01]))


g2, a2, b2 = simulate_sensors(qs, ws, dt, seed=42)
print("Test 27:", np.allclose(gyro, g2) and np.allclose(accel, a2))
print(gyro, accel, bias_true)

g3, _, _ = simulate_sensors(qs, ws, dt, seed=7)
print("Test 28:", not np.allclose(gyro, g3))

x = np.array([1.0, 0, 0, 0,  0, 0, 0])
out = predict(x, np.radians([0, 0, 10]), 1.0)
print("Test 30:", np.allclose(out[:4], q_from_axis_angle([0, 0, 1], 10)))

b = np.array([0.02, -0.015, 0.01])
x = np.concatenate(([1.0, 0, 0, 0], b))
out = predict(x, b, 1.0)
print("Test 31:", np.allclose(out[:4], [1, 0, 0, 0]), out[:4])

x = np.array([1.0, 0, 0, 0,  0, 0, 0])
out = predict(x, np.radians([30, -50, 70]), 0.5)
n = np.linalg.norm(out[:4])
print("Test 32:", np.isclose(n, 1.0), n)

# Test 40: P stays symmetric after many steps
P = np.diag([1, 2, 3, 4, 5, 6]) * 1e-4
x = np.array([1.0, 0, 0, 0,  0, 0, 0])
for i in range(100):
    P = propagate_P(P, x, np.array([0.3, -0.2, 0.5]), 0.01)
print("Test 40:", np.allclose(P, P.T))

# Test 41: start with no doubt -> one step adds exactly Q
P = propagate_P(np.zeros((6, 6)), x, np.zeros(3), 0.01)
expected = np.diag([0.01**2 * 0.01**2] * 3 + [0.0005**2 * 0.01] * 3)
print("Test 41:", np.allclose(P, expected), P.diagonal())

# Test 42: bias doubt leaks into attitude doubt (no noise)
P0 = np.diag([0, 0, 0, 4e-4, 4e-4, 4e-4])
P = propagate_P(P0, x, np.zeros(3), 0.01, sigma_gyro=0, sigma_bias=0)
print("Test 42:", np.isclose(P[0, 0], 4e-8) and np.isclose(P[0, 3], -4e-6),
      P[0, 0], P[0, 3])