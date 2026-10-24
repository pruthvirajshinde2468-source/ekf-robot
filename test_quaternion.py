import numpy as np
from quaternion import q_from_axis_angle, q_conjugate, q_normalize

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

