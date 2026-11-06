import numpy as np
from quaternion import q_from_axis_angle, q_conjugate, q_normalize, q_rotate_vector, qmul, q_to_dcm, skew, q_dot


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
print("Test 13:", np.allclose(q, [0,0,0,10]))
