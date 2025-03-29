import numpy as np
import sympy as sp
import scipy.interpolate as interp
from scipy.optimize import root_scalar

def lagrange_interpolation(x_vals, y_vals, x):
    """Computes the Lagrange interpolation polynomial and evaluates it at x."""
    n = len(x_vals)
    lagrange_poly = 0
    for i in range(n):
        term = y_vals[i]
        for j in range(n):
            if i != j:
                term *= (x - x_vals[j]) / (x_vals[i] - x_vals[j])
        lagrange_poly += term
    return lagrange_poly

# Problem 1: Lagrange interpolation for cos(0.750)
x_values = np.array([0.698, 0.733, 0.768, 0.803])
y_values = np.array([0.7661, 0.7432, 0.7193, 0.6946])
x_target = 0.750

for degree in range(1, 5):
    interp_x_vals = x_values[:degree + 1]
    interp_y_vals = y_values[:degree + 1]
    approx = lagrange_interpolation(interp_x_vals, interp_y_vals, x_target)
    error_bound = abs(0.7317 - approx)
    print(f"Degree {degree}: Approximation = {approx:.6f}, Error Bound = {error_bound:.6f}")

# Problem 2: Root finding for x = e^(-x) using root_scalar
def func(x):
    return x - np.exp(-x)

solution = root_scalar(func, bracket=[0, 1], method='brentq')
print(f"Solution to x - e^(-x) = 0: x ≈ {solution.root:.8f}")

# Problem 3: Hermite Interpolation for Car Motion
def hermite_interpolation(t_vals, d_vals, v_vals, t_target):
    """Computes Hermite interpolation and evaluates it at t_target."""
    hermite_poly = interp.CubicHermiteSpline(t_vals, d_vals, v_vals)
    position = hermite_poly(t_target)
    speed = hermite_poly.derivative()(t_target)
    return position, speed

T = np.array([0, 3, 5, 8, 13])
D = np.array([0, 200, 375, 620, 990])
V = np.array([75, 77, 80, 74, 72])

t_prediction = 10
predicted_position, predicted_speed = hermite_interpolation(T, D, V, t_prediction)
print(f"Predicted position at t = {t_prediction}s: {predicted_position:.2f} ft")
print(f"Predicted speed at t = {t_prediction}s: {predicted_speed:.2f} ft/s")

# Maximum speed predicted
max_speed = max(interp.CubicHermiteSpline(T, D, V).derivative()(T))

# Check if speed exceeds 55 mi/h (≈ 80.67 ft/s)
speed_limit = 80.67
if max_speed > speed_limit:
    print("The car exceeded 55 mi/h")
else:
    print("The car did not exceed 55 mi/h")

print(f"Predicted maximum speed: {max_speed:.2f} ft/s")
