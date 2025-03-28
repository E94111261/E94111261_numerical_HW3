import numpy as np
import sympy as sp
import scipy.interpolate as interp

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
    error_bound = abs(np.cos(x_target) - approx)
    print(f"Degree {degree}: Approximation = {approx:.6f}, Error Bound = {error_bound:.6f}")

# Problem 2: Iterated Inverse Interpolation
def inverse_interpolation(x_vals, y_vals, y_target, iterations=3):
    """Uses inverse interpolation iteratively to estimate the root."""
    x_estimate = np.interp(y_target, y_vals, x_vals)
    for _ in range(iterations):
        f_interp = interp.BarycentricInterpolator(y_vals, x_vals)
        x_estimate = f_interp(y_target)
    return x_estimate

x_vals_exp = np.array([0.3, 0.4, 0.5, 0.6])
y_vals_exp = np.exp(-x_vals_exp)
y_target_exp = 0
solution = inverse_interpolation(x_vals_exp, y_vals_exp, y_target_exp)
print(f"Solution to x - e^(-x) = 0: x ≈ {solution:.6f}")

# Problem 3: Hermite Interpolation for Car Motion
def hermite_interpolation(t_vals, d_vals, v_vals, t_target):
    """Computes Hermite interpolation and evaluates it at t_target."""
    hermite_poly = interp.PchipInterpolator(t_vals, d_vals)
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

# Check if speed exceeds 55 mi/h (≈ 80.67 ft/s)
def find_exceeding_time(t_vals, d_vals, v_vals, speed_limit=80.67):
    hermite_poly = interp.PchipInterpolator(t_vals, d_vals)
    speed_func = hermite_poly.derivative()
    for t in np.linspace(t_vals[0], t_vals[-1], 100):
        if speed_func(t) > speed_limit:
            return t
    return None

exceed_time = find_exceeding_time(T, D, V)
if exceed_time:
    print(f"The car first exceeds 55 mi/h at t ≈ {exceed_time:.2f} s")
else:
    print("The car does not exceed 55 mi/h.")

# Maximum speed predicted
max_speed = max(interp.PchipInterpolator(T, D).derivative()(T))
print(f"Predicted maximum speed: {max_speed:.2f} ft/s")
