import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Base Values
g = 9.81  # Acceleration due to gravity (m/s^2)
L = 1     # Length of the pendulum (m)
m = 1     # Mass of the pendulum (kg)

# Initial conditions
theta0 = np.pi / 3  # Initial angle (radians)
omega0 = 0          # Initial angular velocity (radians/second)
t0 = 0              # Initial time (s)
tmax = 100         # Maximum time (s)
dt = 0.01           # Time step (s)

# Define differential equation method
def D(theta, omega):
    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta)
    return dtheta_dt, domega_dt


#Euler's method step function for single iteration
def euler_step(theta, omega, dt):
    dtheta_dt, domega_dt = D(theta, omega)
    theta_new = theta + dt * dtheta_dt
    omega_new = omega + dt * domega_dt
    return theta_new, omega_new


#Improved Eulers method step function for single iteration
def midpoint_step(theta, omega, dt):
    k1_theta, k1_omega = D(theta, omega)
    theta_half = theta + 0.5 * dt * k1_theta
    omega_half = omega + 0.5 * dt * k1_omega
    k2_theta, k2_omega = D(theta_half, omega_half)
    theta_new = theta + dt * k2_theta
    omega_new = omega + dt * k2_omega
    return theta_new, omega_new
