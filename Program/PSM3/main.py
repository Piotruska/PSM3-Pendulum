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

