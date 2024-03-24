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

#Runge-Kutta 4th method step function for single iteration
def rk4_step(theta, omega, dt):
    k1_theta, k1_omega = D(theta, omega)
    k2_theta, k2_omega = D(theta + 0.5 * dt * k1_theta, omega + 0.5 * dt * k1_omega)
    k3_theta, k3_omega = D(theta + 0.5 * dt * k2_theta, omega + 0.5 * dt * k2_omega)
    k4_theta, k4_omega = D(theta + dt * k3_theta, omega + dt * k3_omega)
    theta_new = theta + (dt / 6) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)
    omega_new = omega + (dt / 6) * (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega)
    return theta_new, omega_new

#Runge-Kutta 5th method step function for single iteration
def rk5_step(theta, omega, dt):
    k1_theta, k1_omega = D(theta, omega)
    k2_theta, k2_omega = D(theta + 0.5 * dt * k1_theta, omega + 0.5 * dt * k1_omega)
    k3_theta, k3_omega = D(theta + 0.5 * dt * k2_theta, omega + 0.5 * dt * k2_omega)
    k4_theta, k4_omega = D(theta + 0.5 * dt * k3_theta, omega + 0.5 * dt * k3_omega)
    k5_theta, k5_omega = D(theta + dt * k4_theta, omega + dt * k4_omega)
    theta_new = theta + (dt / 8) * (k1_theta + 2 * k2_theta + 2 * k3_theta + 2 * k4_theta + k5_theta)
    omega_new = omega + (dt / 8) * (k1_omega + 2 * k2_omega + 2 * k3_omega + 2 * k4_omega + k5_omega)
    return theta_new, omega_new

# Simulate pendulum motion method
def simulate(method):
    theta = [theta0]
    omega = [omega0]
    t = [t0]
    while t[-1] < tmax:
        theta_new, omega_new = method(theta[-1], omega[-1], dt)
        theta.append(theta_new)
        omega.append(omega_new)
        t.append(t[-1] + dt)
    return np.array(theta), np.array(omega), np.array(t)

# Method to calculate KE and PE and TE
def calculate_energy(theta, omega):
    potential_energy = m * g * L * (1 - np.cos(theta))
    kinetic_energy = 0.5 * m * L**2 * omega**2
    total_energy = potential_energy + kinetic_energy
    return potential_energy, kinetic_energy, total_energy

# Plot energies method
def plot_energy(t, pe, ke, te, method):
    plt.figure(figsize=(10, 6))
    plt.plot(t, pe, label='Potential Energy')
    plt.plot(t, ke, label='Kinetic Energy')
    plt.plot(t, te, label='Total Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.title(f'Energy of Pendulum - {method} Method')
    plt.legend()
    plt.show()

# Animation method
def animate_pendulum(theta, title):
    fig, ax = plt.subplots()
    ax.set_xlim((-1.2 * L, 1.2 * L))
    ax.set_ylim((-1.2 * L, 1.2 * L))
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title)
    line, = ax.plot([], [], 'o-', lw=2)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x = L * np.sin(theta[i])
        y = -L * np.cos(theta[i])
        line.set_data([0, x], [0, y])
        return line,

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(theta), interval=20, blit=True)
    plt.show()

    # Perform simulations
    theta_euler, omega_euler, t_euler = simulate(euler_step)
    theta_midpoint, omega_midpoint, t_midpoint = simulate(midpoint_step)
    theta_rk4, omega_rk4, t_rk4 = simulate(rk4_step)
    theta_rk5, omega_rk5, t_rk5 = simulate(rk5_step)

    # Calculate energies
    pe_euler, ke_euler, te_euler = calculate_energy(theta_euler, omega_euler)
    pe_midpoint, ke_midpoint, te_midpoint = calculate_energy(theta_midpoint, omega_midpoint)
    pe_rk4, ke_rk4, te_rk4 = calculate_energy(theta_rk4, omega_rk4)
    pe_rk5, ke_rk5, te_rk5 = calculate_energy(theta_rk5, omega_rk5)

    plot_energy(t_euler, pe_euler, ke_euler, te_euler, 'Euler')
    animate_pendulum(theta_euler, 'Pendulum Motion Animation - Euler Method')

    plot_energy(t_midpoint, pe_midpoint, ke_midpoint, te_midpoint, 'Midpoint')
    animate_pendulum(theta_midpoint, 'Pendulum Motion Animation - Midpoint Method')

    plot_energy(t_rk4, pe_rk4, ke_rk4, te_rk4, 'RK4')
    animate_pendulum(theta_rk4, 'Pendulum Motion Animation - RK4 Method')

    plot_energy(t_rk5, pe_rk5, ke_rk5, te_rk5, 'RK5')
    animate_pendulum(theta_rk5, 'Pendulum Motion Animation - RK5 Method')
