import numpy as np
import matplotlib.pyplot as plt

# Constants
rho = 7780  # Density in kg/m³
Cp = 500  # Specific Heat in J/(kg·K)
eta = 0.75  # Laser Absorption Coefficient
hc = 20  # Convection Coefficient in W/(m²·K)
epsilon = 0.85  # Radiation Coefficient
sigma = 5.67e-8  # Stefan-Boltzmann Constant in W/(m²·K⁴)
T0 = 293.15  # Initial Temperature in K

# Thermal Conductivity as a quadratic function of temperature
def thermal_conductivity(T):
    A = 0.1
    B = 0.02
    C = 15
    return A * T**2 + B * T + C

# Boundary conditions
def heat_loss_convection(T, Ta):
    return hc * (T - Ta)

def heat_loss_radiation(T, Ta):
    return epsilon * sigma * (T**4 - Ta**4)

# Simulation parameters
t1 = 2  # Deposition time in seconds
t2 = 100  # Cooling time in seconds
L = 0.04  # Substrate length in meters
W = 0.02  # Substrate width in meters
H = 0.005  # Substrate height in meters
P = 2000  # Laser Power in watts
v = 0.008  # Scanning speed in meters/second
Ra, Rb, Rc = 0.003, 0.003, 0.001  # Laser spot radius in meters

# Initial temperature distribution
T = np.full((int(L*1000), int(W*1000), int(H*1000)), T0)

# Time steps
dt = 0.01
time_steps = int((t1 + t2) / dt)

# Simulation loop
for step in range(time_steps):
    current_time = step * dt
    if current_time <= t1:
        # Laser is on
        Q_laser = eta * P / (np.pi * Ra * Rb)
    else:
        # Laser is off
        Q_laser = 0
    
    # Update temperature field
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            for k in range(T.shape[2]):
                T[i, j, k] += dt / (rho * Cp) * (
                    thermal_conductivity(T[i, j, k]) * (
                        (T[i+1, j, k] - 2*T[i, j, k] + T[i-1, j, k]) +
                        (T[i, j+1, k] - 2*T[i, j, k] + T[i, j-1, k]) +
                        (T[i, j, k+1] - 2*T[i, j, k] + T[i, j, k-1])
                    ) - Q_laser - heat_loss_convection(T[i, j, k], T0) - heat_loss_radiation(T[i, j, k], T0)
                )

# Plot the final temperature distribution
plt.imshow(T[:, :, int(H*1000/2)])
plt.colorbar()
plt.title('Temperature Distribution')
plt.show()
