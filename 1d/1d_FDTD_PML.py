import numpy as np
import matplotlib.pyplot as plt
import imageio

# Constants
c = 1  # Speed of light
epsilon = 1  # Permittivity
mu = 1  # Permeability

# Grid parameters
nx = 400  # Number of cells
dx = 1  # Cell size
dt = dx / (2 * c)  # Time step
bound_width = 50  # PML boundary width
gradingorder = 6  # Order of polynomial grading
refl_coeff = 1e-6  # Reflection coefficient
sigmamax = (-np.log10(refl_coeff) * (gradingorder + 1) * epsilon * c) / (2 * bound_width * dx)

# Arrays for fields
Ex = np.zeros(nx)
Hy = np.zeros(nx)
sigma = np.zeros(nx)
sigma_star = np.zeros(nx)

# PML sigma calculations
boundfact1 = sigmamax / ((bound_width ** gradingorder) * (gradingorder + 1))
boundfact2 = sigmamax / ((bound_width ** gradingorder) * (gradingorder + 1))
x = np.arange(bound_width)
sigma[:bound_width] = boundfact1 * ((x + 0.5) ** (gradingorder + 1) - (x - 0.5) ** (gradingorder + 1))
sigma[-bound_width:] = boundfact2 * ((x[::-1] + 0.5) ** (gradingorder + 1) - (x[::-1] - 0.5) ** (gradingorder + 1))
sigma_star = (sigma * mu) / epsilon

# Multiplication factors
A = (mu - 0.5 * dt * sigma_star) / (mu + 0.5 * dt * sigma_star)
B = (dt / dx) / (mu + 0.5 * dt * sigma_star)
C = (epsilon - 0.5 * dt * sigma) / (epsilon + 0.5 * dt * sigma)
D = (dt / dx) / (epsilon + 0.5 * dt * sigma)

# Simulation parameters
nt = 500  # Number of time steps
source_position = nx // 2  # Position of the source

# Create a writer object
with imageio.get_writer('1d_fdtd_pml_simulation.gif', mode='I') as writer:
    # Main FDTD loop
    for t in range(nt):
        # Update magnetic field with PML
        Hy[:-1] = A[:-1] * Hy[:-1] + B[:-1] * (Ex[1:] - Ex[:-1])

        # Gaussian source
        source_amplitude = 0.3
        source = source_amplitude * np.exp(-0.5 * ((t - 30) / 10) ** 2)
        Hy[source_position] += source

        # Update electric field with PML
        Ex[1:] = C[1:] * Ex[1:] + D[1:] * (Hy[1:] - Hy[:-1])

        # Plot the electric field at each time step
        if t % 10 == 0:
            plt.plot(Ex)
            plt.xlabel('Grid Index')
            plt.ylabel('Electric Field (Ex)')
            plt.title('1D FDTD with PML at t = ' + str(t))
            plt.ylim([-1.2, 1.2])
            
            # Convert plot to image and append to GIF
            plt.savefig('temp.png')
            writer.append_data(imageio.imread('temp.png'))
            plt.close()