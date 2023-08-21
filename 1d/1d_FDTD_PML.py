import numpy as np
import matplotlib.pyplot as plt
import imageio

# Constants
c = 3e+8
mu0 = 4 * np.pi * 1e-7
epsilon0 = (1 / (36 * np.pi)) * 1e-9
S = 1 / (2**0.5)
delta = 1e-6
deltat = S * delta / c

# Grid parameters
xdim = 500
nt = 500

# Arrays for fields
Ez = np.zeros(xdim)
Hy = np.zeros(xdim)

# Simulation parameters
epsilon = epsilon0 * np.ones(xdim)
mu = mu0 * np.ones(xdim)

# PML parameters
sigma = np.zeros(xdim)
bound_width = 25
gradingorder = 6
refl_coeff = 1e-6
sigmamax = (-np.log10(refl_coeff) * (gradingorder + 1) * epsilon0 * c) / (2 * bound_width * delta)

for i in range(bound_width):
    sigma_value = sigmamax * ((bound_width - i - 0.5) ** gradingorder) / bound_width ** gradingorder
    sigma[i] = sigma_value
    sigma[-i - 1] = sigma_value

sigma_star = (sigma * mu) / epsilon
A = (epsilon - 0.5 * deltat * sigma) / (epsilon + 0.5 * deltat * sigma)
B = (deltat / delta) / (epsilon + 0.5 * deltat * sigma)
C = (mu - 0.5 * deltat * sigma_star) / (mu + 0.5 * deltat * sigma_star)
D = (deltat / delta) / (mu + 0.5 * deltat * sigma_star)

# Source position
source_position = xdim // 2

with imageio.get_writer('1d_fdtd_pml_simulation.gif', mode='I') as writer:
    for t in range(nt):
        # Gaussian source
        source = np.exp(-0.5 * ((t - 30) / 10) ** 2)
        Ez[source_position] += source

        # Update magnetic field
        Hy[:xdim - 1] = C[:xdim - 1] * Hy[:xdim - 1] + D[:xdim - 1] * (Ez[1:xdim] - Ez[:xdim - 1])

        # Update electric field
        Ez[1:xdim] = A[1:xdim] * Ez[1:xdim] + B[1:xdim] * (Hy[1:xdim] - Hy[:xdim - 1])

        # Plot the electric field at each time step
        if t % 5 == 0:
            plt.plot(Ez)
            plt.ylim(-1.5, 1.5)
            plt.title('1D FDTD with PML at t = ' + str(t))
            plt.xlabel('Grid Point')
            plt.ylabel('Electric Field')

            # Convert plot to image and append to GIF
            plt.savefig('temp.png', bbox_inches='tight', pad_inches=0)
            writer.append_data(imageio.imread('temp.png'))
            plt.close()
