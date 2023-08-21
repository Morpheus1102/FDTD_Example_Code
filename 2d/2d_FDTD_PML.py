import numpy as np
import matplotlib.pyplot as plt
import math
import imageio

# Constants
c = 3e+8
mu0 = 4 * np.pi * 1e-7
epsilon0 = (1 / (36 * np.pi)) * 1e-9
S = 1 / (2**0.5)
delta = 1e-6
deltat = S * delta / c

# Grid parameters
xdim = 150
ydim = 150
nt = 500
S = 1 / (2**0.5)

# Arrays for fields
Ez = np.zeros((xdim, ydim))
Ezx = np.zeros((xdim, ydim))
Ezy = np.zeros((xdim, ydim))
Hy = np.zeros((xdim, ydim))
Hx = np.zeros((xdim, ydim))

# Simulation parameters
epsilon = epsilon0 * np.ones((xdim, ydim))
mu = mu0 * np.ones((xdim, ydim))

# PML parameters
sigmax = np.zeros((xdim, ydim))
sigmay = np.zeros((xdim, ydim))
bound_width = 25
gradingorder = 6
refl_coeff = 1e-6

# PML sigma calculations
sigmamax = (-math.log10(refl_coeff) * (gradingorder + 1) * epsilon0 * c) / (2 * bound_width * delta)
boundfact1 = ((epsilon[int(xdim/2), bound_width]/epsilon0) * sigmamax) / ((bound_width**gradingorder) * (gradingorder + 1))
boundfact2 = ((epsilon[int(xdim/2), ydim-bound_width]/epsilon0) * sigmamax) / ((bound_width**gradingorder) * (gradingorder + 1))
boundfact3 = ((epsilon[bound_width, int(ydim/2)]/epsilon0) * sigmamax) / ((bound_width**gradingorder) * (gradingorder + 1))
boundfact4 = ((epsilon[xdim-bound_width, int(ydim/2)]/epsilon0) * sigmamax) / ((bound_width**gradingorder) * (gradingorder + 1))
x = np.arange(bound_width + 1)

for i in range(xdim):
    sigmax[i, bound_width::-1] = boundfact1 * ((x + 0.5)**(gradingorder + 1) - (x - 0.5)**(gradingorder + 1))
    sigmax[i, ydim-bound_width-1:ydim] = boundfact2 * ((x + 0.5)**(gradingorder + 1) - (x - 0.5)**(gradingorder + 1))

for i in range(ydim):
    sigmay[bound_width::-1, i] = boundfact3 * ((x + 0.5)**(gradingorder + 1) - (x - 0.5)**(gradingorder + 1))
    sigmay[xdim-bound_width-1:xdim, i] = boundfact4 * ((x + 0.5)**(gradingorder + 1) - (x - 0.5)**(gradingorder + 1))

sigma_starx = (sigmax * mu) / epsilon
sigma_stary = (sigmay * mu) / epsilon

G = (mu - 0.5 * deltat * sigma_starx) / (mu + 0.5 * deltat * sigma_starx)
H = (deltat / delta) / (mu + 0.5 * deltat * sigma_starx)
A = (mu - 0.5 * deltat * sigma_stary) / (mu + 0.5 * deltat * sigma_stary)
B = (deltat / delta) / (mu + 0.5 * deltat * sigma_stary)
C = (epsilon - 0.5 * deltat * sigmax) / (epsilon + 0.5 * deltat * sigmax)
D = (deltat / delta) / (epsilon + 0.5 * deltat * sigmax)
E = (epsilon - 0.5 * deltat * sigmay) / (epsilon + 0.5 * deltat * sigmay)
F = (deltat / delta) / (epsilon + 0.5 * deltat * sigmay)

# Simulation parameters
source_position_x, source_position_y = xdim // 2 , ydim // 2 # Position of the source

with imageio.get_writer('2d_fdtd_pml_simulation.gif', mode='I') as writer:
    for t in range(nt):

        n1 = max(source_position_x - t - 1, 0)
        n2 = min(source_position_x + t, xdim - 1)
        n11 = max(source_position_y - t - 1, 0)
        n21 = min(source_position_y + t, ydim - 1)

        Hy[n1:n2, n11:n21] = A[n1:n2, n11:n21] * Hy[n1:n2, n11:n21] + B[n1:n2, n11:n21] * (Ezx[n1+1:n2+1, n11:n21] - Ezx[n1:n2, n11:n21] + Ezy[n1+1:n2+1, n11:n21] - Ezy[n1:n2, n11:n21])
        Hx[n1:n2, n11:n21] = G[n1:n2, n11:n21] * Hx[n1:n2, n11:n21] - H[n1:n2, n11:n21] * (Ezx[n1:n2, n11+1:n21+1] - Ezx[n1:n2, n11:n21] + Ezy[n1:n2, n11+1:n21+1] - Ezy[n1:n2, n11:n21])

        # Gaussian source
        source = 3 * np.exp(-0.5 * ((t - 30) / 10) ** 2)
        Ezx[source_position_x, source_position_y] += source
        Ezy[source_position_x, source_position_y] += source

        Ezx[n1+1:n2+1, n11+1:n21+1] = C[n1+1:n2+1, n11+1:n21+1] * Ezx[n1+1:n2+1, n11+1:n21+1] + D[n1+1:n2+1, n11+1:n21+1] * (-Hx[n1+1:n2+1, n11+1:n21+1] + Hx[n1+1:n2+1, n11:n21])
        Ezy[n1+1:n2+1, n11+1:n21+1] = E[n1+1:n2+1, n11+1:n21+1] * Ezy[n1+1:n2+1, n11+1:n21+1] + F[n1+1:n2+1, n11+1:n21+1] * (Hy[n1+1:n2+1, n11+1:n21+1] - Hy[n1:n2, n11+1:n21+1])

        Ez[n1:n2+1, n11:n21+1] = (Ezx[n1:n2+1, n11:n21+1] + Ezy[n1:n2+1, n11:n21+1]) / 2

        # Plot the electric field at each time step
        if t % 5 == 0:
            plt.figure(figsize=(8, 6))
            plt.imshow(Ez.T, cmap='viridis', vmin=0, vmax=0.03)
            plt.colorbar(label='Electric Field Magnitude', fraction=0.046, pad=0.04)
            plt.title('2D FDTD with PML at t = ' + str(t))
            plt.axis('off')

            # Convert plot to image and append to GIF
            plt.savefig('temp.png', bbox_inches='tight', pad_inches=0)
            writer.append_data(imageio.imread('temp.png'))
            plt.close()
