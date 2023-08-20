import numpy as np
import matplotlib.pyplot as plt
import imageio

# Constants
c = 1  # Speed of light
epsilon = 1  # Permittivity
mu = 1  # Permeability

# Grid parameters
nx, ny = 100, 100  # Number of cells
dx, dy = 1, 1  # Cell size
dt = dx / (2 * c)  # Time step

# Arrays for fields
Ex = np.zeros((nx, ny))
Ey = np.zeros((nx, ny))
Hz = np.zeros((nx, ny))

# Simulation parameters
nt = 1000  # Number of time steps
source_position_x, source_position_y = nx // 2, ny // 2  # Position of the source

# Create a writer object
with imageio.get_writer('2d_fdtd_simulation.gif', mode='I') as writer:
    # Main FDTD loop
    for t in range(nt):
        # Update H field
        for i in range(nx - 1):
            for j in range(ny - 1):
                Hz[i, j] += (Ex[i, j + 1] - Ex[i, j] - Ey[i + 1, j] + Ey[i, j]) * dt / (mu * dx)

        # Gaussian source
        source = np.exp(-0.5 * ((t - 30) / 10) ** 2)
        Hz[source_position_x, source_position_y] += source

        # Update E fields
        for i in range(1, nx):
            for j in range(1, ny):
                Ex[i, j] += (Hz[i, j] - Hz[i, j - 1]) * dt / (epsilon * dy)
                Ey[i, j] -= (Hz[i, j] - Hz[i - 1, j]) * dt / (epsilon * dx)

        # Optional: plot the electric field at each time step
        if t % 10 == 0:
            plt.figure(figsize=(8, 6))
            plt.imshow(np.sqrt(Ex ** 2 + Ey ** 2), cmap='viridis')
            plt.colorbar(label='Electric Field Magnitude', fraction=0.046, pad=0.04)
            plt.title('2D FDTD at t = ' + str(t))
            plt.axis('off') # Remove axes

            # Convert plot to image and append to GIF
            plt.savefig('temp.png', bbox_inches='tight', pad_inches=0)
            writer.append_data(imageio.imread('temp.png'))
            plt.close()
