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

# Arrays for fields
Ex = np.zeros(nx)
Hy = np.zeros(nx)

# Simulation parameters
nt = 2000  # Number of time steps
source_position = nx // 2  # Position of the source

# Create a writer object
with imageio.get_writer('1d_fdtd_simulation.gif', mode='I') as writer:
    # Main FDTD loop
    for t in range(nt):
        # Update magnetic field
        Hy[:nx - 1] += (Ex[1:nx] - Ex[:nx - 1]) * dt / (mu * dx)

        # Gaussian source
        source = np.exp(-0.5 * ((t - 30) / 10) ** 2)
        Hy[source_position] += source

        # Update electric field
        Ex[1:nx] += (Hy[1:nx] - Hy[:nx - 1]) * dt / (epsilon * dx)

        # As same as the following code

        # Update magnetic field
        # for i in range(nx - 1):
        #     Hy[i] += (Ex[i + 1] - Ex[i]) * dt / (mu * dx)

        # Gaussian source
        # source = np.exp(-0.5 * ((t - 30) / 10) ** 2)
        # Hy[source_position] += source

        # Update electric field
        for i in range(1, nx):
            Ex[i] += (Hy[i] - Hy[i - 1]) * dt / (epsilon * dx)

        # Plot the electric field at each time step
        if t % 10 == 0:
            plt.plot(Ex)
            plt.xlabel('Grid Index')
            plt.ylabel('Electric Field (Ex)')
            plt.title('1D FDTD at t = ' + str(t))
            plt.ylim([-1.2, 1.2])
            
            # Convert plot to image and append to GIF
            plt.savefig('temp.png')
            writer.append_data(imageio.imread('temp.png'))
            plt.close()
