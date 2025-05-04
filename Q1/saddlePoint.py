import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define function
def f(x, y):
    return 4 * x**2 - 4 * y**2

# Line path: y = 0
x_vals = np.linspace(-2, 2, 100)
y_vals = np.zeros_like(x_vals)
z_vals = f(x_vals, y_vals)

# Set up figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(*np.meshgrid(x_vals, x_vals), f(*np.meshgrid(x_vals, x_vals)), alpha=0.3, cmap='coolwarm')
point, = ax.plot([], [], [], 'ko', markersize=5)

# Set axis labels
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-10, 10])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')

# Animation function
def update(frame):
    point.set_data([x_vals[frame]], [y_vals[frame]])
    point.set_3d_properties([z_vals[frame]])
    return point,

ani = FuncAnimation(fig, update, frames=len(x_vals), interval=50, blit=True)
plt.show()
