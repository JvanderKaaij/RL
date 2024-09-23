import torch
import numpy as np
import matplotlib.pyplot as plt

# Define the mean and covariance matrix
mean = torch.tensor([0.0, 0.0])
cov = torch.tensor([[1.0, 0.5], [0.5, 1.0]])

# Create a PyTorch Multivariate Normal distribution
dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
# Generate a grid of points (x, y)
x, y = np.mgrid[-3:3:.01, -3:3:.01]
xy_grid = np.stack([x, y], axis=-1)  # Shape should be (N, N, 2)

# Convert grid to PyTorch tensor for evaluation
xy_grid_torch = torch.tensor(xy_grid, dtype=torch.float32)

# Compute the probability density function (PDF) for each point on the grid
pdf = torch.exp(dist.log_prob(xy_grid_torch)).numpy()

# Create a contour plot
plt.contourf(x, y, pdf, cmap='Blues')
plt.title('Bivariate Normal Distribution Contour Plot (PyTorch)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.colorbar(label='Density')
plt.show()

# Create a PyTorch Multivariate Normal distribution
dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)

# Create a figure for 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Compute the PDF on the grid
pdf = torch.exp(dist.log_prob(xy_grid_torch)).numpy()

# Plot a 3D surface plot
ax.plot_surface(x, y, pdf, cmap='viridis')
ax.set_title('3D Surface Plot of Bivariate Normal Distribution (PyTorch)')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Probability Density')
plt.show()