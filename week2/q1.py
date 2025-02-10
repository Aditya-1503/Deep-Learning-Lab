import torch

import torch

# Define a and b as torch tensors with requires_grad=True
a = torch.tensor(1.0, requires_grad=True)  # You can change the value of a for testing
b = torch.tensor(2.0, requires_grad=True)  # You can change the value of b for testing

# Compute x, y, and z
x = 2 * a + 3 * b
y = 5 * a**2 + 3 * b**3
z = 2 * x + 3 * y

# Compute the gradient of z with respect to a
z.backward()  # This automatically computes the gradients

# Print the gradient dz/da
print(f"The gradient dz/da is: {a.grad.item()}")
print(f"The gradient dz/db is: {b.grad.item()}")
