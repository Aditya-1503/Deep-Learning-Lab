import torch

x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)

a = 2 * x
b = torch.sin(y)
c = a / b if b != 0 else torch.tensor(0.0, requires_grad=True)
d = c * z
e = torch.log(1 + d) if (1 + d) > 0 else torch.tensor(0.0, requires_grad=True)
f = torch.tanh(e)

print(f"Intermediate values:\n a = {a}, \n b = {b},\n c = {c},\n d = {d},\n e = {e}")
f.backward()
print(y.grad) #df/dy
