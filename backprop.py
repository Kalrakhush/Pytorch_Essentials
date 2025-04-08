import torch

x=torch.tensor(1.0)
y=torch.tensor(2.0)

w=torch.tensor(1.0,requires_grad=True)

#forward pass and compute the loss
y_hat=w*x
loss=(y_hat-y)**2

print(f"y_hat: {y_hat}, loss: {loss}")
print(loss)

#backward pass
loss.backward()
print(f"Gradient: {w.grad}")

# y_hat: 1.0, loss: 1.0
# tensor(1., grad_fn=<PowBackward0>)
# Gradient: -2.0