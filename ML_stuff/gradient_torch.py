#1) Design model(input, output size, forwaed pass)
#2) Construct loss and optimizer
#3) Training loop
 
#  - forward pass: compute prediction
#  - backward pass: gradients
#  - update weights using gradient descent


import numpy as np
import torch
import torch.nn as nn
#f =w*x
#f =2*x

X=torch.tensor([[1], [2], [3],[4]], dtype=torch.float32)
Y=torch.tensor([[2], [4], [6],[8]], dtype=torch.float32)

X_test=torch.tensor([[5]], dtype=torch.float32) #input for prediction
n_samples, n_features=X.shape #number of samples and features
print(n_samples, n_features) #4 1
input_size=n_features #input size=1
output_size=n_features #output size=1
# model=nn.Linear(input_size, output_size) #input and output size

#if we want custom linear model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear=nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

model=LinearRegression(input_size, output_size) #input and output size        
print(f"Prediction before training:  {model(X_test).item():.3f}") #expected 10

#training
learning_rate=0.01
n_iters=100


loss=nn.MSELoss() #MSE loss function
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate) #Stochastic Gradient Descent
for epoch in range(n_iters):
    #prediction=forward pass
    y_pred=model(X)

    #loss
    l=loss(Y, y_pred)

    #gradient=backward pass
    l.backward() #compute gradients(DL/DW)

    optimizer.step() #update weights using gradient descent
    #zero the gradients after updating weights
    optimizer.zero_grad() #reset gradients to zero
    if epoch % 10 == 0:
        [w,b]=model.parameters()
        print(f"epoch {epoch+1}: w={w[0][0].item():.3f}, loss={l:.8f}")
print(f"Prediction after training:  {model(X_test).item():.3f}") #expected 10
