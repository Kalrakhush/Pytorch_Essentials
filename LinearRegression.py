#1) Design model(input, output size, forwaed pass)
#2) Construct loss and optimizer
#3) Training loop
 
#  - forward pass: compute prediction
#  - backward pass: gradients
#  - update weights using gradient descent


import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
#f =w*x
#f =2*x

#Step 0: Prepare data
X_numpy, y_numpy=datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1) #generate data
X=torch.from_numpy(X_numpy).float() #convert to tensor
y=torch.from_numpy(y_numpy).float() #convert to tensor

y=y.view(y.shape[0],1) #reshape y to 2D tensor
n_samples, n_features=X.shape #number of samples and features
X_test=torch.tensor([[5]], dtype=torch.float32) #input for prediction
n_samples, n_features=X.shape #number of samples and features
print(n_samples, n_features) #4 1
input_size=n_features #input size=1
output_size=1 #output size=1

#Step 1: Design model(input, output size, forwaed pass)
model=nn.Linear(input_size, output_size) #input and output size

#if we want custom linear model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear=nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

model=LinearRegression(input_size, output_size) #input and output size        
print(f"Prediction before training:  {model(X_test).item():.3f}") #expected 10


#Step2 : Construct loss and optimizer
#loss function =MSE 
#MSE=1/n * sum(y_pred - y)^2
learning_rate=0.01
n_iters=100
creterion=nn.MSELoss() #MSE loss function
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate) #Stochastic Gradient Descent


for epoch in range(n_iters):
    #prediction=forward pass
    y_pred=model(X)

    #loss
    l=creterion(y_pred, y)

    #gradient=backward pass
    l.backward() #compute gradients(DL/DW)

    optimizer.step() #update weights using gradient descent
    #zero the gradients after updating weights
    optimizer.zero_grad() #reset gradients to zero

    if epoch % 10 == 0:
        # [w,b]=model.parameters()
        print(f"epoch {epoch+1}, loss={l.item():.4f}")
print(f"Prediction after training:  {model(X_test).item():.3f}") #expected 10


#plot
predicted=model(X).detach().numpy() #detach from graph and convert to numpy array
plt.plot(X_numpy, y_numpy, "ro", label="Original data")
plt.plot(X_numpy, predicted,"b", label="Fitted line")
plt.show()