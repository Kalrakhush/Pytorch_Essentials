#1) Design model(input, output size, forwaed pass)
#2) Construct loss and optimizer
#3) Training loop
 
#  - forward pass: compute prediction
#  - backward pass: gradients
#  - update weights using gradient descent


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
#f =w*x
#f =2*x

#Step 0: Prepare data
bc=datasets.load_breast_cancer()
X=bc.data #input data
y=bc.target #output data

n_samples, n_features=X.shape #number of samples and features

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=1234) #split data into train and test set

#scale
X_train=StandardScaler().fit_transform(X_train) #scale data to 0 mean and 1 std
X_test=StandardScaler().fit_transform(X_test) #scale data to 0 mean and 1 std

X_train=torch.from_numpy(X_train.astype(np.float32)) #convert to tensor
X_test=torch.from_numpy(X_test.astype(np.float32)) #convert to tensor
y_train=torch.from_numpy(y_train.astype(np.float32)) #convert to tensor
y_test=torch.from_numpy(y_test.astype(np.float32)) #convert to tensor


y_train=y_train.view(y_train.shape[0],1) #reshape y to 2D tensor
y_test=y_test.view(y_test.shape[0],1) #reshape y to 2D tensor

#Step 1: Design model(input, output size, forwaed pass)e

class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear=nn.Linear(n_input_features, 1) #input and output size

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model=LogisticRegression(n_features) #input and output size        



#Step2 : Construct loss and optimizer
#loss function =MSE 
#MSE=1/n * sum(y_pred - y)^2
learning_rate=0.01
n_iters=100
creterion=nn.BCELoss() #MSE loss function
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate) #Stochastic Gradient Descent

#step 3: Training loop
for epoch in range(n_iters):
    #prediction=forward pass
    y_pred=model(X_train)

    #loss
    l=creterion(y_pred, y_train)

    #gradient=backward pass
    l.backward() #compute gradients(DL/DW)

    optimizer.step() #update weights using gradient descent
    #zero the gradients after updating weights
    optimizer.zero_grad() #reset gradients to zero

    if epoch % 10 == 0:
        # [w,b]=model.parameters()
        print(f"epoch {epoch+1}, loss={l.item():.4f}")


#plot
with torch.no_grad():
    y_pred=model(X_test)
    y_pred_cls=y_pred.round() #round to 0 or 1
    acc=y_pred_cls.eq(y_test).sum()/float(y_test.shape[0]) #accuracy
    print(f"Accuracy: {acc:.4f}")

#     Use torch.no_grad() whenever you are:

# Evaluating the model on test/validation data.
# Visualizing predictions.
# Performing any operation where gradients are not required.
# This is a common practice to optimize performance during inference.