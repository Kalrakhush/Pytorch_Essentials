import torch
import torch.nn as nn
import numpy as np

#cross entropy loss function
def cross_entropy(actual, predicted):
    loss=-np.sum(actual*np.log(predicted))
    return loss

#cross entropy loss function using pytorch
#nn.cross_entropy_loss already applies softmax to the output layer and nn.NLLLoss expects log probabilities as input
# so we don't need to apply softmax to the output layer
def cross_entropy_torch(actual, predicted):
    loss = nn.CrossEntropyLoss()
    return loss(actual, predicted)

y=np.array([1, 0, 0])
y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])
l1= cross_entropy(y, y_pred_bad)
l2= cross_entropy(y, y_pred_good)
print(l1, l2)

y=torch.tensor([0])
#n_samples x n_classes =1x3
y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

l1=cross_entropy_torch(y_pred_good, y)
l2=cross_entropy_torch(y_pred_bad, y)

print(l1, l2)

_, prediction1= torch.max(y_pred_good, 1)
_, prediction2= torch.max(y_pred_bad, 1)
print(prediction1, prediction2)
