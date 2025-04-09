import torch
import torch.nn as nn
import numpy as np


#softmax(x) = exp(x) / sum(exp(x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x=np.array([2.0, 1.0, 0.1])
output = softmax(x)
print("softmax numpy:",output)

x=torch.tensor([2.0, 1.0, 0.1])
output = torch.softmax(x, dim=0)
print("softmax torch:",output)