'''
epoch= 1 forward and backward pass of all training examples
batch_size= number of training examples in one forward/backward pass
number of iterations= number of passes, each pass using batch_size examples
e.g 1000 training examples, batch_size=100, number of iterations=10000/100=100'''


import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import math

class WineDataset(torch.utils.data.Dataset):
    def __init__(self):
        xy=np.loadtxt("./data/wine/wine.csv", delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    
    
dataset=WineDataset()    
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)

dataloader=DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)
# dataiter=iter(dataloader)
# first_batch=next(dataiter)
# features, labels=first_batch
# print(features, labels)

#training_loop
num_epochs=2
total_samples=len(dataset)
n_iterations=math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward pass
        # backward pass
        # update weights
        if (i+1)%n_iterations==0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')
# torchvision.datasets.MNIST()