import torch
import torchvision
import numpy as np

#inbuilt
dataset= torchvision.datasets.MNIST(root='./data', transform=torchvision.transforms.ToTensor(), download=True) #converts numpy to tensors


class WineDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        xy=np.loadtxt("./data/wine/wine.csv", delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        self.x= xy[:, 1:]
        self.y= xy[:, 0]
        self.transform= transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample= self.x[idx], self.y[idx]

        if self.transform:
            sample= self.transform(sample)

        return sample    
    
#custom transform class (if we make)
class ToTensor:
    def __call__(self, sample):
        inputs, targets =sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    

class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        inputs = inputs * self.factor
        return inputs, targets
    
     