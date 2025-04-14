#MNIST
#Dataloader
#Multilayer Neural Net, activation function
#Loss and optimizer
#Training loop (batchtraining)
#Model evaluation
#Gpu support

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn.functional as F
import os
from pytorch_lightning import Trainer




#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
input_size = 784 #28*28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

#now we want to classify digits
class LitNeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LitNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28*28)
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        return optimizer
    
    def train_dataloader(self):
        train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           persistent_workers=True,
                                           num_workers=15,
                                           shuffle=True)
        return train_loader
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28*28)
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        return loss
    
    def val_dataloader(self):
        val_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=False,
                                           transform=transforms.ToTensor())
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           persistent_workers=True,
                                           num_workers=15,
                                           shuffle=False)
        return val_loader

    # def on_validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     return {'val_loss': avg_loss}
    
if __name__ == '__main__':
    trainer = Trainer(max_epochs=num_epochs)  # Set fast_dev_run=True for quick testing
    
    model = LitNeuralNet(input_size, hidden_size, num_classes).to(device)

    trainer.fit(model)  # Train the model
    trainer.test()  # Test the model