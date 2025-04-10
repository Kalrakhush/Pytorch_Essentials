#transfer learning: Models trained on one task can be reused on another task

#transfer learning is a technique where a model trained on one task is reused on another task. It is commonly used in deep learning, especially in computer vision and natural language processing. The idea is to take a pre-trained model and fine-tune it on a new dataset, which can save time and resources compared to training a model from scratch.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import time
import copy
from torch.optim import lr_scheduler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop((224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize((256)),
        transforms.CenterCrop((224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

# Load the dataset
data_dir = 'data/hymenoptera_data'
image_datasets = {x: ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}   
class_names = image_datasets['train'].classes
print(class_names)

def train_model(model, creterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()    

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = creterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]     

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    #load best model weights
    model.load_state_dict(best_model_wts)
    return model


#finetuned weights using last layer

#also we can freeze the weights of the model and only train the last layer
#to freeze the weights of the model, we can set requires_grad = False for all the parameters of the model
model= models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
#this will freeze the weights of the model and only train the last layer

#load a pre-trained model
num_ftrs = model.fc.in_features #get the number of features in the last layer
model.fc = nn.Linear(num_ftrs, 2) #replace the last layer with a new one
model = model.to(device)

creterion=nn.CrossEntropyLoss
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#scheluder to adjust the learning rate

step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#step_size is the number of epochs before the learning rate is reduced by a factor of gamma
#gamma is the factor by which the learning rate is reduced

model = train_model(model, creterion, optimizer, step_lr_scheduler, num_epochs=20)


