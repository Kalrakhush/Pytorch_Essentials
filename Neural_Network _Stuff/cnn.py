#CNN
#CNN layer-> Relu activation-> Maxpooling-> Flatten-> Dense layer-> Softmax activation

#Filters -applying filter kernel
# we position out image on the filter and take the dot product of the filter and the image
#then we slide it to next position and repeat the process
#the output of this is called feature map
#the feature map is then passed through a non-linear activation function like ReLU  
#this is done to introduce non-linearity in the model

#Pooling
#Pooling is used to reduce the size of the feature map and to make it more manageable
#it also helps to reduce overfitting by reducing the number of parameters in the model
#Max pooling is the most common type of pooling 
#it takes the maximum value from a group of pixels in the feature map
#stride is the number of pixels we move the filter each time
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs=10
batch_size=4
learning_rate=0.001

#dataset has PIL Image of range(0,1)

#we transform them to Tensors of normalised range (-1,,1)

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#show some images
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1=nn.Conv2d(3, 6, 5) #input channels, output channels, kernel size
        self.pool=nn.MaxPool2d(2, 2) #kernel size, stride
        self.conv2=nn.Conv2d(6, 16, 5)
        self.fc1=nn.Linear(16*5*5, 120)
        self.fc2=nn.Linear(120, 84)
        self.fc3=nn.Linear(84, 10)


    def forward(self, x):
        x=self.pool(F.relu(self.conv1(x))) #conv1-> relu->pooling
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1, 16*5*5) #flattening the output of conv2
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

model=ConvNet().to(device)

#loss and optimizer
creterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)

#train the model
n_total_step=len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #origin shape: [4,3,32,32]= 4,3, 1024
        #input layer : 3 input channels, 6  output channels, 5 kernel size(5x5 filter channel)
        images=images.to(device)
        labels=labels.to(device)

        #forward pass
        outputs=model(images)
        loss=creterion(outputs, labels)

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%2000==0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_step}], Loss: {loss.item():.4f}')


print('Finished Training')

with torch.no_grad():
    correct=0
    samples=0
    n_class_objects= [0 for i in range(10)]
    n_class_samples= [0 for i in range(10)]
    for images, labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)

        outputs=model(images)
        _, predicted=torch.max(outputs, 1)
        samples+=labels.size(0)
        correct+=(predicted==labels).sum().item()

        for i in range(batch_size):
            label=labels[i]
            pred=predicted[i]
            if(label==pred):
                n_class_objects[label]+=1

            n_class_samples[label]+=1

    acc=100.0 * correct/samples
    print(f'Accuracy of the network on the 10000 test images: {acc}')

    for i in range(10):
        print(f'Accuracy of {classes[i]}: {100 * n_class_objects[i] / n_class_samples[i]} %')