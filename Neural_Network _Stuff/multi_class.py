import torch
import torch.nn as nn

class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First layer
        self.relu = nn.ReLU()                          # Activation function
        self.fc2 = nn.Linear(hidden_size, num_classes) # Second layer

    def forward(self, x):
        out = self.fc1(x)                              # Input to first layer
        out = self.relu(out)                          # Activation function
        out = self.fc2(out)                           # Output layer
        #no sigmoid at end
        # because CrossEntropyLoss applies softmax internally
        return out                                   # Final \
    
model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3) # Example sizes    
creterion=nn.CrossEntropyLoss() # Loss function for multi-class classification