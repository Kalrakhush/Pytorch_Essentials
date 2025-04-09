import torch
import torch.nn as nn

class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size,):
        super(NeuralNet1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First layer
        self.relu = nn.ReLU()                          # Activation function
        self.fc2 = nn.Linear(hidden_size, 1) # Second layer

    def forward(self, x):
        out = self.fc1(x)                              # Input to first layer
        out = self.relu(out)                          # Activation function
        out = self.fc2(out)                           # Output layer
        #sigmoid at end for binary classification
        out = torch.sigmoid(out)                      # Sigmoid activation for binary classification
        return out                                   # Final \
    
model = NeuralNet1(input_size=28*28, hidden_size=5, num_classes=3) # Example sizes    
creterion=nn.BCELoss() # Loss function for multi-class classification