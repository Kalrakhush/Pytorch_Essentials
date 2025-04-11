import torch
import io
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out
        
input_size = 784 #28*28
hidden_size = 100
num_classes = 10    
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

Path="app/mnist.fnn.pth"
model.load_state_dict(torch.load(Path))

model.eval()

#image->tensor
def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.Resize((28, 28)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ), (0.3801, ))
                                    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)
    #predict
def get_prediction(image_tensor):
    images=image_tensor.reshape(-1, 28*28)
    output = model(images)
    _, predicted = torch.max(output.data, 1)
    return predicted
    #return json