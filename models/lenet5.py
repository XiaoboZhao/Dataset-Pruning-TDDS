import torch
import torch.nn as nn

# Define the LeNet-5 architecture
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.embDim = 84

        # Layer 1: Conv 1x32x32 -> 6x28x28, kernel=5x5, stride=1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        # Layer 2: MaxPool 6x28x28 -> 6x14x14, kernel=2x2, stride=2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Layer 3: Conv 6x14x14 -> 16x10x10, kernel=5x5, stride=1
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        # Layer 4: MaxPool 16x10x10 -> 16x5x5, kernel=2x2, stride=2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Layer 5: Flatten and FC 16x5x5=400 -> 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # Layer 6: FC 120 -> 84
        self.fc2 = nn.Linear(120, 84)
        # Layer 7: FC 84 -> 10 (output classes)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                x = torch.nn.functional.relu(self.conv1(x))
                x = self.pool1(x)
                x = torch.nn.functional.relu(self.conv2(x))
                x = self.pool2(x)
                x = x.view(-1, 16 * 5 * 5)  # Flatten
                x = torch.nn.functional.relu(self.fc1(x))
                e = torch.nn.functional.relu(self.fc2(x))
        else:
            x = torch.nn.functional.relu(self.conv1(x))
            x = self.pool1(x)
            x = torch.nn.functional.relu(self.conv2(x))
            x = self.pool2(x)
            x = x.view(-1, 16 * 5 * 5)
            x = torch.nn.functional.relu(self.fc1(x))
            e = torch.nn.functional.relu(self.fc2(x))

        x = self.fc3(e)  # No activation for output layer
        if last:
            return x, e
        else:
            return x

    def get_embedding_dim(self):
        return self.embDim
    