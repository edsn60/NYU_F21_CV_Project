import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, out_size=30):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.maxpool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, out_size)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.maxpool(F.relu(self.conv3(x)))
        x = self.maxpool(F.relu(self.conv4(x)))

        x = x.view(-1, 256 * 6 * 6)
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(self.dropout(x))

        return x