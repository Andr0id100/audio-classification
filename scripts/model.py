import torch.nn as nn
import torch.nn.functional as F

class MFCCModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=input_size, out_features=32)
        self.linear_2 = nn.Linear(in_features=32, out_features=16)
        self.linear_3 = nn.Linear(in_features=16, out_features=num_classes)
        
    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = self.linear_3(x)
        
        return x