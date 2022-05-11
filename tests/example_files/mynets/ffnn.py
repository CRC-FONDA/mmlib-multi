from torch import nn


class FFNN(nn.Module):
    """
    This model is ment to be close to the FFNN model form the paper.
    If I interpret it correctly, their "optimal" configuration is a FFNN with 48 parameters per layer
    They say L_FC = 3, but only with 4 I reach the point of around 5000 parameters (which is the number listed in Table2)
    Also they use RELU as activation function.
    
    Update 07.02.:
        The current history CH1 and CH2 should also be inputs for the FFNN.
        The amount of total trainable parameters is now 5041, as in the paper.

    """

    def __init__(self):
        super().__init__()

        # The first fully connected layer has three inputs: I, T, and Q and CH1, CH2
        self.fc1 = nn.Linear(5, 48, bias=False)  # No bias for input layer
        # self.bn1 = nn.BatchNorm1d(48)
        self.activation1 = nn.ReLU()
        
        # Two fully connected layers follow
        self.fc2 = nn.Linear(48, 48)
        # self.bn2 = nn.BatchNorm1d(48)
        self.activation2 = nn.ReLU()
        
        self.fc3 = nn.Linear(48, 48)
        # self.bn3 = nn.BatchNorm1d(48)
        self.activation3 = nn.ReLU()
        
        # The output is one value: Terminal voltage
        # No normalization and no activation
        self.fc4 = nn.Linear(48, 1)
        
    def forward(self, x):
        # Removed ReLU for last layer
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.activation1(x)
        
        x = self.fc2(x)
        # x = self.bn2(x)
        x = self.activation2(x)
        
        x = self.fc3(x)
        # x = self.bn3(x)
        x = self.activation3(x)
        
        x = self.fc4(x)

        return x
