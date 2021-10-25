import torch
from torch import nn

class LogisticRegression(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LogisticRegression, self).__init__()

        self.model=nn.Linear(in_dim, out_dim)
        nn.init.normal_(self.model.weight, 0, 0.01)
        nn.init.constant_(self.model.bias, 0)

        
    def forward(self, x):
        # returns logits
        return self.model(x)
    
    def calculate_loss(self, logits, targets):
        return nn.BCEWithLogitsLoss()(logits, targets)
