
import torch.nn as nn
from torchvision import models

def create_model(num_classes=5):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )
    return model
