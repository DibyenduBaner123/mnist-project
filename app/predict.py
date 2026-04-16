import torch
import torch.nn as nn

# same model as training
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.model(x)

def load_model():
    model = MNISTModel()
    model.load_state_dict(torch.load("model/mnist_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(image, model):
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    return pred.item()