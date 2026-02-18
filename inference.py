# inference.py
import io, json, torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from torchvision import models

""" class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

def model_fn(model_dir):
    model = SmallCNN(num_classes=2)
    model.load_state_dict(torch.load(f"{model_dir}/model.pth", map_location="cpu"))
    model.eval()
    return model """
    
def model_fn(model_dir):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(f"{model_dir}/model.pth", map_location="cpu"))
    model.eval()
    return model

def input_fn(request_body, content_type):
    img = Image.open(io.BytesIO(request_body)).resize((224, 224)).convert("RGB")
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

""" def input_fn(request_body, content_type):
    img = Image.open(io.BytesIO(request_body)).resize((224, 224)).convert("RGB")
    transform = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    return transform(img).unsqueeze(0) """

def predict_fn(input_data, model):
    device = next(model.parameters()).device
    with torch.no_grad():
        return model(input_data.to(device))

def output_fn(prediction, accept):
    return json.dumps({"logits": prediction.tolist()})