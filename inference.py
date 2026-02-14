# Comments:
# - SageMaker inference entry script for your SmallCNN
# - Fixes: import os + correct indentation + robust weight-path search


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import json

class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64,num_classes)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return self.fc(x)

def model_fn(model_dir):
    candidates = [
        os.path.join(model_dir, "model_cls.pth"),
        os.path.join(model_dir, "model", "model_cls.pth"),
    ]

    for p in candidates:
        if os.path.exists(p):
            model = SmallCNN()
            model.load_state_dict(torch.load(p, map_location="cpu"))
            model.eval()
            return model

    raise FileNotFoundError(f"model_cls.pth not found. Looked in: {candidates}")

def input_fn(request_body, content_type):
    img = Image.open(io.BytesIO(request_body)).convert("RGB").resize((224,224))
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)

def predict_fn(input_data, model):
    with torch.no_grad():
        logits = model(input_data)
        return int(torch.argmax(logits,1).item())

def output_fn(prediction, accept):
    return json.dumps({"prediction": int(prediction)})
