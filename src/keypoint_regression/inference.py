import io
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

KP_NAMES = [
    "left_eye", "right_eye", "mouth",
    "left_ear_1", "left_ear_2", "left_ear_3",
    "right_ear_1", "right_ear_2", "right_ear_3",
]

class KeypointModel(nn.Module):
    def __init__(self, num_keypoints: int = 9, pretrained: bool = False):
        super().__init__()
        self.num_keypoints = num_keypoints
        backbone = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_keypoints * 2),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def model_fn(model_dir):
    """Load model from the model_dir. Called once on container startup."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KeypointModel(num_keypoints=9, pretrained=False)
    model.load_state_dict(
        torch.load(f"{model_dir}/model_best.pth", map_location=device)
    )
    model.to(device)
    model.eval()
    return model


def input_fn(request_body, content_type):
    """Deserialize the request body into a tensor. Called on each request."""
    if content_type == "application/x-npy":
        array = np.load(io.BytesIO(request_body))
        return torch.tensor(array, dtype=torch.float32)
    elif content_type == "image/jpeg" or content_type == "image/png":
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        return transform(image).unsqueeze(0)  # add batch dim
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """Run inference. Called after input_fn."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    with torch.no_grad():
        output = model(input_data)
    return output.cpu().numpy()


def output_fn(prediction, accept):
    """Serialize the prediction output. Called after predict_fn."""
    if accept == "application/json":
        coords = prediction.reshape(-1, 2).tolist()
        result = {kp: {"x": coords[i][0], "y": coords[i][1]} 
                  for i, kp in enumerate(KP_NAMES)}
        return json.dumps(result), "application/json"
    raise ValueError(f"Unsupported accept type: {accept}")