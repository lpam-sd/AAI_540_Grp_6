# Comments:
# - Corrected Dataset: applies Albumentations to BOTH image + keypoints (landmarks)
# - Handles remove_invisible=True case (keypoints may drop)
# - Resizes image AND landmarks together to a fixed (224,224) so coords match model input
# - Keeps code minimal; you can extend to return kp_dict/features if needed

import numpy as np
import albumentations as A
import hashlib
import random


import io
import boto3
from PIL import Image

_s3 = boto3.client("s3")

def read_image_from_s3(s3_uri: str) -> Image.Image:
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    obj = _s3.get_object(Bucket=bucket, Key=key)
    return Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")
    


# --- Augmentation (cats only) ---
cat_aug = A.Compose(
    [
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=10,
            p=0.7
        ),
        A.RandomBrightnessContrast(p=0.4),
    ],
    keypoint_params=A.KeypointParams(
        format="xy",
        remove_invisible=False  # if True, keypoints can drop -> handled below
    )
)



def stable_int_seed(s3_uri: str, aug_idx: int, aug_version: str = "v1") -> int:
    s = f"{s3_uri}|{aug_idx}|{aug_version}".encode("utf-8")
    return int(hashlib.md5(s).hexdigest()[:8], 16)

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

class CatClsDataset:
    def __init__(self, df, augment=False, aug=None, image_size=(224, 224), k_aug=0, aug_version="v1"):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.aug = aug
        self.image_size = image_size  # (new_w, new_h)
        self.k_aug = k_aug
        self.aug_version = aug_version

        if self.augment and self.k_aug > 0 and self.aug is None:
            raise ValueError("k_aug > 0 but aug transform is None")

        # Landmark columns in fixed order
        self.KP_COL_ORDER = [
            ("left_eye_x", "left_eye_y"),
            ("right_eye_x", "right_eye_y"),
            ("mouth_x", "mouth_y"),
            ("left_ear_1_x", "left_ear_1_y"),
            ("left_ear_2_x", "left_ear_2_y"),
            ("left_ear_3_x", "left_ear_3_y"),
            ("right_ear_1_x", "right_ear_1_y"),
            ("right_ear_2_x", "right_ear_2_y"),
            ("right_ear_3_x", "right_ear_3_y"),
        ]

        # Build expanded index: (row_idx, aug_idx). aug_idx=0 means original
        self.indices = []
        for i, row in self.df.iterrows():
            self.indices.append((i, 0))
            if self.augment and int(row["label"]) == 1:
                for j in range(1, self.k_aug + 1):
                    self.indices.append((i, j))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx, aug_idx = self.indices[idx]
        row = self.df.iloc[real_idx]

        label = int(row["label"])
        s3_uri = row["s3_uri"]

        # Load image (expects you already have read_image_from_s3 defined)
        img_np = np.array(read_image_from_s3(s3_uri))

        # Build landmarks dict + keypoints list
        landmarks = {k: float(row[k]) for pair in self.KP_COL_ORDER for k in pair}
        kps = [(landmarks[cx], landmarks[cy]) for (cx, cy) in self.KP_COL_ORDER]

        # Optional: clip to current image bounds before augmentation (prevents weird negatives)
        h0, w0 = img_np.shape[:2]
        kps = [(min(max(x, 0.0), w0 - 1.0), min(max(y, 0.0), h0 - 1.0)) for (x, y) in kps]

        # Apply augmentation only for cats and only for augmented copies
        if aug_idx > 0 and label == 1 and self.aug is not None:
            seed = stable_int_seed(s3_uri, aug_idx, self.aug_version)
            set_all_seeds(seed)

            out = self.aug(image=img_np, keypoints=kps)
            img_np = out["image"]
            kps = out["keypoints"]

            # If remove_invisible=True, some keypoints may drop; fall back to original kp set
            if len(kps) != len(self.KP_COL_ORDER):
                kps = [(landmarks[cx], landmarks[cy]) for (cx, cy) in self.KP_COL_ORDER]
                # also clip to current image bounds
                h1, w1 = img_np.shape[:2]
                kps = [(min(max(x, 0.0), w1 - 1.0), min(max(y, 0.0), h1 - 1.0)) for (x, y) in kps]

        # Convert keypoints back to dict (post-aug, pre-resize)
        kp_dict = {}
        for (cx, cy), (x, y) in zip(self.KP_COL_ORDER, kps):
            kp_dict[cx] = float(x)
            kp_dict[cy] = float(y)

        # IMPORTANT: resize image AND landmarks together so they match model input (224,224)
        img_np, kp_dict = resize_image_and_landmarks(img_np, kp_dict, size=self.image_size)

        # Optionally normalize landmarks now that they are in 224-space
        new_w, new_h = self.image_size
        for (cx, cy) in self.KP_COL_ORDER:
            kp_dict[cx] = kp_dict[cx] / float(new_w)
            kp_dict[cy] = kp_dict[cy] / float(new_h)

        # Image normalization for model
        img_np = img_np.astype(np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))

        # If you later want features, you can compute from kp_dict here.
        # For now, returning image + label to keep behavior close to your original.
        return img_np, label
