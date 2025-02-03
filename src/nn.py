import os
from pathlib import Path
from typing import Tuple, Dict, Any

import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms


class YoloDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        
        self.image_files = sorted([f for f in self.images_dir.iterdir() 
                                 if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_path = self.image_files[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        
        # Загрузка изображения
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # Загрузка аннотаций
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    
                    # Конвертация координат YOLO в абсолютные значения
                    x_min = x_center - width/2
                    y_min = y_center - height/2
                    x_max = x_center + width/2
                    y_max = y_center + height/2
                    
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(class_id))
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        return image, target

    def _load_labels(self, label_path, img_size):
        boxes = []
        labels = []
        h, w = img_size
        with open(label_path, "r") as f:
            for line in f:
                cls, x_center, y_center, width, height = map(float, line.split())
                x_min = (x_center - width / 2) * w
                y_min = (y_center - height / 2) * h
                x_max = (x_center + width / 2) * w
                y_max = (y_center + height / 2) * h
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(cls) + 1)

        return boxes, labels

