import os
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class YoloDataset(Dataset):
    def __init__(self, images_dir: Path, labels_dir: Path, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_path = os.path.join(
            self.labels_dir, self.image_files[idx].replace(".jpg", ".txt")
        )
        boxes, labels = self._load_labels(label_path, image.shape[:2])

        if self.transform:
            image = self.transform(image)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
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

