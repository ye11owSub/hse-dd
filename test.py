import sys
from pathlib import Path

#sys.path.append(".")

import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.metrics import average_precision_score

from src.dataset import Dataset, download_dataset, process_dataset
from src.nn import YoloDataset
from src.analysis import plot_sample_images, print_dataset_summary

# Настройка путей
datasets_dir = Path("./data/raw")
OUTPUT_DIR = Path("./data/processed")
ANNOTATIONS_DIR = OUTPUT_DIR / "annotations"
IMAGES_DIR = OUTPUT_DIR / "images"

# Создание директорий
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Определение датасетов
datasets = [
    Dataset(
        name="dasmehdixtr",
        base_path=datasets_dir / "dasmehdixtr" / "drone_dataset_yolo" / "dataset_txt"
    ),
    Dataset(
        name="dasmehdixtr",
        meta_type="xml",
        base_path=datasets_dir / "dasmehdixtr" / "dataset_xml_format" / "dataset_xml_format"
    ),
    Dataset(
        name="mcagriaksoy",
        base_path=datasets_dir / "mcagriaksoy" / "Database1" / "Database1"
    ),
]

# Загрузка и обработка датасетов
for dataset in datasets:
    download_dataset(dataset, datasets_dir)
    process_dataset(dataset, IMAGES_DIR, ANNOTATIONS_DIR)
    print(f"Processing Dataset {dataset.name}_{dataset.meta_type}...")

# Анализируем все датасеты вместе
stats = print_dataset_summary(datasets)
    
# Показываем примеры из каждого датасета
for dataset in datasets:
    print(f"\nПримеры изображений из датасета {dataset.name}_{dataset.meta_type}:")
    plot_sample_images(dataset, num_samples=3)

# Проверка результатов
image_files = list(IMAGES_DIR.iterdir())
annotation_files = list(ANNOTATIONS_DIR.iterdir())

print(f"\nВсего изображений: {len(image_files)}")
print(f"Всего аннотаций: {len(annotation_files)}")