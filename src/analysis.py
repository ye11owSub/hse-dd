import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple

from .dataset import Dataset

def plot_sample_images(dataset: Dataset, num_samples: int = 5) -> None:
    """
    Отображает случайные изображения из датасета с их разметкой.
    
    Args:
        dataset: Объект датасета
        num_samples: Количество изображений для отображения
    """
    images = list(dataset.base_path.glob("*.jpg")) + \
             list(dataset.base_path.glob("*.jpeg")) + \
             list(dataset.base_path.glob("*.png"))
    
    samples = np.random.choice(images, min(num_samples, len(images)), replace=False)
    
    fig, axes = plt.subplots(1, len(samples), figsize=(20, 4))
    if len(samples) == 1:
        axes = [axes]
    
    for ax, img_path in zip(axes, samples):
        img = Image.open(img_path)
        ax.imshow(img)
        
        annotation_path = img_path.parent / f"{img_path.stem}.txt"
        if annotation_path.exists():
            img_width, img_height = img.size
            with open(annotation_path) as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    
                    x1 = int((x_center - width/2) * img_width)
                    y1 = int((y_center - height/2) * img_height)
                    x2 = int((x_center + width/2) * img_width)
                    y2 = int((y_center + height/2) * img_height)
                    
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                      fill=False, color='red', linewidth=2)
                    ax.add_patch(rect)
        
        ax.axis('off')
        ax.set_title(f"{img_path.name}", fontsize=10)
    
    plt.tight_layout()
    plt.show()

def print_dataset_summary(dataset: Dataset) -> Dict:
    """
    Анализирует и выводит статистику по датасету.
    
    Args:
        dataset: Объект датасета
    
    Returns:
        Dict: Статистика датасета
    """
    stats = {
        "total_images": 0,
        "images_with_drones": 0,
        "total_annotations": 0,
        "avg_boxes_per_image": 0,
        "avg_image_size": (0, 0),
        "image_sizes": [],
        "errors": []
    }
    
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        images.extend(list(dataset.base_path.rglob(ext)))
    
    stats["total_images"] = len(images)
    print(f"\nНайдено изображений: {stats['total_images']}")
    
    print(f"Путь к датасету: {dataset.base_path}")
    print(f"Тип метаданных: {dataset.meta_type}")
    
    for img_path in images:
        try:
            with Image.open(img_path) as img:
                stats["image_sizes"].append(img.size)
            
            annotation_path = img_path.parent / f"{img_path.stem}.txt"
            if dataset.meta_type == "xml":
                annotation_path = img_path.parent / f"{img_path.stem}.xml"
            
            if annotation_path.exists():
                print(f"Обработка аннотации: {annotation_path}")
                with open(annotation_path) as f:
                    boxes = f.readlines()
                    if len(boxes) > 0:
                        stats["images_with_drones"] += 1
                        stats["total_annotations"] += len(boxes)
            else:
                stats["errors"].append(f"Отсутствует аннотация для {img_path.name}")
                
        except Exception as e:
            stats["errors"].append(f"Ошибка при обработке {img_path.name}: {str(e)}")
    
    if stats["total_images"] > 0:
        stats["avg_boxes_per_image"] = stats["total_annotations"] / stats["total_images"]
        if stats["image_sizes"]:
            widths, heights = zip(*stats["image_sizes"])
            stats["avg_image_size"] = (int(np.mean(widths)), int(np.mean(heights)))
    
    print(f"\nДатасет: {dataset.name}")
    print("-" * 40)
    print(f"Всего изображений: {stats['total_images']}")
    print(f"Изображений с дронами: {stats['images_with_drones']}")
    print(f"Всего аннотаций: {stats['total_annotations']}")
    print(f"Среднее кол-во боксов на изображение: {stats['avg_boxes_per_image']:.2f}")
    print(f"Средний размер изображения: {stats['avg_image_size'][0]}x{stats['avg_image_size'][1]}")
    
    if stats["errors"]:
        print("\nОшибки обработки:")
        for error in stats["errors"][:10]:
            print(f"- {error}")
        if len(stats["errors"]) > 10:
            print(f"... и еще {len(stats['errors']) - 10} ошибок")
    
    return stats
