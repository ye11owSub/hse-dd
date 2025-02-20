import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple
import random

from .dataset import Dataset

def plot_sample_images(dataset: Dataset, num_samples: int = 5) -> None:
    """
    Отображает случайные изображения из датасета с их разметкой.
    
    Args:
        dataset: Объект датасета
        num_samples: Количество изображений для отображения
    """
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        images.extend(list(dataset.base_path.glob(ext)))
    
    samples = np.random.choice(images, min(num_samples, len(images)), replace=False)
    
    fig, axes = plt.subplots(1, len(samples), figsize=(20, 4))
    if len(samples) == 1:
        axes = [axes]
    
    for ax, img_path in zip(axes, samples):
        img = Image.open(img_path)
        ax.imshow(img)
        
        # Учитываем тип метаданных
        extension = ".xml" if dataset.meta_type == "xml" else ".txt"
        annotation_path = img_path.parent / f"{img_path.stem}{extension}"
        
        if annotation_path.exists():
            img_width, img_height = img.size
            if dataset.meta_type == "xml":
                import xml.etree.ElementTree as ET
                tree = ET.parse(annotation_path)
                root = tree.getroot()
                for obj in root.findall('.//object'):
                    bndbox = obj.find('bndbox')
                    if bndbox is not None:
                        x1 = int(float(bndbox.find('xmin').text))
                        y1 = int(float(bndbox.find('ymin').text))
                        x2 = int(float(bndbox.find('xmax').text))
                        y2 = int(float(bndbox.find('ymax').text))
                        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                          fill=False, color='red', linewidth=2)
                        ax.add_patch(rect)
            else:
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

def show_correct_examples(dataset: Dataset, num_samples: int = 5) -> None:
    """
    Отображает примеры корректно размеченных изображений с дронами.
    
    Args:
        dataset: Объект датасета
        num_samples: Количество примеров для отображения
    """
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        images.extend(list(dataset.base_path.glob(ext)))
    
    # Находим изображения с корректными аннотациями
    correct_images = []
    for img_path in images:
        annotation_path = img_path.parent / f"{img_path.stem}.txt"
        if annotation_path.exists():
            with open(annotation_path) as f:
                if f.read().strip():  # Проверяем, что файл не пустой
                    correct_images.append(img_path)
    
    if not correct_images:
        print("Не найдено корректно размеченных изображений")
        return
    
    # Выбираем случайные примеры
    samples = np.random.choice(correct_images, min(num_samples, len(correct_images)), replace=False)
    
    # Отображаем примеры
    fig, axes = plt.subplots(1, len(samples), figsize=(20, 4))
    if len(samples) == 1:
        axes = [axes]
    
    for ax, img_path in zip(axes, samples):
        img = Image.open(img_path)
        ax.imshow(img)
        
        # Отображаем боксы
        annotation_path = img_path.parent / f"{img_path.stem}.txt"
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
        ax.set_title(f"{img_path.name}\n{len(open(annotation_path).readlines())} дронов", 
                    fontsize=10)
    
    plt.tight_layout()
    plt.show()

def print_dataset_summary(datasets: List[Dataset]) -> List[Dict]:
    """
    Анализирует и выводит статистику по всем датасетам.
    """
    root_path = Path("reports")
    root_path.mkdir(exist_ok=True)
    
    all_stats = []
    dataset_groups = {}
    
    # Группируем датасеты по имени
    for dataset in datasets:
        if dataset.name not in dataset_groups:
            dataset_groups[dataset.name] = []
        dataset_groups[dataset.name].append(dataset)
    
    for dataset_name, dataset_list in dataset_groups.items():
        report_path = root_path / f"{dataset_name}_errors_report.txt"
        success_path = root_path / f"{dataset_name}_success_report.txt"
        
        stats = {
            "dataset_name": dataset_name,
            "total_images": 0,
            "images_with_drones": 0,
            "images_without_drones": 0,
            "missing_annotations": 0,
            "unprocessed_images": 0,
            "total_annotations": 0,
            "total_boxes": 0,
            "avg_boxes_per_image": 0,
            "avg_image_size": (0, 0),
            "image_sizes": [],
            "errors": []
        }
        
        # Собираем все уникальные изображения
        image_info = {}  # словарь для хранения информации о каждом изображении
        
        for dataset in dataset_list:
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                for img_path in dataset.base_path.rglob(ext):
                    if img_path.name not in image_info:
                        image_info[img_path.name] = {
                            'path': img_path,
                            'has_annotation': False,
                            'num_boxes': 0
                        }
                    
                    # Проверяем YOLO аннотацию
                    txt_path = img_path.parent / f"{img_path.stem}.txt"
                    if txt_path.exists():
                        with open(txt_path) as f:
                            boxes = [box.strip() for box in f.readlines()]
                            num_boxes = len([box for box in boxes if box])
                            if num_boxes > 0:
                                image_info[img_path.name]['has_annotation'] = True
                                image_info[img_path.name]['num_boxes'] = max(
                                    image_info[img_path.name]['num_boxes'],
                                    num_boxes
                                )
        
        # Подсчитываем статистику
        stats["total_images"] = len(image_info)
        
        for info in image_info.values():
            try:
                with Image.open(info['path']) as img:
                    stats["image_sizes"].append(img.size)
            except Exception:
                continue
                
            if info['has_annotation']:
                stats["images_with_drones"] += 1
                stats["total_boxes"] += info['num_boxes']
            else:
                stats["missing_annotations"] += 1
        
        # Вычисляем средние значения
        if stats["image_sizes"]:
            widths, heights = zip(*stats["image_sizes"])
            stats["avg_image_size"] = (int(np.mean(widths)), int(np.mean(heights)))
        
        if stats["total_images"] > 0:
            stats["avg_boxes_per_image"] = stats["total_boxes"] / stats["total_images"]
        
        # Выводим статистику
        print(f"\nДатасет: {dataset_name}")
        print(f"Путь к датасету: {[str(d.base_path) for d in dataset_list]}")
        print(f"Тип метаданных: {[d.meta_type for d in dataset_list]}")
        print("-" * 80)
        print(f"Всего изображений: {stats['total_images']}")
        print(f"Изображений корректно размеченных: {stats['images_with_drones']}")
        print(f"Изображений с пустыми аннотациями: {stats['images_without_drones']}")
        print(f"Изображений без аннотаций: {stats['missing_annotations']}")
        print(f"Изображений не обработано: {stats['unprocessed_images']}")
        print(f"Всего файлов аннотаций: {stats['images_with_drones']}")
        print(f"Всего боксов: {stats['total_boxes']}")
        print(f"Среднее кол-во боксов на изображение: {stats['avg_boxes_per_image']:.2f}")
        print(f"Средний размер изображения: {stats['avg_image_size'][0]}x{stats['avg_image_size'][1]}")
        print("-" * 80)
        
        # Сохраняем отчеты
        save_error_report(report_path, stats, dataset_list)
        save_success_report(success_path, stats, dataset_list, list(image_info.values()))
        
        all_stats.append(stats)
    
    return all_stats

def save_error_report(path: Path, stats: Dict, datasets: List[Dataset]) -> None:
    """Сохраняет отчет об ошибках"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Отчёт об ошибках в датасете: {stats['dataset_name']}\n")
        f.write("=" * 50 + "\n\n")
        f.write("Форматы данных:\n")
        for dataset in datasets:
            f.write(f"- {dataset.meta_type} в {dataset.base_path}\n")
        f.write("\n")
        
        # 1. Изображения без аннотаций
        f.write("1. Изображения без файлов аннотаций:\n")
        f.write("-" * 40 + "\n")
        
        missing_annotations = []
        # Собираем все изображения из всех директорий датасета
        all_images = []
        for dataset in datasets:
            all_images.extend(Path(dataset.base_path).rglob("*.[jJ][pP][eE]?[gG]"))
        
        for img_path in all_images:
            has_annotation = False
            # Проверяем наличие YOLO-аннотации во всех директориях датасета
            for dataset in datasets:
                txt_path = img_path.parent / f"{img_path.stem}.txt"
                if txt_path.exists():
                    has_annotation = True
                    break
            
            if not has_annotation:
                missing_annotations.append(img_path.name)
        
        for file in sorted(missing_annotations):
            f.write(f"- {file}\n")
        f.write(f"\nВсего: {len(missing_annotations)}\n\n")
        
        # 2. Изображения с пустыми аннотациями
        f.write("2. Изображения с пустыми аннотациями:\n")
        f.write("-" * 40 + "\n")
        empty_annotations = []
        
        for img_path in all_images:
            txt_path = img_path.parent / f"{img_path.stem}.txt"
            if txt_path.exists() and txt_path.stat().st_size == 0:
                empty_annotations.append(img_path.name)
        
        for file in sorted(empty_annotations):
            f.write(f"- {file}\n")
        f.write(f"\nВсего: {len(empty_annotations)}\n\n")
        
        # 3. Ошибки обработки
        f.write("3. Ошибки обработки изображений:\n")
        f.write("-" * 40 + "\n")
        processing_errors = [
            error for error in stats["errors"]
            if "Ошибка при обработке" in error
        ]
        for error in sorted(processing_errors):
            f.write(f"- {error}\n")
        f.write(f"\nВсего: {len(processing_errors)}\n")

def save_success_report(path: Path, stats: Dict, datasets: List[Dataset], images: List[Path]) -> None:
    """Сохраняет отчет об успешных обработках"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Отчёт об успешно обработанных изображениях: {stats['dataset_name']}\n")
        f.write("=" * 50 + "\n\n")
        f.write("Форматы данных:\n")
        for dataset in datasets:
            f.write(f"- {dataset.meta_type} в {dataset.base_path}\n")
        f.write("\n")
        f.write("Корректно размеченные изображения:\n")
        f.write("-" * 40 + "\n")
        
        # Собираем все изображения из всех директорий
        all_images = []
        for dataset in datasets:
            all_images.extend(Path(dataset.base_path).rglob("*.[jJ][pP][eE]?[gG]"))
        
        image_boxes = {}
        
        # Проверяем каждое изображение
        for img_path in all_images:
            txt_path = img_path.parent / f"{img_path.stem}.txt"
            
            try:
                if txt_path.exists():
                    with open(txt_path) as f_ann:
                        boxes = f_ann.readlines()
                        num_boxes = sum(1 for box in boxes if box.strip())
                        
                        if num_boxes > 0:
                            current_boxes = image_boxes.get(img_path.name, 0)
                            image_boxes[img_path.name] = max(current_boxes, num_boxes)
            except Exception as e:
                continue
        
        # Сортировка по количеству боксов и имени файла
        for img_name, num_boxes in sorted(image_boxes.items(), key=lambda x: (-x[1], x[0])):
            f.write(f"- {img_name} (боксов: {num_boxes})\n")
        
        f.write(f"\nВсего успешно обработанных изображений: {len(image_boxes)}")
