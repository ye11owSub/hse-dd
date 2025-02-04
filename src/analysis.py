import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import streamlit as st
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple

from .dataset import Dataset

# Пути к данным
IMAGES_DIR = "./combined_dataset/images"
ANNOTATIONS_DIR = "./combined_dataset/annotations"


# Функция для анализа данных
def analyze_data(images_dir, annotations_dir):
    image_files = os.listdir(images_dir)
    annotation_files = os.listdir(annotations_dir)

    # 1. Количество изображений
    total_images = len(image_files)

    # 2. Размеры изображений + потерянные аннотации

    missing_annotations = []


    image_sizes = []
    for image_file in image_files:
        try:
            with Image.open(os.path.join(images_dir, image_file)) as img:
                image_sizes.append(img.size)

            # Проверка на наличие аннотации для изображения
            annotation_file = os.path.splitext(image_file)[0] + ".txt"
            #annotation_file = img.rsplit('.', 1)[0] + ".txt"

            if annotation_file not in annotation_files:
                missing_annotations.append(annotation_file)

        except Exception as e:
            print(f"Ошибка при загрузке изображения {image_file}: {e}")



    sizes_df = pd.DataFrame(image_sizes, columns=["width", "height"])

    # 3. Количество типов объектов и объектов каждого типа
    object_counts = Counter()
    object_counts2 = Counter()

    empty_annotations = 0
    empty_annotation_files = []
    class_0_count = 0

    for annotation_file in annotation_files:
        annotation_path = os.path.join(annotations_dir, annotation_file)
        with open(annotation_path) as f:
            lines = f.readlines()
            if not lines:  # Пустая аннотация
                empty_annotations += 1
                empty_annotation_files.append(annotation_file)
                continue
            for line in lines:
                parts = line.split()
                class_id = int(parts[0])
                if class_id == 0:
                    class_0_count += 1
                    object_counts[class_id] += 1
                else:
                    object_counts2[class_id] += 1


    # 4. Размеры объектов
    object_sizes = []
    for annotation_file in annotation_files:
        with open(os.path.join(annotations_dir, annotation_file)) as f:
            lines = f.readlines()
            for line in lines:
                bbox = list(map(float, line.split()[1:5]))  # [center_x, center_y, width, height]
                object_sizes.append(bbox[2] * bbox[3])  # Площадь объекта

    # 5. Подсчет изображений без аннотаций

    missing_images_count = len(missing_annotations)
    missing_images_percent = (missing_images_count / total_images) * 100

    # 6. Процент пустых аннотаций
    empty_annotations_percent = (empty_annotations / len(annotation_files)) * 100

    # Вывод статистики
    st.write(f"Общее количество изображений: {total_images}")
    st.write(f"Общее количество аннотаций: {len(annotation_files)}")
    st.write(f"Процент изображений без аннотаций: {missing_images_percent:.2f}%")
    st.write(f"Процент пустых аннотаций: {empty_annotations_percent:.2f}%")

    st.write("Имена изображений без аннотаций:")
    st.selectbox("Просмотрите изображения", missing_annotations)

    st.write("Имена файлов с пустыми аннотациями:")
    st.selectbox("Просмотрите файлы с пустой аннотацией", empty_annotation_files)

    st.write(f"\nРазмеры изображений (min, max, mean):")
    st.write(sizes_df.describe())

    st.write(f"\nОбъекты по типам:")
    st.write(object_counts)
    st.write(object_counts2)

    st.write(f"\nРазмеры объектов (min, max, mean):")
    st.write(pd.Series(object_sizes).describe())

    st.write(f"\nКоличество объектов с классом 0 (drone): {class_0_count}")

# Визуализация
def visualize_data(images_dir, annotations_dir):
    image_files = os.listdir(images_dir)

    # Гистограмма по размерам изображений
    image_sizes = []
    for image_file in image_files:
        try:
            with Image.open(os.path.join(images_dir, image_file)) as img:
                image_sizes.append(img.size)
        except Exception as e:
            print(f"Ошибка при загрузке изображения {image_file}: {e}")

    sizes_df = pd.DataFrame(image_sizes, columns=["width", "height"])
    fig, ax = plt.subplots()
    ax.scatter(sizes_df["width"], sizes_df["height"], alpha=0.5)
    ax.set_title("Размеры изображений")
    ax.set_xlabel("Ширина")
    ax.set_ylabel("Высота")
    st.pyplot(fig)

    # Гистограмма по размерам объектов
    object_sizes = []

    class_0_count = 0
    class_minus1_count = 0

    annotation_files = os.listdir(annotations_dir)
    for annotation_file in annotation_files:
        with open(os.path.join(annotations_dir, annotation_file)) as f:
            lines = f.readlines()
            for line in lines:
                bbox = list(map(float, line.split()[1:5]))
                object_sizes.append(bbox[2] * bbox[3])

                parts = line.split()
                class_id = int(parts[0])
                if class_id == 0:
                    class_0_count += 1
                else:
                    class_minus1_count += 1


    fig, ax = plt.subplots()
    ax.hist(object_sizes, bins=50)
    ax.set_title("Размеры объектов")
    ax.set_xlabel("Площадь объекта")
    ax.set_ylabel("Частота")
    st.pyplot(fig)

    # График распределения классов объектов
    fig, ax = plt.subplots()
    ax.bar(class_0_count, class_minus1_count)
    ax.set_title("Распределение типов объектов")
    ax.set_xlabel("Кол-во объектов")
    ax.set_ylabel("Тип объекта (class_id)")
    st.pyplot(fig)

def main():
    analyze_data(IMAGES_DIR, ANNOTATIONS_DIR)
    visualize_data(IMAGES_DIR, ANNOTATIONS_DIR)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Анализ и визуализация данных",
        page_icon="📊",
        layout="wide"
    )
    st.title('Анализ и визуализация данных')
    main()

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
        "image_sizes": []
    }
    
    images = list(dataset.base_path.glob("*.jpg")) + \
             list(dataset.base_path.glob("*.jpeg")) + \
             list(dataset.base_path.glob("*.png"))
    
    stats["total_images"] = len(images)
    
    for img_path in images:
        annotation_path = img_path.parent / f"{img_path.stem}.txt"
        if annotation_path.exists():
            with Image.open(img_path) as img:
                stats["image_sizes"].append(img.size)
            
            with open(annotation_path) as f:
                boxes = f.readlines()
                if len(boxes) > 0:
                    stats["images_with_drones"] += 1
                    stats["total_annotations"] += len(boxes)
    
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
    
    return stats
