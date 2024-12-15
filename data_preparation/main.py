import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image
import json
from kaggle.api.kaggle_api_extended import KaggleApi

# Настройка Kaggle API
api = KaggleApi()
api.authenticate()

# Настройка URL датасетов
DATASET_1_KAGGLE = "mcagriaksoy/amateur-unmanned-air-vehicle-detection-dataset"
DATASET_2_KAGGLE = "dasmehdixtr/drone-dataset-uav"
DATASET_1_LOCAL = "./dataset_1"
DATASET_2_LOCAL = "./dataset_2"

# Пути к исходным и выходным данным
DATASET_1_DIR = "./dataset_1/Database1/Database1"
DATASET_2_YOLO_DIR = "./dataset_2/drone_dataset_yolo/dataset_txt"
DATASET_2_XML_DIR = "./dataset_2/dataset_xml_format/dataset_xml_format"

OUTPUT_DIR = "./combined_dataset"
ANNOTATIONS_DIR = os.path.join(OUTPUT_DIR, "annotations")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")

# Создаем выходные директории
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
problem_files = []

# Конвертация XML в YOLO формат

def convert_xml_to_yolo(xml_file, yolo_file, image_file):
    # Открываем изображение, чтобы получить его размеры
    with Image.open(image_file) as img:
        image_width, image_height = img.size

    # Парсим XML файл
    tree = ET.parse(xml_file)
    root = tree.getroot()

    with open(yolo_file, 'w') as f:
        for obj in root.findall('object'):
            # Получаем название класса
            class_name = obj.find('name').text
            # Здесь предполагается, что класс "drone" имеет индекс 0
            class_id = 0 if class_name == 'drone' else -1  # или другие классы, если их больше

            # Получаем координаты bounding box
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)

            # Нормализуем координаты
            x_center = (xmin + xmax) / 2.0 / image_width
            y_center = (ymin + ymax) / 2.0 / image_height
            width = (xmax - xmin) / float(image_width)
            height = (ymax - ymin) / float(image_height)

            # Записываем в YOLO формат
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Обработка Dataset 1 (только YOLO формат)
def process_dataset_1(dataset_dir, output_images_dir, output_annotations_dir):
    for file_name in tqdm(os.listdir(dataset_dir), desc="Processing Dataset 1"):
        if file_name.lower().endswith((".jpg", ".png", ".jpeg")):
            # Копируем изображение
            shutil.copy2(os.path.join(dataset_dir, file_name), os.path.join(output_images_dir, file_name))

            # Копируем аннотацию YOLO
            txt_file = os.path.join(dataset_dir, file_name.rsplit('.', 1)[0] + ".txt")
            if os.path.exists(txt_file):
                shutil.copy2(txt_file, os.path.join(output_annotations_dir, file_name.rsplit('.', 1)[0] + ".txt"))


# Обработка Dataset 2 (готовый YOLO формат)
def process_dataset_2_yolo(yolo_dir, output_images_dir, output_annotations_dir):
    for file_name in tqdm(os.listdir(yolo_dir), desc="Processing Dataset 2 YOLO"):
        if file_name.lower().endswith((".jpg", ".png", ".jpeg")):

            # Копируем изображение
            shutil.copy2(os.path.join(yolo_dir, file_name), os.path.join(output_images_dir, file_name))

            # Копируем аннотацию YOLO
            txt_file = os.path.join(yolo_dir, file_name.rsplit('.', 1)[0] + ".txt")
            if os.path.exists(txt_file):
                shutil.copy2(txt_file, os.path.join(output_annotations_dir, file_name.rsplit('.', 1)[0] + ".txt"))


# Обработка Dataset 2 (XML формат -> YOLO)
def process_dataset_2_xml(xml_dir, output_images_dir, output_annotations_dir):
    for file_name in tqdm(os.listdir(xml_dir), desc="Processing Dataset 2 XML"):
        if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".JPEG"):
            # Копируем изображение
            shutil.copy2(os.path.join(xml_dir, file_name), os.path.join(output_images_dir, file_name))

            # Конвертируем XML в YOLO
            xml_file = os.path.join(xml_dir, file_name.rsplit('.', 1)[0] + ".xml")
            yolo_file = os.path.join(output_annotations_dir, file_name.rsplit('.', 1)[0] + ".txt")
            if os.path.exists(xml_file):
                # Передаем путь к изображению для динамического получения размеров
                image_file = os.path.join(xml_dir, file_name)
                convert_xml_to_yolo(xml_file, yolo_file, image_file)


# Основная функция
def main():
    # Загрузка датасетов (раскомментируй при первом запуске)
    api.dataset_download_files(DATASET_1_KAGGLE, path=DATASET_1_LOCAL, unzip=True)
    api.dataset_download_files(DATASET_2_KAGGLE, path=DATASET_2_LOCAL, unzip=True)

    print("Processing Dataset 1...")
    process_dataset_1(DATASET_1_DIR, IMAGES_DIR, ANNOTATIONS_DIR)

    print("Processing Dataset 2 YOLO...")
    process_dataset_2_yolo(DATASET_2_YOLO_DIR, IMAGES_DIR, ANNOTATIONS_DIR)

    print("Processing Dataset 2 XML...")
    process_dataset_2_xml(DATASET_2_XML_DIR, IMAGES_DIR, ANNOTATIONS_DIR)

    # Проверка результатов
    image_files = os.listdir(IMAGES_DIR)
    annotation_files = os.listdir(ANNOTATIONS_DIR)
    with open('problem_files.json', 'w') as file:
        json.dump(problem_files, file)  # Сохраняем список в файл

    print(f"\nВсего изображений: {len(image_files)}")
    print(f"Всего аннотаций: {len(annotation_files)}")

if __name__ == "__main__":
    main()
