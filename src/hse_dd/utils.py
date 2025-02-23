import random
from pathlib import Path
from typing import Literal
import xml.etree.ElementTree as ET

import kaggle
from pydantic import BaseModel
from tqdm import tqdm
from PIL import Image


class Dataset(BaseModel):
    name: str
    meta_type: Literal["yolo", "xml"] = "yolo"
    base_path: Path
    not_target: bool = False
    percentage_to_use: int = 100


def download_dataset(
    dataset: Dataset,
    base_folder_path: Path = Path("./share/raw_data"),
) -> str | None:

    kaggle.api.authenticate()

    dataset_full_path_mapping = {
        "mcagriaksoy": "mcagriaksoy/amateur-unmanned-air-vehicle-detection-dataset",
        "dasmehdixtr": "dasmehdixtr/drone-dataset-uav",
        "militaryaircraftdetectiondataset": "a2015003713/militaryaircraftdetectiondataset",
        "bird-species-classification": "akash2907/bird-species-classification",
    }

    data_path = Path(base_folder_path) / dataset.name
    if data_path.exists():
        return
    data_path.mkdir(parents=True)

    kaggle.api.dataset_download_files(
        dataset_full_path_mapping[dataset.name], path=str(data_path), unzip=True
    )


def convert_xml_to_yolo(dataset_dir: Path) -> None:
    for image in dataset_dir.iterdir():
        if image.suffix.lower() not in {".jpg", ".png", ".jpeg"}:
            continue

        xml_file = dataset_dir / f"{image.stem}.xml"
        yolo_file = dataset_dir / f"{image.name}.txt"

        with Image.open(str(image)) as img:
            image_width, image_height = img.size

        if not xml_file.exists():
            raise Exception(str(xml_file))

        tree = ET.parse(str(xml_file))
        root = tree.getroot()

        with open(str(yolo_file), "w") as f:
            for obj in root.findall("object"):
                class_name = obj.find("name").text

                class_id = 0 if class_name == "drone" else -1

                xmin = int(obj.find("bndbox/xmin").text)
                ymin = int(obj.find("bndbox/ymin").text)
                xmax = int(obj.find("bndbox/xmax").text)
                ymax = int(obj.find("bndbox/ymax").text)

                x_center = (xmin + xmax) / 2.0 / image_width
                y_center = (ymin + ymax) / 2.0 / image_height
                width = (xmax - xmin) / float(image_width)
                height = (ymax - ymin) / float(image_height)

                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def process_dataset(
    dataset: Dataset,
    output_images_dir: Path,
    output_annotations_dir: Path,
    seed: None | int = None,
) -> None:

    if dataset.not_target:
        image_files = list(dataset.base_path.iterdir())
        image_files.sort()

        num_files_to_select = int(len(image_files) * (dataset.percentage_to_use / 100))

        if seed is not None:
            random.seed(seed)

        selected_files = random.sample(image_files, num_files_to_select)
        for image in tqdm(
            selected_files, desc=f"Processing Dataset {dataset.name}"
        ):
            if image.suffix.lower() not in {".jpg", ".png", ".jpeg"}:
                continue
            image_dest = output_images_dir / image.name
            image_dest.write_bytes(image.read_bytes())
            annotation_dest = output_annotations_dir / f"{image.stem}.txt"
            annotation_dest.touch()

        return


    if dataset.meta_type == "xml":
        convert_xml_to_yolo(dataset.base_path)

    for image in tqdm(
        dataset.base_path.iterdir(), desc=f"Processing Dataset {dataset.name}"
    ):
        if image.suffix.lower() in {".jpg", ".png", ".jpeg"}:
            image_dest = output_images_dir / image.name
            annotation_dest = output_annotations_dir / f"{image.stem}.txt"
            annotation_src = dataset.base_path / f"{image.stem}.txt"
            if not annotation_src.exists():
                continue
            image_dest.write_bytes(image.read_bytes())
            annotation_dest.write_text(annotation_src.read_text())


