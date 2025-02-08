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


def download_dataset(
    dataset: Dataset,
    base_folder_path: Path = Path("./share/raw_data"),
) -> str | None:

    assert dataset.name in ("mcagriaksoy", "dasmehdixtr")
    kaggle.api.authenticate()

    dataset_full_path_mapping = {
        "mcagriaksoy": "mcagriaksoy/amateur-unmanned-air-vehicle-detection-dataset",
        "dasmehdixtr": "dasmehdixtr/drone-dataset-uav",
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
) -> None:

    if dataset.meta_type == "xml":
        convert_xml_to_yolo(dataset.base_path)

    for image in tqdm(
        dataset.base_path.iterdir(), desc=f"Processing Dataset {dataset.name}"
    ):
        if image.suffix.lower() in {".jpg", ".png", ".jpeg"}:
            dest = output_images_dir / image.name
            dest.write_bytes(image.read_bytes())

        if image.suffix.lower() == ".txt":
            dest = output_annotations_dir / image.name
            dest.write_text(image.read_text())
