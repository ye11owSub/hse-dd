from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm.notebook import tqdm

from hse_dd.nn import YoloDataset
from hse_dd.utils import Dataset, download_dataset, process_dataset


class TrainPipeline:

    config = {
        "root_path": "./share/combined_dataset",
        "transform": transforms.Compose([transforms.ToTensor()]),
        "train_size": 0.8,
        "num_classes": 2,
        "device": (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
        "num_workers": 0,
        "batch_size": 64,
        "MODEL": "resnet50",
        "LR": 1e-3,
        "GRAD_CLIP": 0.1,
        "WEIGHT_DECAY": 1e-4,
        "NUM_EPOCHS": 70,
    }

    def __init__(self, preprocess_data: bool = False):
        if preprocess_data:
            self.preprocess_data()

        root_path = Path(self.config.get("root_path", "./share/combined_dataset"))

        self.images = root_path / "images"
        self.labels = root_path / "annotations"
        self.dataset = YoloDataset(
            self.images, self.labels, transform=self.config["transform"]
        )

    def normalise(self, inputs):
        mean = inputs.mean(dim=(2, 3)).unsqueeze(2).unsqueeze(3)
        std = inputs.std(dim=(2, 3)).unsqueeze(2).unsqueeze(3)

        return (inputs - mean) / (std + 1e-8)

    def get_accuracy(self, trues, predictions):
        trues, predictions = trues.detach().cpu(), predictions.detach().cpu()
        predictions = np.argmax(predictions, axis=1)
        return (trues == predictions).mean(dtype=torch.float32)

    def train(
        self,
        model,
        train_dataloader,
        criterion,
        optimizer,
        scheduler,
        grad_clip,
        device="cpu",
    ):
        model.to(device)
        model.train()

        losses = []
        accuracies = []

        for inputs, labels in tqdm(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.byte()
            inputs = self.normalise(inputs)

            predictions = model(inputs)
            loss = criterion(predictions, labels)
            loss.backward()

            if grad_clip:
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            losses.append(loss.item())
            accuracies.append(self.get_accuracy(labels, predictions))

        mean_accuracy = np.mean(accuracies)
        mean_loss = np.mean(losses)
        return model, mean_accuracy, mean_loss

    def validation(self, model, val_dataloader, criterion, device="cpu"):
        model.to(device)
        model.eval()

        losses = []
        accuracies = []

        for inputs, labels in tqdm(val_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.byte()
            inputs = self.normalise(inputs)

            with torch.no_grad():
                predictions = model(inputs)

                loss = criterion(predictions, labels)

                losses.append(loss.item())
                accuracies.append(self.get_accuracy(labels, predictions))

        mean_accuracy = np.mean(accuracies)
        mean_loss = np.mean(losses)
        return mean_accuracy, mean_loss

    def run(self):
        train_size = int(self.config["train_size"] * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=self.config["num_workers"],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=self.config["num_workers"],
        )

        #model = fasterrcnn_resnet50_fpn(pretrained=True)
        #model.to(self.config["device"])

        model = MODELS_MAPPING[self.config["MODEL"]]
        num_inputs = model.fc.in_features
        model.fc = torch.nn.Linear(num_inputs, self.config["NUM_CLASSES"])

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["LR"], weight_decay=self.config["WEIGHT_DECAY"])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.config["LR"],
            epochs=self.config["NUM_EPOCHS"],
            steps_per_epoch=len(train_loader)
        )



        train_accuracy = []
        train_loss = []

        val_accuracy = []
        val_loss = []

        best_accuracy = 0

        for epoch in range(self.config["NUM_EPOCHS"]):
            print(f"Epoch #{epoch} started.")

            model, t_acc, t_loss = self.train(model, train_loader, criterion, optimizer, scheduler, grad_clip=self.config["GRAD_CLIP"], device=self.config["DEVICE"])
            train_accuracy.append(t_acc)
            train_loss.append(t_loss)

            print(f"Train loss: {t_loss}, accuracy: {t_acc}")

            v_acc, v_loss = self.validation(model, test_loader, criterion, device=self.config['DEVICE'])
            val_accuracy.append(v_acc)
            val_loss.append(v_loss)

            print(f"Validation loss: {v_loss}, accuracy: {v_acc}")

            if v_acc > best_accuracy:
                print(f"Saving best accuracy so far, {v_acc}")
                best_accuracy = v_acc
                state = {"epoch": epoch, "config": self.config, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),}
                torch.save(state, "./best_state.pth")


        print(f"Best accuracy {best_accuracy}")

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

        ax[0, 0].set_title("Train loss")
        ax[0, 0].plot(train_loss)

        ax[0, 1].set_title("Train accuracy")
        ax[0, 1].plot(train_accuracy)

        ax[1, 0].set_title("Validation loss")
        ax[1, 0].plot(val_loss)

        ax[1, 1].set_title("Validation accuracy")
        ax[1, 1].plot(val_accuracy)

        plt.tight_layout()
        plt.show()

    def preprocess_data(self) -> None:
        raw_data = Path(self.config.get("raw_data_path", ""))

        if not raw_data.exists():
            raise ValueError("raw path is not provided")

        self.images.mkdir(parents=True, exist_ok=True)
        self.labels.mkdir(parents=True, exist_ok=True)

        datasets = (
            Dataset(
                name="dasmehdixtr",
                base_path=raw_data
                / "dasmehdixtr"
                / "drone_dataset_yolo"
                / "dataset_txt",
            ),
            Dataset(
                name="dasmehdixtr",
                meta_type="xml",
                base_path=raw_data
                / "dasmehdixtr"
                / "dataset_xml_format"
                / "dataset_xml_format",
            ),
            Dataset(
                name="mcagriaksoy",
                base_path=raw_data / "mcagriaksoy" / "Database1" / "Database1",
            ),
        )

        for dataset in datasets:
            download_dataset(dataset, raw_data)
            process_dataset(dataset, self.images, self.labels)
            print(f"Processing Dataset {dataset.name}_{dataset.meta_type}...")
