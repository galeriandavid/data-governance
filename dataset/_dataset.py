"""Pytorch custom dataset for image segmentation tasks
"""
import json

import torch
from torchvision.io import read_image

from utils import rle2mask


class SegmentationDataset(torch.utils.data.Dataset):
    """Pytorch custom dataset for image segmentation tasks"""

    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        dataset = self._load_json(data_path)
        self.image_list = dataset["images"]
        self.num_channels = dataset["data_description"]["num_channels"]
        self.num_classes = len(dataset["data_description"]["classes"])
        self.classes = dataset["data_description"]["classes"]
        self._class_mapper = {
            class_name: i
            for i, class_name in enumerate(dataset["data_description"]["classes"])
        }
        self.dataset_len = dataset["data_description"]["num_images"]
        self.transform = transform

    def __len__(self):
        """Return num images in dataset"""
        return self.dataset_len

    def __getitem__(self, index):
        """Return img and true segmentation mask"""
        img_info = self.image_list[index]
        img = read_image(self.data_path + img_info["path"])
        img = img / 255
        height, width = img.shape[1:]
        mask = torch.zeros((self.num_classes, height, width), dtype=torch.int8)
        for i in range(len(img_info["labels"])):
            mask[self._class_mapper[img_info["labels"][i]["class"]]] = rle2mask(
                img_info["labels"][i]["rle"], (height, width)
            )

        return img, mask

    @staticmethod
    def _load_json(data_path):
        """Load json with data descriptions and labels"""
        with open(f"{data_path}labels.json") as json_file:
            dataset = json.load(json_file)
        return dataset
