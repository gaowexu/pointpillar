import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.transforms import Lambda, transforms


class CustomImageDataset(Dataset):
    def __init__(self, images_root_dir, transform=None, target_transform=None):
        """
        Constructor of customized image dataset

        :param images_root_dir: root directory of images
        :param transform: transform handler for input images
        :param target_transform: transform handler for ground truth
        """
        self._images_root_dir = images_root_dir
        self._transform = transform
        self._target_transform = target_transform
        self._samples = self.collect_samples()

    @staticmethod
    def name2id(name):
        name_id_lut = {"buildings": 0, "forest": 1, "glacier": 2, "mountain": 3, "sea": 4, "street": 5}
        return name_id_lut[name]

    @staticmethod
    def id2name(cls_id):
        id_name_lut = {0: "buildings", 1: "forest", 2: "glacier", 3: "mountain", 4: "sea", 5: "street"}
        return id_name_lut[cls_id]

    def collect_samples(self):
        """
        collect all samples including both input images and ground truth labels

        :return: list of samples
        """
        samples = list()
        cls_names = os.listdir(self._images_root_dir)
        for cls_name in cls_names:
            cls_folder_root_dir = os.path.join(self._images_root_dir, cls_name)
            image_names = [name for name in os.listdir(cls_folder_root_dir) if name.endswith(".jpg")]

            for image_name in image_names:
                image_full_path = os.path.join(cls_folder_root_dir, image_name)
                label = self.name2id(cls_name)
                samples.append({"image_full_path": image_full_path, "label": label})

        return samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        image_full_path = self._samples[idx]["image_full_path"]
        label = self._samples[idx]["label"]
        image = read_image(image_full_path)     # tensor, torch.uint8

        if self._transform:
            image = self._transform(image)
        if self._target_transform:
            label = self._target_transform(label)
        return image, label
