import os
from torch.utils.data import Dataset


class PC3DDetDataset(Dataset):
    def __init__(self, dataset_root_dir, category_names, phase="train"):
        self._dataset_root_dir = dataset_root_dir
        self._category_names = category_names
        self._phase = phase
        self._samples = self.collect_samples()

    def collect_samples(self):
        if self._phase == "train":
            names = [x.strip() for x in open(os.path.join(self._dataset_root_dir, "ImageSets/train.txt")).readlines()]
        elif self._phase == "validation":
            names = [x.strip() for x in open(os.path.join(self._dataset_root_dir, "ImageSets/val.txt")).readlines()]
        else:
            raise RuntimeError("Phase {} not valid.".format(self._phase))

    def __len__(self):
        return 0

    def __getitem__(self, index):
        pass

    #
    #
    #     samples = list()
    #     cls_names = os.listdir(self._images_root_dir)
    #     for cls_name in cls_names:
    #         cls_folder_root_dir = os.path.join(self._images_root_dir, cls_name)
    #         image_names = [name for name in os.listdir(cls_folder_root_dir) if name.endswith(".jpg")]
    #
    #         for image_name in image_names:
    #             image_full_path = os.path.join(cls_folder_root_dir, image_name)
    #             label = self.name2id(cls_name)
    #             samples.append({"image_full_path": image_full_path, "label": label})
    #
    #     return samples
    #
    # def __len__(self):
    #     return len(self._samples)
    #
    # def __getitem__(self, idx):
    #     image_full_path = self._samples[idx]["image_full_path"]
    #     label = self._samples[idx]["label"]
    #     image = read_image(image_full_path)     # tensor, torch.uint8
    #
    #     if self._transform:
    #         image = self._transform(image)
    #     if self._target_transform:
    #         label = self._target_transform(label)
    #     return image, label


if __name__ == "__main__":
    pass


