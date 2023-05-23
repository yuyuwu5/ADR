import logging

import numpy as np
import torch
from PIL import Image


class AuxDDPM(torch.utils.data.Dataset):
    def __init__(self, file_path, transform=None):
        self.transform = transform
        logging.info("Loading auxiliary dataset from: %s..." % file_path)
        data = np.load(file_path)
        self.img = data["image"]
        self.label = data["label"]

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        img = self.img[idx]
        label = self.label[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
