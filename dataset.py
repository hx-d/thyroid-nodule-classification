# -*- coding: utf-8 -*-
"""Dataset for TN5000 thyroid ultrasound images with XML annotations.

Expected layout::

    <data_root>/
    ├── JPEGImages/     *.jpg
    ├── Annotations/    *.xml   (the nodule label is stored in <object><name>)
    └── ImageSets/Main/ train.txt / val.txt / test.txt  (one image name per line)

Label convention: ``0`` = benign, ``1`` = malignant.
"""

import os
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset


class ThyroidDataset(Dataset):
    """Read images by split list and parse the label from the XML annotation."""

    def __init__(self, data_root, split='train', transform=None):
        self.img_dir = os.path.join(data_root, 'JPEGImages')
        self.ann_dir = os.path.join(data_root, 'Annotations')
        list_file = os.path.join(data_root, 'ImageSets', 'Main', f'{split}.txt')

        if not os.path.isfile(list_file):
            raise FileNotFoundError(f'split file not found: {list_file}')

        with open(list_file, encoding='utf-8') as f:
            self.names = [line.strip() for line in f if line.strip()]

        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        image = Image.open(os.path.join(self.img_dir, f'{name}.jpg')).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        tree = ET.parse(os.path.join(self.ann_dir, f'{name}.xml'))
        label = int(tree.find('object/name').text.strip())
        return image, label

    # ------------------------------------------------------------------
    def class_counts(self):
        """Return ``[n_benign, n_malignant]`` for this split."""
        counts = [0, 0]
        for name in self.names:
            tree = ET.parse(os.path.join(self.ann_dir, f'{name}.xml'))
            counts[int(tree.find('object/name').text.strip())] += 1
        return counts
