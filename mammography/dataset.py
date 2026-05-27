#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MammographyDataset(Dataset):
    """
    Custom Dataset for mammography images.
     - Expects a DataFrame with columns: 'image_path' and 'view'.
     - 'view' should be either 'CC' or 'MLO'.
     - Images are read in grayscale, normalized to [0,1], and transformed to tensors.
    """
    def __init__(self, dataframe, transform=None):

        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_map = {
            "CC": 0,
            "MLO": 1
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = row["image_path"]

        # Read grayscale image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Convert to float32
        image = image.astype("float32")

        # Normalize to [0,1]
        image /= 255.0

        label = self.label_map[row["view"]]

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        return image, label
    

def get_transform():
    """
    Define image transformations using Albumentations.
    Specifically designed for mammography images:
     - Resizes the longest side to 512 pixels while maintaining aspect ratio.
     - Pads the image to 512x512 if necessary, using constant padding with a value of 0 (black).
     - This preserve the original aspect ratio and prevents distortion, which is crucial for medical images.
     For instance, pectoral muscle still triangular in MLO.
    """
    transform =  A.Compose([
        
        A.LongestMaxSize(max_size=512),
        A.PadIfNeeded(
            min_height=512,
            min_width=512,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0
        ),
        ToTensorV2()
    ])
    
    return transform