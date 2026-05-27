#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from dataset import MammographyDataset, get_transform


PROJECT_ROOT = Path(__file__).resolve().parent.parent

split_path = PROJECT_ROOT / "raw_data" / "splits" / "split.csv"

df = pd.read_csv(split_path)

train_df = df[df["split"] == "train"]

dataset = MammographyDataset(train_df, transform=get_transform())

image, label = dataset[1]

print(image.shape)
print(label)

plt.imshow(image.squeeze(), cmap="gray")

plt.title(f'Label: {"CC" if label == 0 else "MLO"} ')

plt.axis("off")
plt.show()