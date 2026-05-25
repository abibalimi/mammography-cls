#!/usr/bin/env python3

import torch
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt



IMAGE_DIR = Path('raw_data/data')
IMAGE_ANNOTATIONS_TRAIN = IMAGE_DIR / f'train_image_annotations.csv'
IMAGE_ANNOTATIONS_VAL = IMAGE_DIR / f'val_image_annotations.csv'
IMAGE_ANNOTATIONS_TEST = IMAGE_DIR / f'test_image_annotations.csv'


class MammogramDataset(Dataset):
    """Retrieves dataset’s features and labels one sample at a time."""
    
    def __init__(self, image_directory, annotations_file, transform=None, target_transform=None):
        """
        Arguments:
            image_directory (string): Directory with all the images.
            annotations_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_transform (callable, optional): Optional transform to be applied on a label.
        """
        self.image_directory = image_directory
        self.image_labels = pd.read_csv(annotations_file)
        self.transform = transform  # padding etc, check out within pytorch
        self.target_transform = target_transform
      
    
    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.image_labels)
     
    
    def __getitem__(self, index):
        """loads and returns a sample from the dataset at the given index 'index'."""
        image_path =  Path(self.image_directory / self.image_labels.iloc[index, 0])
        image = read_image(image_path)  # converts that to a tensor
        label = self.image_labels.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label



def show_images(image_set):
    """Shows cols * rows mammograms selected randomly from the dataset."""
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 2, 2
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(image_set), size=(1,)).item()
        img, label = image_set[sample_idx]
        
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze())#, cmap="gray")
    plt.show()


def show_image_and_label(split_dataloader):
    """Display image and label."""
    image, labels = next(iter(split_dataloader))
    print(f"Feature batch shape: {image.size()}")
    print(f"Labels batch shape: {labels.size()}")
    img = image[0].squeeze()
    label = labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")



if __name__ == "__main__":
    val_data = MammogramDataset(image_directory=IMAGE_DIR / f'validation', 
                                     annotations_file=IMAGE_ANNOTATIONS_VAL)
    show_images(val_data)
    
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    show_image_and_label(val_dataloader)