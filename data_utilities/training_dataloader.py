import os
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
 
import torch.nn as nn
from torchvision.io import read_image, ImageReadMode

 
class HouseTrainingDataset(Dataset):
    def __init__(self, images_directory, image_price_csv_path):
        # Directory to where the images are loaded
        self.images_directory = images_directory

        # Load the image filenames and their actual prices
        df = pd.read_csv(image_price_csv_path)
        self.filenames = df.iloc[:, 0].astype(str).tolist()
        self.prices = df.iloc[:, 1].astype(float).tolist()

        # Define the transform that will be applied to the images
        # when initializing, 
        self.transform =  nn.Identity()
 
    def __len__(self):
        return len(self.filenames)
 
    def __getitem__(self, index):
        # obtain an image when given a filename
        image_filename = self.filenames[index]
        
        # uint8 tensor, (3, H, W)
        image = read_image(os.path.join(self.images_directory, image_filename), 
                         mode=ImageReadMode.RGB)
        
        # Apply the transform to the image
        image = self.transform(image)
        
        # Get the corresponding price of the house
        price = torch.tensor(self.prices[index], dtype=torch.float32)

        return image, price, image_filename

    def set_transform(self, transform):
        # Helper method to change the transform
        self.transform = transform
        
    def get_dataloader(self, batch_size=1, shuffle=False, **kwargs):
        # images are still variable-size here (no resize transform yet), so batch_size=1
        # until a Resize is added — the default collate can't stack differently-shaped tensors.
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)