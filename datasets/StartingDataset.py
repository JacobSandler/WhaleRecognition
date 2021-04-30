import torch
import torchvision
import pandas as pd

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, csv_path, image_dir, image_size):
        self.images_frame = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.image_size = image_size

        #load in this one image at a time every epoch
    def __getitem__(self, index):
        """if torch.is_tensor(index):
            index = index.tolist"""
        
        label = self.images_frame.iloc[index, 1]
        image_path = self.image_dir + self.images_frame.iloc[index, 0]
        image = torchvision.io.read_image(image_path)

        if(len(image)==3):
            image = torchvision.transforms.Grayscale().forward(image)
        image = torchvision.transforms.Resize(self.image_size).forward(image)
        image = image.float()
        
        return image, label #map the label to an int
    
    def __len__(self):
        return len(self.images_frame)

