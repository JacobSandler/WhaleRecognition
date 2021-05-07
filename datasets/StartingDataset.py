import torch
import csv


class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self):
        pass

        #load in this one image at a time every epoch
    def __getitem__(self, index):
        #inputs = torch.zeros([3, 224, 224])
        #label = 0
        inputs = self.image[index]
        #input_crop  = inputs[ystart:ystop, xstart:xstop] <-- y_start and y_stop dictated by the bounding box IDs
        label = self.labels[index]
        


        return inputs, label

    def __len__(self):
        return 10000
