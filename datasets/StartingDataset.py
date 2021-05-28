from os import access
import torch
import torchvision
import pandas as pd
import random
import tensorflow as tf

class StartingDataset():
    def __init__(self, csv_path, image_dir, image_size, percent_train):
        self.images_frame = pd.read_csv(csv_path)
        #self.bbox_frame = pd.read_csv(bbox_path)
        self.labels = dict() 
        self.image_dir = image_dir
        self.image_size = image_size

        #Shuffle the dataframe (taken from stackoverflow, I'm assuming this works)
        self.images_frame = self.images_frame.sample(frac=1).reset_index(drop=True)

        # Convert labels to numeric ids
        ids = self.images_frame["Id"]
        ids.drop_duplicates(inplace=True)
        index = 0
        for id in ids:
            self.labels[id] = index
            index += 1

        #Splits the image frame into a frame for the train set and validation set
        train_len = int(len(self.images_frame)*percent_train)
        self.test_frame = self.images_frame.iloc[:,:train_len]
        self.val_frame = self.images_frame.iloc[:,:len(self.images_frame)-train_len]
        self.train_set = Dataset(self.test_frame, self.labels, image_dir, image_size)
        self.val_set= Dataset(self.val_frame, self.labels, image_dir, image_size)
    
class Dataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, images_frame, labels, image_dir, image_size):
        self.images_frame = images_frame
        self.image_dir = image_dir
        self.image_size = image_size
        self.labels = labels
        self.num_images = self.images_frame['Id'].size

        self.images_frame.sort_values('Id')

        #load in this one image at a time every epoch
    def __getitem__(self, index):
        """if torch.is_tensor(index):
            index = index.tolist"""
        
        indices = torch.tensor([])
        images = torch.tensor([])
        labels = torch.tensor([])

        index1 = index
        index2 = index + 1
        while (torch.numel(indices) < 2):
            if(index2 >= len(self.images_frame)):
                index1 = random.randrange(self.num_images)
                index2 = index1 + 1
            label1 = self.images_frame.iloc[index1, 1]
            label2 = self.images_frame.iloc[index2, 1]
            if (self.labels[label1] == self.labels[label2]):
                indices = torch.cat(tensors=(indices, torch.tensor([index1])))
                indices = torch.cat(tensors=(indices, torch.tensor([index2])))
                labels = torch.cat(tensors=(labels, torch.tensor([self.labels[label1]])))
                labels = torch.cat(tensors=(labels, torch.tensor([self.labels[label2]])))
            else:
                index1 = random.randrange(self.num_images)
                index2 = index1 + 1

        index3 = random.randrange(self.num_images)
        index4 = random.randrange(self.num_images)
        label3 = self.images_frame.iloc[index3, 1]
        label4 = self.images_frame.iloc[index4, 1]

        indices = torch.cat(tensors=(indices, torch.tensor([index3])))
        indices = torch.cat(tensors=(indices, torch.tensor([index4])))
        labels = torch.cat(tensors=(labels, torch.tensor([self.labels[label3]])))
        labels = torch.cat(tensors=(labels, torch.tensor([self.labels[label4]])))    

        first = True
        for index in indices:
            #current_label = self.images_frame.iloc[int(index.item()), 1]
            image_path = self.image_dir + self.images_frame.iloc[int(index.item()), 0]
            image = torchvision.io.read_image(image_path)

            """
            boxes = self.bbox_frame.loc[self.bbox_frame['Images'] == current_label]

            start_x = boxes[1]
            start_y = boxes[2]
            end_x = boxes[3]
            end_y = boxes[4]

            image = image[start_y:end_y, start_x:end_x]
            """

            if(len(image)==1):
                image = image.repeat(3,1,1)
            image = torchvision.transforms.Resize(self.image_size).forward(image)
            image = image.float()
            image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).forward(image)

            image = torch.unsqueeze(image, dim=0)

            if first:   
                images = image
                first = False
            else:
                images = torch.cat(tensors=(images, image))
                
        return images, labels #map the label to an int
    
    def __len__(self):
        return len(self.images_frame)
                            
