from constants import BATCH_SIZE, PERCENT_TRAIN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard
import pandas as pd
import numpy as np
from pytorch_metric_learning import losses, miners
import os
from os import path
import torchvision.transforms as transforms
from PIL import Image



def starting_train(
    train_dataset, test_dataset, model, hyperparameters, summary_path
):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
        summary_path:    Path where Tensorboard summaries are located.
    """

    save_path = './model.pt'

    if(path.exists(save_path)):
        model.load_state_dict(torch.load(save_path))

    #data = pd.read_csv(train_path)
    #data = data.sample(frac=1) # Shuffle data
    train_eval_data = test_dataset.iloc[:int(PERCENT_TRAIN*len(test_dataset))]
    test_eval_data = test_dataset[int(PERCENT_TRAIN*len(test_dataset)) + 1:]
    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]



    train_eval_loader = torch.utils.data.DataLoader(train_eval_data, batch_size=BATCH_SIZE)
    test_eval_loader = torch.utils.data.DataLoader(test_eval_data, batch_size=BATCH_SIZE)



    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    #val_loader = torch.utils.data.DataLoader(
        #val_dataset, batch_size=batch_size, shuffle=True
    #)

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = losses.TripletMarginLoss(margin=hyperparameters["margin"])
    miner = miners.BatchEasyHardMiner(pos_strategy='all', neg_strategy='hard')

    # Initialize summary writer (for logging)
    if summary_path is not None:
        writer = torch.utils.tensorboard.SummaryWriter(summary_path)

    step = 0
    correct = 0
    total = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for i, batch in enumerate(train_loader):

            # TODO: unload each "item grouping" from batches
            # Likely can just use: torch.cat(batch) 

            print(f"\rIteration {i + 1} of {len(train_loader)} ...", end="")
            images, labels = batch

            images = torch.cat(list(images))
            labels = torch.cat(list(labels))

            print(images.size())
            print(labels.size())

            labels = torch.tensor(labels)

            optimizer.zero_grad()
            embeddings = model(images) # images is a batch of images
            hard_triplets = miner(embeddings, labels)
            loss = loss_fn(embeddings, labels, hard_triplets)
            loss.backward()
            optimizer.step()

            # Periodically evaluate our model + log to Tensorboard (ADD IN TRAINING EVALUTATIONS LATER)
            #if step % n_eval == 0:
                #if summary_path is not None:
                    #writer.add_scalar('train_loss', loss, global_step=step)
                #if(writer.init):
                
                #train_accuracy = evaluate(train_eval_loader, test_eval_loader, model)


                # TODO:
                # Compute training accuracy.
                # Log the results to Tensorboard.
                #images, labels = batch

                #outputs = model(images)
                #loss = loss_fn(outputs, labels)
                #predictions = torch.argmax(outputs, dim=1)
                    
            if(path.exists(save_path)):
                torch.save(model.state_dict(), save_path)
        
            step += 1

        # TODO:
        # Log the results to Tensorboard.
        # Don't forget to turn off gradient calculations!
        val_accuracy = evaluate(train_eval_loader, test_eval_loader, model)
        #print("loss: " + val_loss)
        print("loss: " + val_accuracy)
        
        if summary_path is not None:
            writer.add_scalar('train_loss', loss, global_step=step)
            #writer.add_scalar('val_loss', val_loss, global_step=step)
            writer.add_scalar('val_accuracy', val_accuracy, global_step=step)
            #writer.add_scalar('train_accuracy', train_accuracy, global_step=step)



def evaluate(train_loader, test_loader, model, final=False):
    """
    Evaluates model performance. Both `train_loader` and `test_loader` should be
    instances of `EvaluationDataset`.
    """

    model.eval()

    """
    STEP 1. Compute TRAIN SET embeddings! We will use these embeddings to
    compare test images to.
    """

    # train_whale_ids[i] is the whale id corresponding to train_embeddings[i]
    train_embeddings = []
    train_whale_ids = []
    with torch.no_grad():  # Parentheses are important :)
        for batch in train_loader:
            print('x')
            images, whale_ids = batch

            images = torch.cat(list(images))
            whale_ids = torch.cat(list(whale_ids))

            batch_embeddings = model.forward(images)

            train_embeddings += list(batch_embeddings)
            train_whale_ids += list(whale_ids)

    # This will convert a list to a tensor
    train_embeddings = torch.stack(train_embeddings)

    """
    STEP 2. Compute TEST SET embeddings!
    """

    test_embeddings = []
    test_whale_ids = []
    with torch.no_grad():
        for batch in test_loader:
            print('y')
            images, whale_ids = batch

            images = torch.cat(list(images))
            whale_ids = torch.cat(list(whale_ids))

            batch_embeddings = model.forward(images)

            test_embeddings += list(batch_embeddings)
            test_whale_ids += list(whale_ids)

    """
    STEP 3. Compute the model's ACCURACY!
    """
    if final:
        accuracy = compute_final_accuracy(
            train_embeddings, train_whale_ids, test_embeddings, test_whale_ids
        )
    else:
        accuracy = compute_accuracy(
            train_embeddings, train_whale_ids, test_embeddings, test_whale_ids
        )

    model.train()

    return accuracy




def compute_final_accuracy(train_embeddings, train_ids, test_embeddings, test_ids):
    """
    Same as compute_accuracy, but will identify the optimal threshold for predicting
    "new_whale".
    """

    """
    NOTE: You may have to modify some of the values below depending on your choice of
    triplet loss margin (and other hyperparameters). Currently, the thresholds being
    tried are between 0.2 and 0.65.
    """

    """
    NOTE: This is actually bad practice because we are using the "test" set to determine
    what the threshold for predicting "new_whale" is. It is actually best to divide the
    dataset into train/validation/test sets, and use the validation set for these
    purposes, only touching the test set to get a final performance metric. But in
    practice, the threshold found would be similar even if there was a validation set.
    """

    best_threshold, best_accuracy = None, 0
    for i in range(10):
        # Threshold will range from 0.2 to 0.65
        threshold = 0.2 + 0.05 * i
        accuracy = compute_accuracy(
            train_embeddings, train_ids, test_embeddings, test_ids, threshold
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    print(f"Best threshold: {best_threshold}, Best accuracy: {best_accuracy}")
    return best_accuracy


def compute_accuracy(
    train_embeddings, train_ids, test_embeddings, test_ids, threshold=1000
):
    """
    Computes test accuracy. If the distance between a test embedding and every train
    embedding is at least `threshold`, then "new_whale" will be predicted.
    """

    correct, total = 0, 0

    for whale_id, embedding in zip(test_ids, test_embeddings):
        # This line will compute the distance between the test embedding and EVERY train
        # embedding
        distances = torch.norm(train_embeddings - embedding.view((1, 64)), dim=1)

        min_index = torch.argmin(distances)
        prediction = (
            "new_whale" if distances[min_index] > threshold else train_ids[min_index]
        )
        if prediction == whale_id:
            correct += 1
        total += 1

    return correct / total

class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        crop_info_path,
        image_folder,
        train=True,
        drop_duplicate_whales=False,
    ):
        self.data = data
        self.data = pd.read_csv(data)
        self.crop_info = pd.read_csv(crop_info_path, index_col="Image")
        self.image_folder = image_folder

        #self.device = None

        if train:
            self.data = self.data[self.data.Id != "new_whale"]
        if drop_duplicate_whales:
            self.data = self.data.drop_duplicates(subset="Id")

    #def to(self, device):
    #    self.device = device
    #    return self

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_file, whale_id = row.Image, row.Id

        """
        You may want to modify the code STARTING HERE...
        """
        bbox = self.crop_info.loc[row.Image]
        image = Image.open(os.path.join(self.image_folder, image_file))
        image = image.convert('P') # Maybe change this
        image = image.crop((bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]))

        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)), # Probably change this
                transforms.ToTensor(),
                transforms.Normalize((0.485,0.456,0.406,), (0.229, 0.224, 0.225,)), # and maybe this too
            ]
        )
        image = preprocess(image)
        """
        ... and ENDING HERE. In particular, we converted the image to grayscale with a
        size of 224x448. You probably want to change that.
        """

        #image = image.to(self.device)

        return image, whale_id

    def __len__(self):
        return len(self.data)