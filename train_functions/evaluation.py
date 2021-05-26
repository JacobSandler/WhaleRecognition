import os

import torch
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image


"""
Two things to pay attention to:
    1. Modify the code in EvaluationDataset to make sure images are loaded correctly.
    2. Pay attention to when drop_duplicate_whales is set to True or False.
"""


def example_usage():
    """
    This code won't work, so don't run this!
    """

    data = pd.read_csv("path/to/train.csv")
    data = data.sample(frac=1) # Shuffle data
    train_data = data.iloc[:int(0.9*len(data))]
    test_data = data.iloc[int(0.9*len(data)) + 1:]

    train_dataset = EvaluationDataset(
        train_data,
        "path/to/bboxs.csv",
        "path/to/train",
        train=True,
        drop_duplicate_whales=True, # If you set this to True, your evaluation accuracy will be lower!!
                                    # If you set this to False, evaluate() will take longer!!
                                    # Recommendation: set this to True during training, and when you're done,
                                    # create a new dataset with drop_duplicate_whales=False to get a final
                                    # evaluation metric.
    )
    train_dataset.to('cuda')

    test_dataset = EvaluationDataset(
        test_data,
        "path/to/bboxs.csv",
        "path/to/train",
        train=False,
        drop_duplicate_whales=False,
    )
    test_dataset.to('cuda')

    train_eval_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
    test_eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)




    model = None # Initialize your model...
    # and train your model a bit...
    for epoch in range(5):
        # do stuff
        accuracy = evaluate(train_eval_loader, test_eval_loader, model)
        # Log accuracy somewhere



    # Probably easier to just run the below code in a separate Colab cell when you're done
    final_accuracy = evaluate(train_eval_loader, test_eval_loader, model, final=True)


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
        self.crop_info = pd.read_csv(crop_info_path, index_col="Image")
        self.image_folder = image_folder

        self.device = None

        if train:
            self.data = self.data[self.data.Id != "new_whale"]
        if drop_duplicate_whales:
            self.data = self.data.drop_duplicates(subset="Id")

    def to(self, device):
        self.device = device
        return self

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

        image = image.to(self.device)

        return image, whale_id

    def __len__(self):
        return len(self.data)


def evaluate(train_dataset, val_dataset, model, hyperparameters, n_eval, summary_path, final=False):
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
