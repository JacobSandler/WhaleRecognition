import argparse
import os
import pandas as pd


import constants
from datasets.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train_updated import EvaluationDataset, starting_train


SUMMARIES_PATH = "training_summaries"


def main():
    # Get command line arguments
    args = parse_arguments()
    hyperparameters = {"epochs": args.epochs, "batch_size": args.batch_size, "margin": constants.MARGIN}

    # Create path for training summaries
    summary_path = None
    if args.logdir is not None:
        summary_path = f"{SUMMARIES_PATH}/{args.logdir}"
        os.makedirs(summary_path, exist_ok=True)

    # TODO: Add GPU support. This line of code might be helpful.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Summary path:", summary_path)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)

    # Initalize dataset and model. Then train the model!
    dataset = StartingDataset(constants.CSV_PATH, constants.IMAGE_DIR, constants.IMAGE_SIZE, constants.PERCENT_TRAIN)
    train_dataset = dataset.train_set
    val_dataset = dataset.val_set
    

    data = pd.read_csv(constants.CSV_PATH)
    #data = data.sample(frac=1) # Shuffle data
    train_eval_data = data.iloc[:int(constants.PERCENT_TRAIN*len(data))]
    test_eval_data = data.iloc[int(constants.PERCENT_TRAIN*len(data)) + 1:]
    train_eval_dataset = EvaluationDataset(data=train_eval_data , crop_info_path=constants.BBOX_PATH, image_folder="./datasets/train", train=True, drop_duplicate_whales=True)
    test_eval_dataset = EvaluationDataset(data=test_eval_data , crop_info_path=constants.BBOX_PATH, image_folder="./datasets/train", train=False, drop_duplicate_whales=False)

    model = StartingNetwork()
    starting_train(
        train_dataset=train_dataset,
        test_eval_dataset=test_eval_dataset,
        train_eval_dataset=train_eval_dataset,
        model=model,
        hyperparameters=hyperparameters,
        summary_path=summary_path,
        n_eval=constants.N_EVAL
    )

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=constants.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=constants.BATCH_SIZE)
    parser.add_argument("--n_eval", type=int, default=constants.N_EVAL)
    parser.add_argument("--logdir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
