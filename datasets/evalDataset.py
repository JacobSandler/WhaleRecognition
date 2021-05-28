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