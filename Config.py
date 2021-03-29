class Config:
    def __init__(self):
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.destination_folder = "Model"
        self.num_epochs = 6

        self.train_csv = 'std_data/Race/middle/train.csv'
        self.MAX_SEQ_LEN = 128
        self.batch_size = 16

        print(f"[-] Training Dataset: {self.train_csv}")