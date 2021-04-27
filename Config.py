class Config:
    def __init__(self):
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.source_folder = "std_data"
        self.destination_folder = "Model"
        self.num_epochs = 10



        self.train_csv = 'std_data/MCTest/Mc500/mc500.train.csv'
        self.test_csv = 'std_data/MCTest/Mc500/mc500.test.csv'
        self.MAX_SEQ_LEN = 128
        self.batch_size = 16

        self.n = 1 # Extract top related sentences

        print(f"[-] Training Dataset: {self.train_csv}")