import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# Preliminaries
#import torchtext
#from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
#from torchtext.legacy.data import Dataset, Example

# Models
import torch.nn as nn
import torch.optim as optim
import torch
from transformers import BertConfig, BertTokenizer,CONFIG_NAME, WEIGHTS_NAME,BertForMultipleChoice
from torch.nn.modules import Softmax
from torch.utils.data import Dataset, DataLoader

def save_checkpoint(save_path, model, valid_loss):
    if save_path == None:
        return
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model,device):
    if load_path==None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']

class MCDataset(Dataset):
  def __init__(self, A,B,C,D, targets, tokenizer, max_len):
    self.A = A
    self.B = B
    self.C = C
    self.D = D
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  def __len__(self):
    return len(self.targets)

  def __getitem__(self, item):
    A = str(self.A[item])
    B = str(self.B[item])
    C = str(self.C[item])
    D = str(self.D[item])
    target = self.targets[item]

    encoding = self.tokenizer(
        [A,B,C,D],
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
        )

    return {
      'A': A,
      'B':B,
      'C':C,
      'D':D,
      'input_ids': encoding['input_ids'],
      'attention_mask': encoding['attention_mask'],
      'targets': torch.tensor(target, dtype=torch.long)
    }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = MCDataset(
        A=df.concat_A,
        B=df.concat_B,
        C=df.concat_C,
        D=df.concat_D,
        targets=df.labels.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0
    )


def prepare_data(device,MAX_SEQ_LEN,batch_size,train_csv,test_csv=""):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_df_std = pd.read_csv(train_csv,index_col=0)
    test_df_std = pd.read_csv(test_csv,index_col=0)

    le = preprocessing.LabelEncoder()


    train_df_std = pd.read_csv(train_csv,index_col=0)
    train_df = pd.DataFrame({})
    train_df["labels"] = le.fit_transform(train_df_std['answer'])
    for i in ["A","B","C","D"]:
        col = f"concat_{i}"
        train_df[col] = train_df_std['article']+ '[SEP]' +train_df_std['question']+ '[SEP]' +train_df_std[i]
        #train_df[col] = train_df_std['question']+ '[SEP]' +train_df_std[i] + '[SEP]' +train_df_std['article']

    df_train, df_test = train_test_split(
        train_df,
        test_size=0.2
    )

 

    train_iter = create_data_loader(df_train.reset_index(drop=True), tokenizer, MAX_SEQ_LEN, batch_size)
    test_iter = create_data_loader(df_test.reset_index(drop=True), tokenizer, MAX_SEQ_LEN, batch_size)

    return train_iter,test_iter

