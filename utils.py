import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Preliminaries
import torchtext
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
from torchtext.legacy.data import Dataset, Example

# Models
import torch.nn as nn
import torch.optim as optim
import torch
from transformers import BertConfig, BertTokenizer,CONFIG_NAME, WEIGHTS_NAME,BertForMultipleChoice
from torch.nn.modules import Softmax
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

def prepare_data(device,MAX_SEQ_LEN,batch_size,train_csv,test_csv=""):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Model parameter
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    # Fields
    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                    fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
    fields = [('label', label_field), ('input_ids_A', text_field),('input_ids_B', text_field),('input_ids_C', text_field),('input_ids_D', text_field)]


    class DataFrameDataset(Dataset):
        def __init__(self, df: pd.DataFrame, fields: list):
            super(DataFrameDataset, self).__init__(
                [
                    Example.fromlist(list(r), fields) 
                    for i, r in df.iterrows()
                ], 
                fields
            )

    le = preprocessing.LabelEncoder()
    train_df_std = pd.read_csv(train_csv,index_col=0)
    train_df = pd.DataFrame({})
    train_df["labels"] = le.fit_transform(train_df_std['answer'])
    for i in ["A","B","C","D"]:
        col = f"input_ids_{i}"
        train_df[col] = train_df_std['article']+ '[SEP]' +train_df_std['question']+ '[SEP]' +train_df_std[i]

    train, valid = DataFrameDataset(
        df=train_df, 
        fields=fields
    ).split()

    train_iter = BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.input_ids_B),
                                device=device, train=True, sort=True, sort_within_batch=True)
    valid_iter = BucketIterator(valid, batch_size=batch_size, sort_key=lambda x: len(x.input_ids_B),
                                device=device, train=True, sort=True, sort_within_batch=True)
    #test_iter = Iterator(test, batch_size=batch_size, device=device, train=False, shuffle=False, sort=False)

    return train_iter,valid_iter

