import pandas as pd
import numpy as np
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
# Preliminaries
import torchtext
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from torchtext.data import Dataset, Example

# Models
import torch.nn as nn
import torch.optim as optim
import torch
from transformers import BertConfig, BertTokenizer,CONFIG_NAME, WEIGHTS_NAME,BertForMultipleChoice
from torch.nn.modules import Softmax

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']



tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# Model parameter
MAX_SEQ_LEN = 128
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


# TabularDataset
#train, valid, test = TabularDataset.splits(path="./", train='train.csv', validation='val.csv',
 #                                          test='test.csv', format='CSV', fields=fields, skip_header=True)

train_df_std = pd.read_csv('std_data/MCTest/mc160/mc160.train.csv',index_col=0)
train_df = pd.DataFrame({})
train_df["labels"] = le.fit_transform(train_df_std['answer'])
for i in ["A","B","C","D"]:
    col = f"input_ids_{i}"
    train_df[col] = train_df_std['article']+ '[SEP]' +train_df_std['question']+ '[SEP]' +train_df_std[i]

train, valid = DataFrameDataset(
    df=train_df, 
    fields=fields
).split()

train_iter = BucketIterator(train, batch_size=16, sort_key=lambda x: len(x.input_ids_B),
                            device=device, train=True, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid, batch_size=16, sort_key=lambda x: len(x.input_ids_B),
                            device=device, train=True, sort=True, sort_within_batch=True)
# test_iter = Iterator(test, batch_size=16, device=device, train=False, shuffle=False, sort=False)


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        options_name = "bert-base-uncased"
        self.encoder = BertForMultipleChoice.from_pretrained(options_name,num_labels=4)
    def forward(self, model_input, labels):
        enc_output = self.encoder(model_input, labels=labels)
        loss, text_fea = enc_output[:2]
        return loss, text_fea



destination_folder = "Model"
def Train(model,
          optimizer,
          criterion = nn.BCELoss(),
          train_loader = train_iter,
          valid_loader = valid_iter,
          num_epochs = 10,
          eval_every = len(train_iter) // 2,
          file_path = destination_folder,
          best_valid_loss = float("Inf")):
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (labels, A,B,C,D),_ in train_loader:
            labels = labels.type(torch.LongTensor)           
            labels = labels.to(device)

            model_input = torch.stack([A,B,C,D],1)
            model_input = model_input.type(torch.LongTensor).to(device)

            output = model(model_input, labels)
            loss, _ = output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    
                    # validation loop
                    for ( labels, A,B,C,D), _ in valid_loader:
                        labels = labels.type(torch.LongTensor)           
                        labels = labels.to(device)

                        model_input = torch.stack([A,B,C,D],1)
                        model_input = model_input.type(torch.LongTensor).to(device)
                        
                        output = model(model_input, labels)
                        loss, _ = output
                        
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
                
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' + 'model.pt', model, best_valid_loss)
                    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    
    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')
    
model = BERT().to(device)
Train(model=model, optimizer = optim.Adam(model.parameters(),lr=2e-5))