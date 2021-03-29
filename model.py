# pip install pandas numpy sklearn torchtext transformers
# pip install torch==1.8.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
import pandas as pd
import numpy as np
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
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


from utils import *
from Config import Config
config = Config()

device = config.device
print(f"***Available Device: {device} ***")

train_iter,valid_iter = prepare_data(device)

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