# %%
# pip install pandas numpy sklearn torchtext transformers matplotlib
# pip install torch==1.8.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
from model import BERT
config = Config()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"***Available Device: {device} ***")

# %%

train_iter,valid_iter= prepare_data(device,config.MAX_SEQ_LEN,config.batch_size,config.train_csv,config.test_csv)


def Train(model,
          optimizer,
          train_loader = train_iter,
          valid_loader = valid_iter,
          num_epochs = config.num_epochs,
          eval_every = len(train_iter) // 2,
          file_path = config.destination_folder,
          best_valid_loss = float("Inf")):
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    running_acc = 0
    acc_list = []

    y_pred = []
    y_true = []
    
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
                    y_pred = []
                    y_true = []
                    for ( labels, A,B,C,D), _ in valid_loader:
                        labels = labels.type(torch.LongTensor)           
                        labels = labels.to(device)

                        model_input = torch.stack([A,B,C,D],1)
                        model_input = model_input.type(torch.LongTensor).to(device)
                        
                        output = model(model_input, labels)
                        loss, val_output = output

                        y_pred.extend(torch.argmax(val_output, 1).tolist())
                        y_true.extend(labels.tolist())
                        
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                running_acc = accuracy_score(y_true,y_pred)
                acc_list.append(running_acc)
                print(f"Validation Accuracy: {running_acc}")

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
                # if running_acc >= acc_list[-2]
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' + 'model.pt', model, best_valid_loss)
                    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    
    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')
    
# %%
model = BERT().to(device)
Train(model=model, optimizer = optim.Adam(model.parameters(),lr=1e-5))
# %%
