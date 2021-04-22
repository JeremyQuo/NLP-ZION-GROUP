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
class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        options_name = "bert-base-uncased"
        self.bert = BertForMultipleChoice.from_pretrained(options_name,num_labels=4)  
        # self.drop = nn.Dropout(p=0.3)
        # self.out = nn.Linear(self.bert.config.hidden_size, 4)

    def forward(self, model_input, labels, attention_mask=None):
        loss,x = self.bert(model_input, labels=labels,attention_mask=attention_mask)[:2]
        # x = self.drop(x)
        # x = self.out(x)
        return loss, x