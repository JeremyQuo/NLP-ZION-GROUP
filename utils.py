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
from num2words import num2words

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
    # print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model,device):
    if load_path==None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


import pandas as pd
from gensim import corpora, models, similarities
from nltk.tokenize import word_tokenize
import numpy as np
import nltk
# nltk.download('punkt')

def compare(options,query,argmax=True):
    
    options = [word_tokenize(t.lower()) for t in  options] #lower + tokenise
    #options = [rmv_stwd(opt) for opt in options] 
    
    dictionary = corpora.Dictionary(options)
    feature_cnt = len(dictionary.token2id)
    corpus = [dictionary.doc2bow(opt) for opt in options]
    tfidf = models.TfidfModel(corpus) 

    kw_vector = dictionary.doc2bow(word_tokenize(query.lower()))
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features = feature_cnt)
    sim = index[tfidf[kw_vector]]
    if argmax:
        return np.argmax(sim)
    return sim


def convert_num_to_words(utterance):
      utterance = ' '.join([num2words(i) if i.isdigit() else i for i in utterance.split()])
      return utterance

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

    A = convert_num_to_words(A)
    B = convert_num_to_words(B)
    C = convert_num_to_words(C)
    D = convert_num_to_words(D)

    encoding = self.tokenizer(
        [A,B,C,D],
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        # pad_to_max_length=True,
        padding='max_length',
        truncation=True,
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






def mapDf(train_df_std,train_df,n):

    extracted_art = []
    for art,ques in zip(train_df_std['article'],train_df_std['question']):
        sents = art.split('.')
        arr = compare(sents,ques,False)
        idx = (-arr).argsort()[:n]
        idx = np.append(idx,0) if 0 not in idx else idx
        idx = sorted(idx)
        extracted_art.append( '.'.join([sents[_id] for _id in idx]) + '.')
    train_df_std['extracted_art'] = extracted_art
    
    for i in ["A","B","C","D"]:
        col = f"concat_{i}"
        train_df[col] = train_df_std['extracted_art'] + '[SEP]' +train_df_std['question']+ '[SEP]' +train_df_std[i]
    return train_df



def prepare_data(device,MAX_SEQ_LEN,batch_size,train_csv,test_csv="",n=5):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_df_std = pd.read_csv(train_csv,index_col=0)
    test_df_std = pd.read_csv(test_csv,index_col=0)

    le = preprocessing.LabelEncoder()

    train_df = pd.DataFrame({})
    train_df["labels"] = le.fit_transform(train_df_std['answer'])
    test_df = pd.DataFrame({})
    test_df['labels'] = le.transform(test_df_std['answer'])

    df_train = mapDf(train_df_std,train_df,n)
    df_test = mapDf(test_df_std,test_df,n)


    # df_train, df_test = train_test_split(
    #     train_df,
    #     test_size=0.2
    # )

 

    train_iter = create_data_loader(df_train.reset_index(drop=True), tokenizer, MAX_SEQ_LEN, batch_size)
    test_iter = create_data_loader(df_test.reset_index(drop=True), tokenizer, MAX_SEQ_LEN, batch_size)

    return train_iter,test_iter

