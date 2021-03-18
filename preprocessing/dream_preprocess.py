#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json


with open('./train.json') as f:
    json_list = json.load(f)
    conversation = [' '.join(i[0]) for i in json_list]
    question = [i[1][0]['question'] for i in json_list]
    choice1 = [i[1][0]['choice'][0] for i in json_list]
    choice2 = [i[1][0]['choice'][1] for i in json_list]
    choice3 = [i[1][0]['choice'][2] for i in json_list]
    answer = [i[1][0]['answer'] for i in json_list]
    number = [i[2] for i in json_list]


import pandas as pd

dataframe = pd.DataFrame({'Number':number,'Conversation':conversation,'Question':question, 'Choice1':choice1, 
                          'Choice2':choice2, 'Choice3':choice3, 'Answer':answer})

dataframe.to_csv("train.csv",index=False,sep=',')





