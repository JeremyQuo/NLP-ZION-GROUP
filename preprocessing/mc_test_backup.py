#!/usr/bin/env python
# coding: utf-8

# In[1]:


f = open('mc160.test.txt')


# In[2]:


lines = f.read()


# In[3]:


parts = lines.split('***************************************************')


# In[4]:


story_name = []
author = []
worktime = []
story = []
one_or_multi = []
question = []
A = []
B = []
C = []
D = []
answer = []
for part in parts[1:]:
    stroy_string = ''
    lines = part.splitlines()
    for line in lines:
        if 'Story:' in line:
            story_name.append(line.split(':            ')[-1])
            story_name.append(line.split(':            ')[-1])
            story_name.append(line.split(':            ')[-1])
            story_name.append(line.split(':            ')[-1])
        elif 'Author:' in line:
            author.append(line.split(':           ')[-1])
            author.append(line.split(':           ')[-1])
            author.append(line.split(':           ')[-1])
            author.append(line.split(':           ')[-1])
        elif 'Work Time(s):' in line:
            worktime.append(line.split(':     ')[-1])
            worktime.append(line.split(':     ')[-1])
            worktime.append(line.split(':     ')[-1])
            worktime.append(line.split(':     ')[-1])
        elif '1: ' in line:
            if 'multiple' in line:
                one_or_multi.append('multiple')
                question.append(line.split(': ')[-1])
            elif 'one' in line:
                one_or_multi.append('one')
                question.append(line.split(': ')[-1])
            else:
                pass
        elif '2: ' in line:
            if 'multiple' in line:
                one_or_multi.append('multiple')
                question.append(line.split(': ')[-1])
            elif 'one' in line:
                one_or_multi.append('one')
                question.append(line.split(': ')[-1])
            else:
                pass
        elif '3: ' in line:
            if 'multiple' in line:
                one_or_multi.append('multiple')
                question.append(line.split(': ')[-1])
            elif 'one' in line:
                one_or_multi.append('one')
                question.append(line.split(': ')[-1])
            else:
                pass
        elif '4: ' in line:
            if 'multiple' in line:
                one_or_multi.append('multiple')
                question.append(line.split(': ')[-1])
            elif 'one' in line:
                one_or_multi.append('one')
                question.append(line.split(': ')[-1])
            else:
                pass
        elif 'A) ' in line:
            A.append(line.split(') ')[-1])
        elif 'B) ' in line:
            B.append(line.split(') ')[-1])
        elif 'C) ' in line:
            C.append(line.split(') ')[-1])
        elif 'D) ' in line:
            D.append(line.split(') ')[-1])
        else:
            stroy_string += line
        if '*' in line:
            answer.append(line.split(') ')[-1])
    story.append(stroy_string)
    story.append(stroy_string)
    story.append(stroy_string)
    story.append(stroy_string)
        
    


# In[5]:


import pandas as pd

#任意的多组列表 

dataframe = pd.DataFrame({'Story':story_name,'Author':author, 'WorkTime':worktime, 'Article':story, 
                          'OneOrMulti':one_or_multi, 'Question':question, 'A':A, 'B':B,
                          'C':C, 'D':D, 'Answer':answer})

dataframe.to_csv("mc160.test.csv",index=False,sep=',')


# In[ ]:


dataframe.head()


# In[ ]:




