from __future__ import absolute_import

import argparse
import csv
import logging
import os
import random
import sys
from io import open

import numpy as np
import torch

import glob
import json

import pandas as pd
import numpy as np


# remove redundancy word and make it can be open by pandas
def modify_data(file):
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.replace(';Work', '\tWork')
            line = line.replace('Author: ', '')
            line = line.replace('Work Time(s): ', '')
            file_data += line
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)


# get all answer of the question
def get_answer(file):
    output_list = pd.read_csv(file, sep='	', header=None)
    return output_list

# separate each 4 question to 4 columns
def separate_question(input_list):
    output_list = []
    if input_list[0].find('one:') != -1:
        output_list.append(input_list[0][0:3])
        output_list.append(input_list[0][5:])
    elif input_list[0].find('multiple:') != -1:
        output_list.append(input_list[0][0:8])
        output_list.append(input_list[0][10:])
    # print(input_list[1:])
    output_list.extend(input_list[1:])
    return output_list


def transfer(data_file, answer_file):
    modify_data(data_file)
    data = pd.read_csv(data_file, sep='\t', header=None)
    answer_data = get_answer(answer_file)
    df_new = pd.DataFrame()
    for index, row in data.iterrows():
        new_columns = row[0:4].values
        for i in range(4):
            # print(row[4 + 5 * i: 4 + 5 * i + 5].values)
            temp_list = separate_question(row[4 + 5 * i: 4 + 5 * i + 5].values)
            final_columns = np.append(new_columns, temp_list)
            final_columns = np.append(final_columns, answer_data[i][index])
            final_columns = final_columns.tolist()
            df_new = df_new.append([final_columns], ignore_index=True)
    df_new.columns = ['id', 'author', 'work_times', 'article', 'question_type', 'question', 'A', 'B', 'C', 'D',
                      'answer']
    return df_new




class MCTest(object):
    """We are going to train race dataset with bert."""
    def __init__(self,
                 id,
                 article,
                 question,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 answer = None):
        self.id = id
        self.question = question
        self.article = article
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
        ]
        self.answer = answer

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            "id: {}".format(self.id),
            "article_sentence: {}".format(self.article),
            "question: {}".format(self.question),
            "ending_0: {}".format(self.endings[0]),
            "ending_1: {}".format(self.endings[1]),
            "ending_2: {}".format(self.endings[2]),
            "ending_3: {}".format(self.endings[3]),
        ]

        if self.answer is not None:
            l.append("answer: {}".format(self.answer))

        return ", ".join(l)

class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


# data_grade is a list of grades of the dataset.
# for example: ["high","middle"]
def read_MCtest(df):
    samples = []
    for item in df.iterrows():
        id = item[1][0]
        article = item[1][3]
        question = item[1][5]
        ending_0 = item[1][6]
        ending_1 = item[1][7]
        ending_2 = item[1][8]
        ending_3 = item[1][9]
        answer = item[1][10]
        # print(id, article, question, ending_0, ending_1, ending_2, ending_3, answer)

        # break

        samples.append(MCTest(id, article, question, ending_0, ending_1, ending_2, ending_3, answer))


    return samples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # RACE is a multiple choice task like Swag. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given Race example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    print(len(examples),examples[0])
    for example_index, example in enumerate(examples):
        context_tokens = tokenizer.tokenize(example.article)
        start_ending_tokens = tokenizer.tokenize(example.question)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]
            ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(start_ending_tokens)) +[2] * (len(tokenizer.tokenize(ending)) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.answer
        if example_index < 5:
            logger.info("*** Example ***")
            logger.info("id: {}".format(example.id))
            for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("tokens: {}".format(' '.join(tokens)))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
                logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
            if is_training:
                logger.info("label: {}".format(label))
        if (example_index%5000 ==0): print(example_index)	
        features.append(
            InputFeatures(
                example_id = example.id,
                choices_features = choices_features,
                label = label
            )
        )

    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
            
            
def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    #print(outputs,outputs == labels)
    return np.sum(outputs == labels)

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]