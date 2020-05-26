from re import sub

import torch

import csv
import itertools
import random
from random import shuffle

import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split as split_data
import numpy as np
from tqdm import tqdm


class Read_Data:
    def __init__(self, column_names, data_name, data_file, train_ratio=0.8, test_val_ratio=0.5,
                 med_flag=2,
                 weight_flag=0,
                 seed=5, upos=False, umed=False, uad=False, qk=False, cluster_size=0, wordnet=False, class_number=3,
                 vocab_limit=None):

        self.upos = upos
        self.umed = umed
        self.uad = uad
        self.qk = qk
        self.class_number = class_number

        self.data_file = data_file
        self.train_ratio = train_ratio
        self.test_val_ratio = test_val_ratio

        self.vocab_limit = vocab_limit
        self.med_flag = med_flag
        self.weight_policy = weight_flag
        self.cluster_size = cluster_size
        self.wordnet = wordnet

        self.questions_name = column_names.question_name
        self.score_col = column_names.question_name_label
        self.features_cols = column_names.question_name_features
        self.pos_tag_old = column_names.question_name_pos
        self.pos_tag_new = column_names.question_name_pos_medical  # pos_tag_medical

        if self.wordnet:
            self.medical_question_name = column_names.wordnet_name
        else:
            self.medical_question_name = column_names.question_name_new

        self.medical_terms = column_names.question_name_new_flag

        self.question_cluster = column_names.question_name_cluster

        self.final_name = [self.questions_name]
        self.number_channels = 1
        if self.uad:
            self.final_name = self.final_name + self.features_cols
        if self.upos:
            self.final_name = self.final_name + [self.pos_tag_old]
            self.number_channels = self.number_channels + 1
        if self.umed:
            self.final_name = self.final_name + [self.medical_question_name]
            self.number_channels = self.number_channels + 2
        if self.qk:
            self.final_name = self.final_name + [self.question_cluster]

        self.x_train = list()
        self.y_train = list()
        self.x_val = list()
        self.y_val = list()
        self.create_voc_variable()
        # ---------------------
        self.max_sentence_lenght = 0

        self.use_cuda = torch.cuda.is_available()
        self.seed = seed
        self.create_data()

        self.create_alphabet()

    def text_to_word_list(self, text):
        ''' Pre process and convert texts to a list of words '''
        text = str(text)
        text = text.lower()

        text = text.split()

        return text

    def change_words_to_numbers(self, data_df, stops):
        # Iterate over required sequences of provided dataset
        for index, row in data_df.iterrows():

            # if we add concepts/ medical_concept to the sentences
            if self.med_flag == 0 or self.med_flag == 2:
                coll = [self.questions_name]
                pos = self.pos_tag_old
            elif self.med_flag == 1:
                coll = [self.medical_question_name]
                pos = self.pos_tag_new

            # if we have multiple collumns in data
            for sequence in coll:
                list_temp = {}
                for i in range(0, self.number_channels):
                    list_temp[i] = []

                words_to_number = []
                words_temp = self.text_to_word_list(row[sequence].strip())
                if self.max_sentence_lenght < len(row[sequence].strip()):
                    self.max_sentence_lenght = len(row[sequence].strip())

                if self.upos:
                    words_to_number_pos = []
                    pos_temp = self.text_to_word_list(row[pos])

                if self.umed:
                    words_to_number_medical = []
                    if not self.wordnet:
                        medical_flag = row[self.medical_terms].split(",")  # if there is a medical concept
                    content_temp = row[self.medical_question_name].split()

                for word_index in range(0, len(words_temp)):
                    word = words_temp[word_index]
                    list_temp[0].append(word)

                    if self.upos:
                        word_pos = pos_temp[word_index]
                        list_temp[1].append(word_pos)
                    if self.umed:
                        content_word = content_temp[word_index]
                        list_temp[2].append(content_word)
                        if not self.wordnet:
                            medical_flags_word = medical_flag[word_index]
                            if self.med_flag == 2:
                                if not medical_flags_word == "1":
                                    content_word = 'PAD'

                    # Remove unwanted words
                    if word in stops:
                        continue

                    if word not in self.word2index[0]:
                        self.word2index[0][word] = len(self.word2index[0])
                        words_to_number.append(self.word2index[0][word])
                        # for printing the sentence in prediction
                        self.number_to_word[self.word2index[0][word]] = word
                    else:

                        words_to_number.append(self.word2index[0][word])
                    # ---------------------------------pos tag-----------------------------------------
                    if self.upos:

                        if word_pos not in self.word2index[1]:

                            self.word2index[1][word_pos] = len(self.word2index[1])
                            words_to_number_pos.append(self.word2index[1][word_pos])
                        else:
                            words_to_number_pos.append(self.word2index[1][word_pos])
                    # ---------------------------for medical concepts-------------------------------------
                    if self.umed:

                        if content_word not in self.word2index[2]:

                            self.word2index[2][content_word] = len(self.word2index[2])
                            words_to_number_medical.append(self.word2index[2][content_word])

                        else:
                            words_to_number_medical.append(self.word2index[2][content_word])

                self.final_list[0].append(list_temp[0])
                # Replace |sequence as word| with |sequence as number| representation
                data_df.at[index, self.questions_name] = words_to_number

                if self.upos:
                    self.final_list[1].append(list_temp[1])
                    data_df.at[index, self.pos_tag_old] = words_to_number_pos
                if self.umed:
                    self.final_list[2].append(list_temp[2])
                    data_df.at[index, self.medical_question_name] = words_to_number_medical

        return data_df

    def split_pos(self, pos_list):
        return pos_list.replace("[", "").replace("]", "").replace("'", "").split(",")

    def read_data(self):
        self.final_list = {}
        for i in range(0, self.number_channels):
            self.final_list[i] = []
        stops = set(stopwords.words('english'))
        # Load data set
        data_df = pd.read_csv(self.data_file, sep='\t')
        data_df = self.change_words_to_numbers(data_df, stops)

        return data_df

    def to_pytorch_tensors(self):
        for data in [self.x_train, self.x_val, self.x_test]:
            for i, pair in enumerate(data):
                data[i][0] = torch.LongTensor(data[i][0])
                # ---pos_tagging tensor
                if self.upos:
                    data[i][self.pos_index] = torch.LongTensor(data[i][self.pos_index])
                # ---medical knowledge to tensor
                if self.umed:
                    data[i][self.med_index] = torch.LongTensor(data[i][self.med_index])
                if self.use_cuda:
                    data[i][0] = data[i][0].cuda()
                    if self.upos:
                        data[i][self.pos_index] = data[i][self.pos_index].cuda()
                    if self.umed:
                        data[i][self.med_index] = data[i][self.med_index].cuda()

        self.y_train = torch.FloatTensor(self.y_train)
        self.y_val = torch.FloatTensor(self.y_val)
        self.y_test = torch.FloatTensor(self.y_test)

        if self.use_cuda:
            self.y_train = self.y_train.cuda()
            self.y_val = self.y_val.cuda()
            self.y_test = self.y_test.cuda()
        return

    def create_data(self):
        # Loading data and building vocabulary.
        data_df = self.read_data()
        X = data_df[self.final_name]
        Y = data_df[self.score_col]

        self.question_index = []
        self.x_train, self.x_val_temp, self.y_train, self.y_val_temp = split_data(X, Y, train_size=self.train_ratio,
                                                                                  stratify=Y,
                                                                                  random_state=self.seed)
        self.x_test, self.x_val, self.y_test, self.y_val = split_data(self.x_val_temp, self.y_val_temp,
                                                                      train_size=self.test_val_ratio,
                                                                      stratify=self.y_val_temp,
                                                                      random_state=self.seed)

        # Convert labels to their numpy representations
        self.y_train = self.y_train.values
        self.y_val = self.y_val.values
        self.y_test = self.y_test.values

        bins = np.bincount(Y)
        class_bin = []
        for i in range(0, self.class_number):
            class_bin.append(bins[i])

        self.final_weights = []
        if self.weight_policy == 0:
            max_number = 1
            for i in range(0, self.class_number):
                self.final_weights.append(float(max_number) / class_bin[i])

        elif self.weight_policy == 1:
            max_number = max(class_bin)
            for i in range(0, self.class_number):
                self.final_weights.append(float(max_number) / class_bin[i])

        elif self.weight_policy == 2:
            n_samples = Y.count()
            n_classes = bins.shape[0]
            for i in range(0, self.class_number - 1):
                self.final_weights.append(float(n_samples) / (n_classes * class_bin[i]))

        features = self.read_features(data=0)
        self.x_train = features[0]
        self.y_train = features[1]

        features = self.read_features(data=1)
        self.x_test = features[0]
        self.y_test = features[1]

        features = self.read_features(data=2)
        self.x_val = features[0]
        self.y_val = features[1]

        assert len(self.x_train) == len(self.y_train)
        assert len(self.x_val) == len(self.y_val)
        assert len(self.x_test) == len(self.y_test)

        self.to_pytorch_tensors()
        return

    def read_features(self, data=0):
        # Split to lists
        self.feature_index = None
        self.pos_index = None
        self.med_index = None
        self.cluster_index = None
        if data == 0:
            features_rows = self.x_train
            score_rows = self.y_train
        elif data == 1:
            features_rows = self.x_test
            score_rows = self.y_test
        elif data == 2:
            features_rows = self.x_val
            score_rows = self.y_val

        features = []
        scores = []
        i = 0
        for index, row in features_rows.iterrows():
            list_temp = []
            sequence_1 = row[self.questions_name]
            if len(sequence_1) > 0:
                list_temp.append(sequence_1)
                if self.uad:
                    self.feature_index = features_rows.columns.get_loc(self.features_cols[0])
                    for j in range(0, len(self.features_cols)):
                        list_temp.append(row[self.features_cols[j]])
                if self.upos:
                    self.pos_index = features_rows.columns.get_loc(self.pos_tag_old)
                    list_temp.append(row[self.pos_tag_old])
                if self.umed:
                    self.med_index = features_rows.columns.get_loc(self.medical_question_name)
                    list_temp.append(row[self.medical_question_name])
                if self.qk:
                    self.cluster_index = features_rows.columns.get_loc(self.question_cluster)
                    cluster_ids = self.create_cluster(row[self.question_cluster])
                    list_temp.append(cluster_ids)
                features.append(list_temp)
                scores.append(float(score_rows[i]))
            i += 1
        return features, scores

    def create_voc_variable(self):
        self.number_to_word = {}
        self.word2index = {}
        for i in range(0, self.number_channels):
            self.word2index[i] = {}
            self.word2index[i]['PAD'] = 0
        return

    def create_cluster(self, cluster_list):
        list_index = [0] * self.cluster_size
        try:
            for number in cluster_list.split("-"):
                list_index[int(number) - 1] = 1
        except:
            pass
        return list_index

    def create_alphabet(self):
        self.alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
                         "t", "u", "v", "w", "x", "y", "z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", ",",
                         ";", ".", "!", "?", ":", "'", "\"", "\\", "/", "|", "_", "@", "#", "$", "%", "^", "&", "*",
                         "~", "`", "+", "-", "=", "<", ">", "(", ")", " ", "[", "]", "{", "}", "\n"]
