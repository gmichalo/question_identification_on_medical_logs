import numpy as np
import gensim
import math
import torch


class Create_Embedding:

    def __init__(self, file_path=None, embd_file_mimic=None, word2index=None, custom=0,
                 final_list=None, one_hot_pos=True, one_hot_medical=True, custom_medical=0, upos=False, umed=False,
                 seed=42, one_hot_word=False, embedding_size=300):
        self.use_cuda = torch.cuda.is_available()
        self.embd_file_mimic = embd_file_mimic
        self.file_path = file_path

        self.embedding_size = embedding_size  # Dimensionality of Google News' Word2Vec
        self.embedding_matrix = None
        self.embedding_matrix_pos = None
        self.embedding_matrix_medical = None
        self.seed = seed

        if one_hot_word:
            self.embedding_matrix = self.create_one_hot_embed_matrix(word2index[0])
        else:
            if custom == 1:
                self.embedding_matrix = self.create_embed_matrix_google(file_path, word2index[0], final_list[0])
            elif custom == 2:
                self.embedding_matrix = self.create_embed_matrix_mimic(embd_file_mimic, word2index[0], final_list[0])
            elif custom == 3:
                self.embedding_matrix = torch.Tensor(len(word2index[0]) + 1, self.embedding_size).uniform_(
                    -math.sqrt(float(3 / self.embedding_size)), math.sqrt(float(3 / self.embedding_size)))
        if upos:
            self.embedding_matrix_pos = self.create_pos_embed_matrix(word2index[1], one_hot_pos)
        if umed:
            self.embedding_matrix_medical = self.create_medical_embed_matrix(word2index[2], final_list[2],
                                                                             one_hot_medical,
                                                                             custom_medical)
        return

    def create_one_hot_embed_matrix(self, word_index):
        one_hot_dimension = len(word_index)
        one_hot = torch.FloatTensor(one_hot_dimension, one_hot_dimension).zero_()
        target = one_hot.fill_diagonal_(1)
        embedding_matrix = torch.autograd.Variable(target)
        if self.use_cuda: embedding_matrix = embedding_matrix.cuda()
        return embedding_matrix

    def create_pos_embed_matrix(self, word_index_pos, one_hot_pos):

        if one_hot_pos:
            if self.embedding_size < len(word_index_pos) + 1:
                raise Exception("the embedding size should be higher")
            else:
                one_hot_dimension = self.embedding_size
            one_hot = torch.FloatTensor(one_hot_dimension, one_hot_dimension).zero_()
            target = one_hot.fill_diagonal_(1)
            embedding_matrix = torch.autograd.Variable(target)
        else:
            embedding_matrix = torch.Tensor(len(word_index_pos) + 1, self.embedding_size).uniform_(
                -math.sqrt(float(3 / self.embedding_size)), math.sqrt(float(3 / self.embedding_size)))

            if self.use_cuda: embedding_matrix = embedding_matrix.cuda()

        return embedding_matrix

    def create_medical_embed_matrix(self, word_index, final_list, one_hot_medical, custom):
        if one_hot_medical:

            if self.embedding_size < len(word_index) + 1:
                raise Exception("the embedding size should be higher")
            else:
                one_hot_dimension = self.embedding_size
            one_hot = torch.FloatTensor(one_hot_dimension, one_hot_dimension).zero_()
            target = one_hot.fill_diagonal_(1)
            embedding_matrix = torch.autograd.Variable(target)
        else:
            name = "medical"

            if custom == 1:
                embedding_matrix = self.create_embed_matrix_google(self.file_path, word_index, final_list,
                                                                   name)
            elif custom == 2:
                embedding_matrix = self.create_embed_matrix_mimic(self.embd_file_mimic, word_index, final_list,
                                                                  name)
            elif custom == 3:
                embedding_matrix = torch.Tensor(len(word_index) + 1, self.embedding_size).uniform_(
                    -math.sqrt(float(3 / self.embedding_size)), math.sqrt(float(3 / self.embedding_size)))

        return embedding_matrix

    def create_embed_matrix_google(self, file_path, word_index, final_list, name="word"):
        """
        use a mimic or google embedding in order to get a better representation

        :param file_path: of the google embedding
        :param word_index: for words not in the dataset (this will not effect)
        :param final_list: dataset list
        :param pretrained: get a pretrained embeding
        :return:
        """

        # train model
        print("start google embedding")
        model = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)

        # Prepare Embedding Matrix.
        embedding_matrix = np.zeros((len(word_index) + 1, self.embedding_size))
        for word, i in word_index.items():

            if word not in model.wv.vocab:
                if word != "PAD":
                    embedding_matrix[i] = torch.Tensor(self.embedding_size).uniform_(
                        -math.sqrt(float(3 / self.embedding_size)), math.sqrt(float(3 / self.embedding_size)))
            else:
                embedding_matrix[i] = model[word]

        del model
        embedding_matrix = torch.FloatTensor(embedding_matrix)
        if self.use_cuda: embedding_matrix = embedding_matrix.cuda()

        return embedding_matrix

    def create_embed_matrix_mimic(self, file_path, word_index, final_list, name="word"):
        """
        use a mimic or google embedding in order to get a better representation

        :param file_path: of mimic embedding
        :param word_index: for words not in the dataset (this will not effect)
        :param final_list: dataset list
        :param pretrained: get a pretrained embeding
        :return:
        """

        print("start mimic embedding")

        model = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)

        # Prepare Embedding Matrix.
        embedding_matrix = np.zeros((len(word_index) + 1, self.embedding_size))
        for word, i in word_index.items():

            if word not in model.wv.vocab:
                if word != "PAD":
                    embedding_matrix[i] = torch.Tensor(self.embedding_size).uniform_(
                        -math.sqrt(float(3 / self.embedding_size)), math.sqrt(float(3 / self.embedding_size)))
            else:
                embedding_matrix[i] = model[word]

        del model
        embedding_matrix = torch.FloatTensor(embedding_matrix)
        if self.use_cuda: embedding_matrix = embedding_matrix.cuda()

        return embedding_matrix
