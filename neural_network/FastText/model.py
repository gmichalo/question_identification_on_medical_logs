import torch
import torch.nn as nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F
import math


class FastText(nn.Module):
    def __init__(self, number_of_class, data_name, embedding, use_embedding=False, train_embedding=True):
        super(FastText, self).__init__()
        self.data_name = data_name
        self.use_cuda = torch.cuda.is_available()
        self.name = "FastText"
        self.number_of_class = number_of_class
        # -----------------------------------------------------------------------------
        self.embedding_size = embedding.shape[1]
        ''''
        Averaging the word embedding to a linear layer
        '''

        if use_embedding:
            self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
            self.embedding.weight = nn.Parameter(embedding)
            self.input_size = embedding.shape[1]  # V - Size of embedding vector
            self.embedding.weight.requires_grad = train_embedding
        else:
            embedding_matrix = torch.Tensor(embedding.shape[0] + 1, embedding.shape[1]).uniform_(
                -math.sqrt(float(3 / self.embedding_size)), math.sqrt(float(3 / self.embedding_size)))
            self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
            self.embedding.weight = nn.Parameter(embedding_matrix)
            self.embedding.weight.requires_grad = True
            self.input_size = embedding.shape[1]


        self.linear = nn.Linear(self.input_size, self.number_of_class)

    def forward(self, input):
        """
        :param input: word embedding of words
        """

        embedding_string = self.embedding(input[0])
        embedding_string = embedding_string.transpose(0, 1)
        embedding_string = embedding_string.transpose(1, 2)

        x = F.avg_pool1d(embedding_string, embedding_string.size(2)).squeeze(2)

        final = self.linear(x)

        return final
