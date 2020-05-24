import torch
import torch.nn as nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F
import math


class Seq_CNN(nn.Module):
    def __init__(self, args, number_of_class, data_name, embedding ):
        super(Seq_CNN, self).__init__()
        self.data_name = data_name
        self.use_cuda = torch.cuda.is_available()
        self.name = "SeqCNN"
        self.number_of_class = number_of_class
        number_channels = 1
        self.dropout_rate = args.dropout

        # -----------------------------------------------------------------------------
        self.embedding_size = embedding.shape[1]



        self.out_ch = args.feature_maps

        filter_heights = args.filter_sizes

        self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
        self.embedding.weight = nn.Parameter(embedding)
        self.input_size = embedding.shape[1]  # V - Size of embedding vector
        self.embedding.weight.requires_grad = False

        self.input_size = embedding.shape[1]
        self.conv = nn.ModuleList(
            [nn.Conv2d(number_channels, self.out_ch, (fh, self.input_size), padding=(fh - 1, 0)) for fh in
             filter_heights])

        linear_size = self.out_ch * len(filter_heights)


        self.lrn = nn.LocalResponseNorm(1, 1, 1 / 2, 1)  # for the normalization of (1 +z^2)^(-1/2)
        self.linear = nn.Linear(linear_size, self.number_of_class)

        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, input):
        """
        :param input: word embedding of words
        """
        final_x = []
        embedding_string = self.embedding(input[0])
        embedding_string = embedding_string.transpose(0, 1)
        embedding_string = embedding_string.unsqueeze(1)
        embedding_string_final = embedding_string

        for i in range(0, len(self.conv)):
            result = F.relu(self.conv[i](embedding_string_final)).squeeze(3)
            final_x.append(result)

        x = [self.lrn(F.max_pool1d(i, i.size(2))).squeeze(2) for i in final_x]
        x = torch.cat(x, 1)

        final = self.linear(self.dropout(x))

        return final
