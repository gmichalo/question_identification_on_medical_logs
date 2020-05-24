import torch
import torch.nn as nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F
import math


class KimCNN(nn.Module):
    def __init__(self, args, number_of_class, data_name, embedding, use_embedding=False, train_embedding=True,
                 multichannel=False):
        super(KimCNN, self).__init__()
        self.data_name = data_name
        self.use_cuda = torch.cuda.is_available()
        self.name = "KIM_CNN"
        self.number_of_class = number_of_class

        self.multichannel = multichannel
        # -----------------------------------------------------------------------------
        self.embedding_size = embedding.shape[1]

        self.dropout_rate = args.dropout
        self.out_ch = args.feature_maps
        filter_heights = args.filter_sizes

        number_channels = 1

        if self.multichannel == 2:
            self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
            self.embedding.weight = nn.Parameter(embedding)
            self.input_size = embedding.shape[1]  # V - Size of embedding vector
            self.embedding.weight.requires_grad = False
            # -----------------------------------------------------------
            self.embedding_non_static = nn.Embedding(embedding.shape[0], embedding.shape[1])
            self.embedding_non_static.weight = nn.Parameter(embedding)
            self.input_size = embedding.shape[1]  # V - Size of embedding vector
            self.embedding_non_static.weight.requires_grad = True
            number_channels = 2
        else:
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

        self.conv = nn.ModuleList(
            [nn.Conv2d(number_channels, self.out_ch, (fh, self.input_size), padding=(fh - 1, 0)) for fh in
             filter_heights])

        linear_size = self.out_ch * len(filter_heights)

        self.linear = nn.Linear(linear_size, self.number_of_class)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, input):
        """
        :param input: word embedding of words
        """

        final_x = []
        embedding_string = self.embedding(input[0])
        embedding_string = embedding_string.transpose(0, 1)  # (batch_size, seq_len(number of words), embd_dim)
        embedding_string = embedding_string.unsqueeze(1)  # (N, Cin, W, embd_dim)
        embedding_string_final = embedding_string
        if self.multichannel == 2:
            embedding_string2 = self.embedding_non_static(input[0])
            embedding_string2 = embedding_string2.transpose(0, 1)
            embedding_string2 = embedding_string2.unsqueeze(1)
            embedding_string_final = torch.cat((embedding_string_final, embedding_string2), 1)

        for i in range(0, len(self.conv)):
            result = F.relu(self.conv[i](embedding_string_final)).squeeze(3)
            final_x.append(result)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in final_x]
        x = torch.cat(x, 1)

        final = self.linear(self.dropout(x))

        return final
