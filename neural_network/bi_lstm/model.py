import torch
import torch.nn as nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F
import math


class BI_LSTM(nn.Module):
    def __init__(self, args, number_of_class, data_name, embedding, use_embedding=False,
                 train_embedding=True,
                 metadata_features=0, cluster_features=0, train_embedding_pos=False, embedding_matrix_pos=None,
                 train_embedding_medical=False, embedding_matrix_medical=None, hidden_size=50, upos=False, umed=False,
                 uad=False, qk=False):
        super(BI_LSTM, self).__init__()

        self.data_name = data_name
        self.use_cuda = torch.cuda.is_available()

        self.name = "BI_LSTM"
        self.number_of_class = number_of_class
        self.metadata_features = metadata_features
        self.cluster_features = cluster_features
        self.upos = upos
        self.umed = umed
        self.uad = uad
        self.qk = qk

        # -----------------------------------------------------------------------------
        self.embedding_size = embedding.shape[1]

        self.dropout = args.dropout
        self.bidirectional = args.bidirectional
        self.hidden_size = hidden_size

        if use_embedding:
            self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
            self.embedding.weight = nn.Parameter(embedding)
            self.input_size = embedding.shape[1]  # V - Size of embedding vector

            self.embedding.weight.requires_grad = train_embedding

            # ----------------------------------------pos tagging embedding  use_embedding for pos means one hot vector------------------------------
            if self.upos:
                self.embedding_pos = nn.Embedding(embedding_matrix_pos.shape[0], embedding_matrix_pos.shape[1])

                self.embedding_pos.weight = nn.Parameter(embedding_matrix_pos)
                self.input_size_pos = embedding_matrix_pos.shape[1]  # V - Size of embedding vector

                self.embedding_pos.weight.requires_grad = train_embedding_pos

            if self.umed:
                self.embedding_medical = nn.Embedding(embedding_matrix_medical.shape[0],
                                                      embedding_matrix_medical.shape[1])

                self.embedding_medical.weight = nn.Parameter(embedding_matrix_medical)
                self.input_size_medical = embedding_matrix_medical.shape[1]  # V - Size of embedding vector

                self.embedding_medical.weight.requires_grad = train_embedding_medical



        else:
            embedding_matrix = torch.Tensor(embedding.shape[0] + 1, embedding.shape[1]).uniform_(
                -math.sqrt(float(3 / self.embedding_size)), math.sqrt(float(3 / self.embedding_size)))
            self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
            self.embedding.weight = nn.Parameter(embedding_matrix)
            self.embedding.weight.requires_grad = True
            self.input_size = embedding.shape[1]

            if self.upos:
                embedding_matrix = torch.Tensor(embedding_matrix_pos.shape[0] + 1,
                                                embedding_matrix_pos.shape[1]).uniform_(
                    -math.sqrt(float(3 / self.embedding_size)), math.sqrt(float(3 / self.embedding_size)))
                self.embedding_pos = nn.Embedding(embedding_matrix_pos.shape[0], embedding_matrix_pos.shape[1])
                self.embedding_pos.weight = nn.Parameter(embedding_matrix)

                self.input_size_pos = embedding_matrix_pos.shape[1]  # V - Size of embedding vector
                self.embedding_pos.weight.requires_grad = True

            if self.umed:
                embedding_matrix = torch.Tensor(embedding_matrix_medical.shape[0] + 1,
                                                embedding_matrix_medical.shape[1]).uniform_(
                    -math.sqrt(float(3 / self.embedding_size)), math.sqrt(float(3 / self.embedding_size)))

                self.embedding_medical = nn.Embedding(embedding_matrix_medical.shape[0],
                                                      embedding_matrix_medical.shape[1])
                self.embedding_medical.weight = nn.Parameter(embedding_matrix)

                self.input_size_medical = embedding_matrix_medical.shape[1]  # V - Size of embedding vector
                self.embedding_medical.weight.requires_grad = True

        final_size = self.input_size

        if upos:
            final_size = final_size + self.input_size_pos
        if umed:
            final_size = final_size + self.input_size_medical

        if self.bidirectional:
            linear_size = 2 * self.hidden_size
        else:
            linear_size = self.hidden_size

        if uad:
            linear_size = linear_size + self.metadata_features
        if self.qk:
            linear_size = linear_size + self.cluster_features

        self.input_size = final_size

        self.lstm_1 = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, bidirectional=self.bidirectional,
                              batch_first=True)

        self.linear = nn.Linear(linear_size, self.number_of_class)

    def forward(self, input, hidden, x_lenghts, additional_features, sequences_1_pos, sequences_1_med,
                additional_features_temp):
        embedding_final = self.embedding(input[0])

        if self.upos:
            embedded_pos = self.embedding_pos(sequences_1_pos)
            embedding_final = torch.cat((embedding_final, embedded_pos), 2)
        if self.umed:
            embedded_med = self.embedding_medical(sequences_1_med)
            embedding_final = torch.cat((embedding_final, embedded_med), 2)

        lengths = torch.FloatTensor(x_lenghts[0])
        lengths, perm_index = lengths.sort(0, descending=True)
        embedding_final = embedding_final.transpose(0, 1)
        embedding_final = embedding_final[perm_index]

        embedded_final = torch.nn.utils.rnn.pack_padded_sequence(embedding_final, list(lengths.data), batch_first=True)

        outputs_1, hidden_1 = self.lstm_1(embedded_final, hidden)

        if self.bidirectional:
            final_output = torch.cat((hidden_1[0][-2, :, :], hidden_1[0][-1, :, :]), dim=1)
        else:
            final_output = hidden_1[0][-1]

        if self.uad:
            final1 = torch.cat((final_output, additional_features), 1)
        else:
            final1 = final_output

        if self.qk:
            final2 = torch.cat((final1, additional_features_temp), 1)
        else:
            final2 = final1

        lin_output = self.linear(final2)

        return lin_output, perm_index

    def init_hidden(self, batch_size):

        if self.bidirectional:
            layer = 2
        else:
            layer = 1
        hidden_state = torch.zeros(layer, batch_size, self.hidden_size)
        cell_state = torch.zeros(layer, batch_size, self.hidden_size)
        hidden = (hidden_state, cell_state)

        if self.use_cuda:
            return hidden.cuda()
        else:
            return hidden
