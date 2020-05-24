import torch
import torch.nn as nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F
import math


class QUEST_CNN(nn.Module):
    def __init__(self, args, number_of_class, data_name, embedding, use_embedding=False,
                 train_embedding=True,
                 metadata_features=0, cluster_features=0, train_embedding_pos=False, embedding_matrix_pos=None,
                 train_embedding_medical=False, embedding_matrix_medical=None, upos=False, umed=False, uad=False,
                 qk=False):
        super(QUEST_CNN, self).__init__()
        self.data_name = data_name
        self.use_cuda = torch.cuda.is_available()
        self.name = "QUEST_CNN"
        self.number_of_class = number_of_class
        self.metadata_features = metadata_features
        self.cluster_features = cluster_features

        self.upos = upos
        self.umed = umed
        self.uad = uad
        self.qk = qk

        # -----------------------------------------------------------------------------
        self.embedding_size = embedding.shape[1]

        self.dropout_rate = args.dropout
        self.out_ch = args.feature_maps
        self.dropout_embedding = args.dropout_embedding
        self.itermidiate = args.intermidiate
        self.third_flag = args.third_flag
        # ---------------------------------

        filter_heights = args.filter_sizes
        number_channels = 1

        if use_embedding:
            self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
            self.embedding.weight = nn.Parameter(embedding)
            self.input_size = embedding.shape[1]  # V - Size of embedding vector
            self.embedding.weight.requires_grad = train_embedding

            # ----------------------------------------pos tagging embedding  use_embedding for pos means one hot vector------------------------------
            if self.upos:
                self.embedding_pos = nn.Embedding(embedding_matrix_pos.shape[0], embedding_matrix_pos.shape[1])
                # this does one-hot embedding
                self.embedding_pos.weight = nn.Parameter(embedding_matrix_pos)
                self.input_size_pos = embedding_matrix_pos.shape[1]  # V - Size of embedding vector
                self.embedding_pos.weight.requires_grad = train_embedding_pos
                number_channels = number_channels + 1

            if self.umed:
                self.embedding_medical = nn.Embedding(embedding_matrix_medical.shape[0],
                                                      embedding_matrix_medical.shape[1])
                number_channels = number_channels + 1

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
                number_channels = number_channels + 1

            if self.umed:
                embedding_matrix = torch.Tensor(embedding_matrix_medical.shape[0] + 1,
                                                embedding_matrix_medical.shape[1]).uniform_(
                    -math.sqrt(float(3 / self.embedding_size)), math.sqrt(float(3 / self.embedding_size)))

                self.embedding_medical = nn.Embedding(embedding_matrix_medical.shape[0],
                                                      embedding_matrix_medical.shape[1])
                self.embedding_medical.weight = nn.Parameter(embedding_matrix)

                self.input_size_medical = embedding_matrix_medical.shape[1]  # V - Size of embedding vector
                self.embedding_medical.weight.requires_grad = True
                number_channels = number_channels + 1

        if self.third_flag:
            self.conv = nn.ModuleList(
                [nn.Conv3d(1, self.out_ch, (number_channels, fh, self.input_size,), padding=(0, fh - 1, 0)) for fh in
                 filter_heights])
            self.conv3_bn = nn.ModuleList([nn.BatchNorm3d(self.out_ch) for fh in filter_heights])
        else:
            self.conv = nn.ModuleList(
                [nn.Conv2d(number_channels, self.out_ch, (fh, self.input_size), padding=(fh - 1, 0)) for fh in
                 filter_heights])
            self.conv2_bn = nn.ModuleList([nn.BatchNorm2d(self.out_ch) for fh in filter_heights])

        self.number_channels = number_channels

        linear_size = self.out_ch * len(filter_heights)
        self.linear_size = linear_size
        if uad:
            linear_size = linear_size + self.metadata_features
        if self.qk:
            linear_size = linear_size + self.cluster_features

        self.dropout = nn.Dropout(self.dropout_rate)

        self.linear = nn.Linear(linear_size, self.itermidiate)
        self.dense1_bn = nn.BatchNorm1d(self.itermidiate)

        self.linear2 = nn.Linear(self.itermidiate, self.number_of_class)

        self.embedding_dropout_layer = nn.Dropout(p=self.dropout_embedding)

    def forward(self, input, additional_features, sequences_1_pos, sequences_1_med, additional_features_temp):

        final_x = []

        if self.training:
            embedding_string = self.embedding_dropout_layer(self.embedding(input[0]))
        else:
            embedding_string = self.embedding(input[0])

        embedding_string = embedding_string.transpose(0, 1)
        embedding_string = embedding_string.unsqueeze(1)
        embedding_string_final = embedding_string
        if self.upos:
            if self.training:
                embedded_pos = self.embedding_dropout_layer(
                    self.embedding_pos(sequences_1_pos))
            else:
                embedded_pos = self.embedding_pos(sequences_1_pos)

            embedded_pos = embedded_pos.transpose(0, 1)
            embedding_string_pos = embedded_pos.unsqueeze(1)
            embedding_string_final = torch.cat((embedding_string_final, embedding_string_pos), 1)
        if self.umed:
            if self.training:
                embedding_med = self.embedding_dropout_layer(
                    self.embedding_medical(sequences_1_med))
            else:
                embedding_med = self.embedding_medical(sequences_1_med)

            embedding_med = embedding_med.transpose(0, 1)
            embedding_string_med = embedding_med.unsqueeze(1)
            embedding_string_final = torch.cat((embedding_string_final, embedding_string_med), 1)

        for i in range(0, len(self.conv)):
            if self.third_flag:
                result = F.relu(self.conv3_bn[i](self.conv[i](embedding_string_final.unsqueeze(1)))).squeeze(4).squeeze(
                    2)
            else:
                result = F.relu(self.conv2_bn[i](self.conv[i](embedding_string_final))).squeeze(3)
            final_x.append(result)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in final_x]
        x = torch.cat(x, 1)

        if self.uad:
            final1 = torch.cat((x, additional_features), 1)
        else:
            final1 = x

        if self.qk:
            final2 = torch.cat((final1, additional_features_temp), 1)
        else:
            final2 = final1
        lin_output = F.relu(self.linear(self.dropout(final2)))
        final = self.linear2(self.dense1_bn(lin_output))

        return final
