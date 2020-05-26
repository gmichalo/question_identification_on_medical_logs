
import torch
import torch.nn.utils.rnn as rnn


class Running_Network:
    def __init__(self, model, feature_index=0, pos_index=0, med_index=0, cluster_index=0, upos=False, umed=False,
                 uad=False, qk=False, class_number=3,
                 save_path="neural_network/model.pt", alphabet=None, max_sentence=None, voc=None):
        self.model = model
        self.use_cuda = torch.cuda.is_available()
        self.save_path = save_path
        self.name = model.name

        self.feature_index = feature_index
        self.pos_index = pos_index
        self.med_index = med_index
        self.cluster_index = cluster_index

        self.upos = upos
        self.umed = umed
        self.uad = uad
        self.qk = qk
        self.class_number = class_number
        self.softmax = torch.nn.Softmax(dim=1)
        self.alphabet = "".join(alphabet)

        self.max_sentence = max_sentence
        self.voc = voc

    def get_parameters(self):
        """
        get the number of parameters of the model
        :return:
        """
        #pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) #if we want only the trainable parameters
        return str(pytorch_total_params)

    def train(self, input_sequences, labels, criterion, model_optimizer=None, evaluate=False, nb_digits=4):

        self.model.train()

        sequences_1_pos = None
        sequences_1_med = None

        sequences_1 = [sequence[0] for sequence in input_sequences]
        sequences_1_temp = sequences_1
        temp = rnn.pad_sequence(sequences_1)

        batch_size = len(sequences_1)
        lenght1 = [sentence.size()[0] for sentence in sequences_1]
        sequences_1 = temp[:, :batch_size]

        if self.upos:
            sequences_1_pos = [sequence[self.pos_index] for sequence in input_sequences]
            temp_pos = rnn.pad_sequence(sequences_1_pos)
            sequences_1_pos = temp_pos[:, :batch_size]
        if self.umed:
            sequences_1_med = [sequence[self.med_index] for sequence in input_sequences]
            temp_med = rnn.pad_sequence(sequences_1_med)
            sequences_1_med = temp_med[:, :batch_size]

        additional_features = []
        if self.uad:
            for i in range(0, len(input_sequences)):
                list_temp = []
                for feat in range(self.feature_index, nb_digits + 1):
                    list_temp.append(input_sequences[i][feat])
                additional_features.append(list_temp)

        additional_features_temp = []
        if self.qk:
            for i in range(0, len(input_sequences)):
                additional_features_temp.append(input_sequences[i][self.cluster_index])

        if model_optimizer: model_optimizer.zero_grad()
        loss = 0.0

        labels = labels.type(torch.LongTensor)
        if self.name == "QUEST_CNN":
            output_scores = self.model([sequences_1], torch.tensor(additional_features), sequences_1_pos,
                                       sequences_1_med, torch.FloatTensor(additional_features_temp))
        elif self.name == "KIM_CNN":
            output_scores = self.model([sequences_1])
        elif self.name == "XML_CNN":
            output_scores = self.model([sequences_1])
        elif self.name == "BI_LSTM":
            hidden = self.model.init_hidden(batch_size)

            output_scores, perm_index = self.model([sequences_1], hidden, [lenght1], torch.tensor(additional_features),
                                                   sequences_1_pos, sequences_1_med,
                                                   torch.FloatTensor(additional_features_temp))
            labels = labels[perm_index]
        elif self.name == "FastText":
            output_scores = self.model([sequences_1])
        elif self.name == "SeqCNN":
            output_scores = self.model([sequences_1])
        elif self.name == "CHAR_CNN":
            char_embeding = self.onehot_encoding(self.number_to_sentence(sequences_1_temp))
            output_scores = self.model(char_embeding)

        loss = criterion(output_scores, labels)

        if not evaluate:
            loss.backward()
            model_optimizer.step()

        return loss.item(), output_scores

    def evaluation(self, input_sequences, labels, criterion, model_optimizer=None, evaluate=True,
                   validation_accuracy=False, nb_digits=4):

        if validation_accuracy:
            pass
        else:
            self.model.load_state_dict(torch.load(self.save_path), strict=False)
        self.model.eval()

        sequences_1_pos = None
        sequences_1_med = None

        sequences_1 = [sequence[0] for sequence in input_sequences]
        temp = rnn.pad_sequence(sequences_1)
        sequences_1_temp = sequences_1

        batch_size = len(sequences_1)
        lenght1 = [sentence.size()[0] for sentence in sequences_1]
        sequences_1 = temp[:, :batch_size]

        if self.upos:
            sequences_1_pos = [sequence[self.pos_index] for sequence in input_sequences]
            temp_pos = rnn.pad_sequence(sequences_1_pos)
            sequences_1_pos = temp_pos[:, :batch_size]
        if self.umed:
            sequences_1_med = [sequence[self.med_index] for sequence in input_sequences]
            temp_med = rnn.pad_sequence(sequences_1_med)
            sequences_1_med = temp_med[:, :batch_size]

        additional_features = []
        if self.uad:
            for i in range(0, len(input_sequences)):
                list_temp = []
                for feat in range(self.feature_index, nb_digits + 1):
                    list_temp.append(input_sequences[i][feat])
                additional_features.append(list_temp)

        additional_features_temp = []
        if self.qk:
            for i in range(0, len(input_sequences)):
                additional_features_temp.append(input_sequences[i][self.cluster_index])

        ''' No need to send optimizer in case of evaluation. '''
        if model_optimizer: model_optimizer.zero_grad()
        loss = 0.0

        labels = labels.type(torch.LongTensor)
        if self.name == "QUEST_CNN":
            output_scores = self.model([sequences_1], torch.tensor(additional_features), sequences_1_pos,
                                       sequences_1_med, torch.FloatTensor(additional_features_temp))
        elif self.name == "KIM_CNN":
            output_scores = self.model([sequences_1])
        elif self.name == "XML_CNN":
            output_scores = self.model([sequences_1])
        elif self.name == "BI_LSTM":
            hidden = self.model.init_hidden(sequences_1.size()[1])
            output_scores, perm_index = self.model([sequences_1], hidden, [lenght1], torch.tensor(additional_features),
                                                   sequences_1_pos, sequences_1_med,
                                                   torch.FloatTensor(additional_features_temp))
            labels = labels[perm_index]
        elif self.name == "FastText":
            output_scores = self.model([sequences_1])
        elif self.name == "SeqCNN":
            output_scores = self.model([sequences_1])
        elif self.name == "CHAR_CNN":
            char_embeding = self.onehot_encoding(self.number_to_sentence(sequences_1_temp))
            output_scores = self.model(char_embeding)
        loss = criterion(output_scores, labels)

        if not evaluate:
            loss.backward()
            model_optimizer.step()

        return loss.item(), output_scores, labels

    def onehot_encoding(self, sentences):
        x_tensor = []
        for sentence in sentences:
            x_temp = torch.zeros(len(self.alphabet), self.max_sentence)
            sentence_string = " ".join(sentence)
            for index_char in range(0, len(sentence_string)):
                char = sentence_string[index_char]
                index = self.alphabet.find(char)
                if index != -1:
                    x_temp[index][index_char] = 1.0
                else:
                    pass
            x_tensor.append(x_temp.unsqueeze(0))
        x_tensor = torch.cat(x_tensor, dim=0)
        return x_tensor

    def number_to_sentence(self, input_variables):
        senteces = []
        for index in range(0, len(input_variables)):
            sequence = input_variables[index]
            sentence = []
            for i in range(0, sequence.size()[0]):
                word = self.voc[int(sequence[i])]
                sentence.append(word)
            senteces.append(sentence)
        return senteces
