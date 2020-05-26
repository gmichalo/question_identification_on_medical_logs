import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from sklearn import metrics
import re
from sklearn.metrics import f1_score
from tqdm import tqdm


class Network_Iterations:
    def __init__(self, print_flag, data_name, model, x_train, y_train,
                 batch_size, num_iters,
                 learning_rate, x_val=[], y_val=[], x_test=[], y_test=[], class_names=[1, 2, 3], validation_score=None,
                 weight_class=[1, 1, 1], weight_decay=0, save_path="neural_network/model.pt", class_number=3,
                 print_flag_probab=False):

        self.use_cuda = torch.cuda.is_available()
        self.data_name = data_name
        self.model_train = model
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.print_flag = print_flag
        self.class_names = class_names
        self.class_number = class_number
        self.print_flag_proba = print_flag_probab

        # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class. no need for softmax in the model
        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight_class))

        ''' Lists that will contain data in the form of tensors. '''
        # Training data.
        self.x_train = x_train
        self.y_train = y_train
        self.train_samples = len(self.x_train)

        # Development data.
        self.x_val = x_val
        self.y_val = y_val

        # Development data.
        self.x_test = x_test
        self.y_test = y_test

        self.validation_score = validation_score
        self.save_path = save_path
        self.softmax = torch.nn.Softmax(dim=1)

    def train_iters(self, args):

        print_loss_total = 0.0
        plot_loss_total = 0.0

        model_trainable_parameters = list(filter(lambda p: p.requires_grad, self.model_train.model.parameters()))
        if args.optim == "adam":
            self.model_optimizer = optim.Adam(model_trainable_parameters, lr=self.learning_rate,
                                              weight_decay=self.weight_decay)

        print('Beginning Model Training.\n')

        for epoch in tqdm(range(1, self.num_iters + 1)):
            epoch_loss = 0
            for i in range(0, self.train_samples, self.batch_size):
                input_variables = self.x_train[i: i + self.batch_size]  # Batch Size x Sequence Length
                class_id = self.y_train[i: i + self.batch_size]  # Batch Size

                loss, _ = self.model_train.train(input_variables, class_id, self.criterion, self.model_optimizer)
                print_loss_total += loss
                plot_loss_total += loss
                epoch_loss += loss
            validation_score_temp = self.validation_accuracy(validation_accuracy=True)
            if self.validation_score is None:
                self.validation_score = validation_score_temp
                torch.save(self.model_train.model.state_dict(), self.save_path)

            else:
                if self.validation_score < validation_score_temp:
                    self.validation_score = validation_score_temp
                    torch.save(self.model_train.model.state_dict(), self.save_path)
            if self.print_flag:
                print('Epoch %d : Training Loss: %f ' % (epoch, epoch_loss))

    def validation_accuracy(self, validation_accuracy=True):

        y_pred, y_true = list(), list()

        for i in range(0, len(self.x_val), self.batch_size):
            input_variables = self.x_val[i: i + self.batch_size]  # Batch Size x Sequence Length
            class_id = self.y_val[i: i + self.batch_size]  # Batch Size
            loss, scores, labels = self.model_train.evaluation(input_variables, class_id, self.criterion,
                                                               evaluate=True, validation_accuracy=validation_accuracy)

            y_pred.extend(torch.max(self.softmax(scores), 1)[1].data.numpy())

            y_true.extend(labels.data.numpy())

        if self.class_number == 2:
            micro_f1 = f1_score(y_true, y_pred)
        else:
            micro_f1 = f1_score(y_true, y_pred, average='micro')

        return micro_f1

    def model_eval(self, accuracy_dictionary, test=True, time=None, voc=None):

        total_loss = 0

        if test:
            input_variables_full = self.x_test  # Batch Size x Sequence Length
            actual_scores = self.y_test  # Batch Size
        else:
            input_variables_full = self.x_val  # Batch Size x Sequence Length
            actual_scores = self.y_val  # Batch Size

        y_pred, y_true, scores_final = list(), list(), list()
        for i in range(0, len(input_variables_full), self.batch_size):
            input_variables = input_variables_full[i: i + self.batch_size]  # Batch Size x Sequence Length
            class_id = actual_scores[i: i + self.batch_size]  # Batch Size
            loss, scores, labels = self.model_train.evaluation(input_variables, class_id, self.criterion, evaluate=True)
            scores_final.extend(scores)
            y_pred.extend(torch.max(self.softmax(scores), 1)[1].data.numpy())

            if self.print_flag_proba:
                sentences = self.number_to_sentence(input_variables, voc)
                for index in range(0, len(sentences)):
                    probabilities = ""
                    for i in range(0, self.class_number):
                        probabilities = probabilities + " " + str(round(float(self.softmax(scores)[index][i].data), 3))
                    print("sentence:" + " ".join(
                        sentences[index]) + " probabilities: " + probabilities + " label class:" + str(
                        labels[index].item()))

            y_true.extend(labels.data.numpy())
            total_loss += loss

        accuracy_dictionary = self.calculate_metric(accuracy_dictionary, y_true, y_pred, time)

        if self.class_number == 2:
            micro_f1 = f1_score(y_true, y_pred)
        else:
            micro_f1 = f1_score(y_true, y_pred, average='micro')

        return accuracy_dictionary, micro_f1

    def number_to_sentence(self, input_variables, voc):
        senteces = []
        for index in range(0, len(input_variables)):
            sequence = input_variables[index][0]
            sentence = []
            for i in range(0, sequence.size()[0]):
                word = voc[int(sequence[i])]
                sentence.append(word)
            senteces.append(sentence)

        return senteces

    def calculate_metric(self, accuracy_dictionary, y_true, y_pred, time):
        list_of_accuracies = (metrics.classification_report(y_true, y_pred, digits=3)).replace("\n\n", "\n").split("\n")
        for classes in self.class_names:
            clean_string = re.sub(' +', ' ', list_of_accuracies[classes]).split(" ")
            accuracy_dictionary[classes]["precision"].append(clean_string[2])
            accuracy_dictionary[classes]["recall"].append(clean_string[3])
            accuracy_dictionary[classes]["f1"].append(clean_string[4])

        clean_string = re.sub(' +', ' ', list_of_accuracies[len(self.class_names) + 1]).split(" ")
        accuracy_dictionary["accuracy"]["result"].append(clean_string[2])

        clean_string = re.sub(' +', ' ', list_of_accuracies[len(self.class_names) + 2]).split(" ")
        accuracy_dictionary["macro_avg"]["precision"].append(clean_string[3])
        accuracy_dictionary["macro_avg"]["recall"].append(clean_string[4])
        accuracy_dictionary["macro_avg"]["f1"].append(clean_string[5])

        clean_string = re.sub(' +', ' ', list_of_accuracies[len(self.class_names) + 3]).split(" ")
        accuracy_dictionary["weightmacro _avg"]["precision"].append(clean_string[2])
        accuracy_dictionary["weightmacro _avg"]["recall"].append(clean_string[3])
        accuracy_dictionary["weightmacro _avg"]["f1"].append(clean_string[4])
        accuracy_dictionary["time"]["result"].append(str(time))

        return accuracy_dictionary
