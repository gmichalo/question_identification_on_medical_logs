import argparse

import torch
import numpy as np
import random
from data import Read_Data
from embedding import Create_Embedding
from running_network import Running_Network
from network_iterations import Network_Iterations
from accuracy import create_accuracy_dictionary, write_csv, write_param, write_hyper
from search_spaces.read_json import read_hyp, assign

from neural_network.quest_cnn.model import QUEST_CNN
from neural_network.kim_cnn.model import KimCNN
from neural_network.xml_cnn.model import XMLCNN
from neural_network.bi_lstm.model import BI_LSTM
from neural_network.FastText.model import FastText
from neural_network.seq_cnn.model import Seq_CNN
from neural_network.char_cnn.model import CHARCNN


import time
import json



use_cuda = torch.cuda.is_available()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-modn", "--model_name", type=str, help="name of the model we are using", default="CHARCNN")

    parser.add_argument("-fn", "--data_final_name", type=str, help="result name.", default="kim")

    parser.add_argument("-dn", "--data_name", type=str, help="Dataset name.", default="question")

    parser.add_argument("-df", "--data_file", type=str, help="Path to dataset.",default="dataset_input/question_complete_final.csv")

    parser.add_argument("-med", "--medical_data", type=int,
                        help="0 not use semantic knowledge, 1 use concept in word vector, 2 additional channel for semantic concept ",
                        default=2)
    parser.add_argument("-wcl", "--weight_class", type=int,
                        help="0 1/class_size, 1 max_size/class_size, 2 additional use sklearn n_samples / (n_classes * np.bincount(y)) ",
                        default=0)

    parser.add_argument("-e", "--embd_file", type=str, help="Path to Embedding File of google.",
                        default="embedding_input/google_embedding/GoogleNews-vectors-negative300.bin.gz")
    parser.add_argument("-e_mimic", "--embd_file_mimic", type=str, help="Path to Embedding File of mimic.",
                        default="embedding_input/mimic_embedding/mimic.bin")
    parser.add_argument("-e_flag", "--embedding_flag", type=float,
                        help="  1 use google embedding, 2 use mimic dataset.", default=2)

    parser.add_argument("-oh", "--one_hot_pos", type=bool, help="flag if pos tag is one hot vector", default=False)
    parser.add_argument("-ohm", "--one_hot_medical", type=bool, help="flag if semantic concepts are one hot vector",
                        default=False)
    parser.add_argument("-ohseq", "--one_hot_word", type=bool, help="flag if words are presented as one-hot",
                        default=False)
    parser.add_argument("-em_flag_med", "--embedding_flag_medical", type=float,
                        help="  1 use google embedding, 2 use mimic dataset for semantic, 3 random start",
                        default=2)

    parser.add_argument("-cn", "--class_number", type=int, help="Number of class", default=3)

    parser.add_argument("-f", "--feature_number", type=int, help="Number of statistical features", default=4)
    parser.add_argument("-cf", "--cluster_number", type=int, help="Number of question extraction method number",
                        default=6)
    parser.add_argument("-tr", "--training_ratio", type=float, help="Ratio of training set.", default=0.8)
    parser.add_argument("-tv", "--test_val_ratio", type=float, help="Ratio of testing/validation set.", default=0.5)
    parser.add_argument("-l", "--embedding_size", type=int, help="embedding size", default=300)

    # -----dataset characteristics

    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size.", default=64)
    parser.add_argument("-n", "--num_iters", type=int, help="Number of iterations/epochs.", default=30)
    parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate for optimizer.", default=0.0383)
    parser.add_argument("-wd", "--weight_decay", type=float, help="weight decay", default=0)

    parser.add_argument("-usemb", "--use_embedding", type=bool, help="if we use pre-training embedding", default=False)

    parser.add_argument("-upos", "--upos", type=bool, help="whether we are using pos_tags", default=False)
    parser.add_argument("-uad", "--uad", type=bool, help="whether we are using statistical features", default=False)
    parser.add_argument("-umed", "--umed", type=bool, help="whether we are using semantic features", default=False)
    parser.add_argument("-qk", "--question_knowledge", type=bool,
                        help="whether use the knowledge from which question methods the sentence come",
                        default=False)

    parser.add_argument("-tr_e", "--training_embedding", type=bool,
                        help="If we will continue the  training of embedding.",
                        default=False)
    parser.add_argument("-tr_e_pos", "--training_embedding_pos", type=bool,
                        help="If we will continue the  training of pos-tag embedding.", default=False)
    parser.add_argument("-tr_e_med", "--training_embedding_med", type=bool,
                        help="If we will continue the  training of  semantic embedding.", default=False)

    parser.add_argument("-wordnet", "--wordnet", type=bool, help="whether we are using wordnet for semantic features",
                        default=False)

    parser.add_argument("-opt", "--optim", type=str,
                        help="optimization", default="adam")

    parser.add_argument("-pr", "--printing_loss", type=bool, help="whether we print the training loss in each epoch ",
                        default=False)

    parser.add_argument("-sp", "--save_path", type=str, help="path where the model will be saved",
                        default="neural_network/model.pt")

    # -----for cnn_text
    parser.add_argument("-multi", "--kmultichannel", type=int, help="whether we use mutlichannel for Kim ",
                        default=1)
    parser.add_argument("-tf", "--third_flag", type=bool, help="whether we use 2d or 3d tensor ",
                        default=False)
    parser.add_argument("-dr", "--dropout", type=float,
                        help="dropout for cnn_text", default=0.5)
    parser.add_argument("-dre", "--dropout_embedding", type=float,
                        help="dropout embedding for cnn_text", default=0.1)
    parser.add_argument("-inter", "--intermidiate", type=int,
                        help="size of the intermidiate layer for cnn_text", default=50)
    parser.add_argument("-fm", "--feature_maps", type=int,
                        help="size of feature map for each filter", default=100)
    parser.add_argument("-fs", "--filter_sizes", nargs='*', type=int, help="size  for each filter", default=[3, 4, 5])
    parser.add_argument("-dyn", "--dynamic_pool", type=int,
                        help="size of dynamic pooling map", default=8)
    parser.add_argument("-pk", "--pool_size", type=int, help="size  for   max pool", default=3)


    parser.add_argument("-z", "--hidden_size", type=int, help="Number of Units in LSTM layer.", default=50)
    parser.add_argument("-bi", "--bidirectional", type=bool, help="if it is bidirectional or not.", default=True)

    # -----------data columns attributes
    parser.add_argument("-qm", "--question_name", type=str,
                        help="name of the column that contain questions", default="question")
    parser.add_argument("-qml", "--question_name_label", type=str,
                        help="name of the column that contain label of the sentences", default='is_good_question')
    parser.add_argument("-qmf", "--question_name_features", type=list,
                        help="name of the columns that contain the list of the statistical features",
                        default=["lenght", "words", "coverage", "capitalize"])
    parser.add_argument("-qmp", "--question_name_pos", type=str,
                        help="name of the column that contain pos tag", default="pos_tag_list")
    parser.add_argument("-qmpm", "--question_name_pos_medical", type=str,
                        help="name of the column that contain pos tag of new sentences using semantic concepts",
                        default="question_name_pos_medical")
    parser.add_argument("-qmn", "--question_name_new", type=str,
                        help="name of the column that contain new question using semantic concepts",
                        default="new_question")
    parser.add_argument("-qmnf", "--question_name_new_flag", type=str,
                        help="name of the column that contain flags of the semantic concepts", default="new_flag")
    parser.add_argument("-qmnc", "--question_name_cluster", type=str,
                        help="name of the column that contain pre-determine information from which question method the sentence come from",
                        default="cluster")
    parser.add_argument("-wordnet_name", "--wordnet_name", type=str,
                        help="name of the collumn of the wordnet feature  ", default="word_net")

    parser.add_argument("-jf", "--json_file", type=str,
                        help="file with json file for hyperpameter search", default="search_spaces/seq_cnn.json")
    parser.add_argument("-hs", "--hyper_search", type=bool,
                        help="whether we want to be in hyper_parameter_test", default=True)
    parser.add_argument("-st", "--search_trial", type=int,
                        help="how many search trials we will use for hyperpameters testing", default=50)


    args = parser.parse_args()
    class_numbers = []
    for i in range(1, args.class_number + 1):
        class_numbers.append(i)
    accuracy_dictionary, accuracy_dictionary_val = create_accuracy_dictionary(class_names=class_numbers)

    if args.hyper_search:
        param = read_hyp(args.json_file)

    best_f1 = 0
    f1_list = []

    # if we use one seed
    seed = param['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # if you are using GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    for trial in range(0, args.search_trial):
        print(trial)
        if args.hyper_search:
            trial_parm = assign(param)
            for parameter in trial_parm:
                if parameter in args.__dict__:
                    args.__dict__[parameter] = trial_parm[parameter]

        seed = trial_parm['seed']
 
        seed_index = 0
        start = time.time()
        args.num_iters = 2

        seed_index = seed_index + 1
        print('Reading Data. with seeds: ' + str(seed))
        data = Read_Data(column_names=args, data_name=args.data_name, data_file=args.data_file,
                         train_ratio=args.training_ratio,
                         test_val_ratio=args.test_val_ratio,
                         med_flag=args.medical_data, weight_flag=args.weight_class, seed=seed, upos=args.upos,
                         umed=args.umed,
                         uad=args.uad, qk=args.question_knowledge, cluster_size=args.cluster_number,
                         wordnet=args.wordnet, class_number=args.class_number)

        print('Building Embedding Matrix')

        embedding = Create_Embedding(file_path=args.embd_file, embd_file_mimic=args.embd_file_mimic,
                                     word2index=data.word2index, custom=args.embedding_flag,
                                     final_list=data.final_list,
                                     one_hot_pos=args.one_hot_pos, one_hot_medical=args.one_hot_medical,
                                     custom_medical=args.embedding_flag_medical, upos=args.upos, umed=args.umed,
                                     seed=seed, one_hot_word=args.one_hot_word, embedding_size=args.embedding_size)

        print('Building model.')


        if args.model_name == "QUEST_CNN":

            model = QUEST_CNN(args, args.class_number, args.data_name, embedding.embedding_matrix,
                              use_embedding=args.use_embedding,
                              train_embedding=args.training_embedding, metadata_features=args.feature_number,
                              cluster_features=args.cluster_number,
                              embedding_matrix_pos=embedding.embedding_matrix_pos,
                              embedding_matrix_medical=embedding.embedding_matrix_medical,
                              train_embedding_pos=args.training_embedding_pos,
                              train_embedding_medical=args.training_embedding_med, upos=args.upos, umed=args.umed,
                              uad=args.uad, qk=args.question_knowledge)

        elif args.model_name == "KIM_CNN":
            model = KimCNN(args, args.class_number, args.data_name, embedding.embedding_matrix,
                           use_embedding=args.use_embedding,
                           train_embedding=args.training_embedding, multichannel=args.kmultichannel)
        elif args.model_name == "XML_CNN":
            model = XMLCNN(args, args.class_number, args.data_name, embedding.embedding_matrix,
                           use_embedding=args.use_embedding,
                           train_embedding=args.training_embedding, multichannel=args.kmultichannel)
        elif args.model_name == "BI_LSTM":
            model = BI_LSTM(args, args.class_number, args.data_name, embedding.embedding_matrix,
                            use_embedding=args.use_embedding,
                            train_embedding=args.training_embedding, metadata_features=args.feature_number,
                            embedding_matrix_pos=embedding.embedding_matrix_pos,
                            embedding_matrix_medical=embedding.embedding_matrix_medical,
                            train_embedding_pos=args.training_embedding_pos,
                            train_embedding_medical=args.training_embedding_med, hidden_size=args.hidden_size,
                            upos=args.upos, umed=args.umed, uad=args.uad)
        elif args.model_name == "FastText":
            model = FastText(args.class_number, args.data_name, embedding.embedding_matrix,
                             use_embedding=args.use_embedding,
                             train_embedding=args.training_embedding)
        elif args.model_name == "SeqCNN":
            model = Seq_CNN(args, args.class_number, args.data_name, embedding.embedding_matrix)
        elif args.model_name == "CHAR_CNN":
            model = CHARCNN(args, args.class_number, data.max_sentence_lenght, input_channel=len(data.alphabet),
                            output_channel=args.feature_maps,
                            dropout=args.dropout, linear_size=args.intermidiate)

        if use_cuda: model = model.cuda()

        print("Training Network.")
        running_network = Running_Network(model, feature_index=data.feature_index, pos_index=data.pos_index,
                                          cluster_index=data.cluster_index,
                                          med_index=data.med_index, upos=args.upos, umed=args.umed,
                                          uad=args.uad, qk=args.question_knowledge, class_number=args.class_number,
                                          save_path=args.save_path, alphabet=data.alphabet,
                                          max_sentence=data.max_sentence_lenght, voc=data.number_to_word)

        network_iteration = Network_Iterations(args.printing_loss, args.data_name, running_network, data.x_train,
                                               data.y_train,
                                               args.batch_size, args.num_iters, args.learning_rate,
                                               x_val=data.x_val,
                                               y_val=data.y_val,
                                               x_test=data.x_test, y_test=data.y_test,
                                               weight_class=data.final_weights, class_names=class_numbers,
                                               weight_decay=args.weight_decay, class_number=args.class_number,
                                               save_path=args.save_path)
        network_iteration.train_iters(args)
        end = time.time()
        run_end = end - start
        accuracy_dictionary, f1_model = network_iteration.model_eval(accuracy_dictionary, test=False, time=run_end,
                                                                     voc=data.number_to_word)
        print(f1_model)


        if f1_model > best_f1:
            best_f1 = f1_model
            final_param = trial_parm
        f1_list.append(str(f1_model))


    parameters = running_network.get_parameters()
    name1 = "dataset_output/hyperpameters/results_" + args.data_final_name + ".csv"
    name2 = "dataset_output/hyperpameters/results_" + args.data_final_name + ".json"

    write_hyper(name1, name2, f1_list, final_param)

    name1 = "dataset_output/hyperpameters/results_param_" + args.data_final_name + ".csv"
    write_param(name1, parameters)


