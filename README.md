#  Where's the Question? A Multi-channel Deep Convolutional Neural Network for Question Identification in Textual Data

(accepted paper in [The 3rd Clinical Natural Language Processing Workshop](https://clinical-nlp.github.io/2020/))

## General info
This is the code that was used of the paper : Where's the Question? A Multi-channel Deep Convolutional Neural Network for Question Identification in Textual Data where we created a multi-channel convolutional neural network for the seperation of sentences to question, not-questions and *c-questions* questions referring to an issue mentioned in a nearby sentence (e.g.,  can you clarify this?)

## Technologies
This project was created with python 3.7 and PyTorch 0.4.1

## Models
We provide code of the following models:
- [Quest_CNN](neural_network/quest_cnn): code for Where's the Question? A Multi-channel Deep Convolutional Neural Network for Question Identification in Textual Data
- [KIM_CNN](neural_network/kim_cnn):  [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)
- [XML_CNN](neural_network/xml_cnn): [Deep Learning for Extreme Multi-label Text Classification](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf)
- [Seq_cnn](neural_network/seq_cnn):[Effective Use of Word Order for Text Categorization with Convolutional Neural Networks](https://arxiv.org/pdf/1412.1058.pdf)
- [FastText](neural_network/FastText):[Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)
- [CHAR_CNN](neural_network/char_cnn):[Character-level Convolutional Network ](https://arxiv.org/pdf/1509.01626.pdf)
- [Bi_LSTM](neural_network/bi_lstm): a bi-lstm implementation which is equivalent of the **quest-cnn** in the paper Where's the Question? A Multi-channel Deep Convolutional Neural Network for Question Identification in Textual Data

For each model, we provide additional README in the folder with directions about how to run each model

## Setup
We recommend installing and running the code from within a virtual environment.

### Creating a Conda Virtual Environment
First, download Anaconda  from this [link](https://www.anaconda.com/distribution/)

Second, create a conda environment with python 3.7.
```
$ conda create -n cnn37 python=3.7
```
Upon  restarting your terminal session, you can activate the conda environment:
```
$ conda activate cnn37
```
### Install the required python packages
In the project root directory, run the following to install the required packages.
```
pip install -r requirements.txt
```
Finally, the stopwords from the NLTK library need to be download:
```
python
import nltk
nltk.download()
```
 


### Dowload pre-trained embeddings
1. Google pre-trained embeddings

In order to use pre-trained embeddings for the word embeddings (or the semantic embeddings), you need to dowload GoogleNews-vectors-negative300.bin.gz into the folder *embedding_input/google_embedding*

An easy way for dowloading is by:
```
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
```
2. Mimic pre-trained embeddings

Unfortunately, we cannot provide the embeddings of the MIMIC III dataset as training course is mandatory in order to access the particular dataset but the code can be still executed by only using the Google embeddings.

However, we provide the code for the creation of the mimic embeeding in the [file](https://github.com/gmichalo/question_identification_on_medical_logs/blob/master/embedding_input/mimic_embedding/mimic.py) which will require the NOTEEVENTS.csv from the MIMIC III dataset

### Extracting questions and  creation of   features of the deep neural network models

In the **preprocessing** folder, we provide code and instruction about how to extract potential question and the creation of all the features of the deep neural network models
## Running code
### Hyperpameter tuning
In order to tune the hyperpameter of each model you need to create a json file like the file the *search_spaces/cnn.json* and add it to *search_spaces/*

Afterwards, run:

```
python3 param_json.py --model_name "model_name"  -fn "results_file_name" - -jf "search_spaces/model.json" -st search_trials
```

The end results will be saved in *dataset_output/hyperpameters/* and it will create three files:
* results_file_name.csv : contains all the final F1 scores for each search trial
* results_file_name.json : contains the best hyper-parameters for the model
* results_file_name_param.csv : number of parameters of the model

### Running model
In order to run any model firstly you need to add the file that contains the sentences in question in *dataset_input/*.
This files need to have at least two collumns (sentences, label) but in order to use more features it needs additional columns (like pos-tag, medical-terms, ...)

Afterwards, run:

```
python3 main_iterations.py --model_name "model_name"  -fn "results_file_name"
```
The end results will be saved in *dataset_output/results/* and it will create two files:
* results_file_name.csv : contains all the results for each seed, the mean and standard deviation for the testing set
* results_file_name_val.csv : contains all the results for each seed, the mean and standard deviation for the validation set

In order to see all the parameters can be changed for additional experiments:

```
python main_iterations.py -help
 

usage: main_iterations.py [-h] [-modn MODEL_NAME] [-fn DATA_FINAL_NAME]
                          [-dn DATA_NAME] [-ner NER] [-df DATA_FILE]
                          [-dft DATA_FILE_TEST] [-dd DATA_FILE_DEV]
                          [-e EMBD_FILE] [-e_mimic EMBD_FILE_MIMIC]
                          [-e_flag EMBEDDING_FLAG] [-cn CLASS_NUMBER]
                          [-tr TRAINING_RATIO] [-tv TEST_VAL_RATIO]
                          [-l EMBEDDING_SIZE] [-opt OPTIM] [-b BATCH_SIZE]
                          [-n NUM_ITERS] [-lr LEARNING_RATE]
                          [-wd WEIGHT_DECAY] [-usemb USE_EMBEDDING]
                          [-tr_e TRAINING_EMBEDDING] [-sp SAVE_PATH]
                          [-pr PRINTING_LOSS] [-multi KMULTICHANNEL]
                          [-dr DROPOUT] [-fm FEATURE_MAPS]
                          [-fs [FILTER_SIZES [FILTER_SIZES ...]]]
                          [-z HIDDEN_SIZE] [-qm QUESTION_NAME]
                          [-qml QUESTION_NAME_LABEL]

optional arguments:
  -h, --help            show this help message and exit
  -modn MODEL_NAME, --model_name MODEL_NAME
                        name of the anmodel we are using
  -fn DATA_FINAL_NAME, --data_final_name DATA_FINAL_NAME
                        result name.
  -dn DATA_NAME, --data_name DATA_NAME
                        Dataset name.
  -ner NER, --ner NER   whether we use ner or re task
  -df DATA_FILE, --data_file DATA_FILE
                        Path to dataset.
  -dft DATA_FILE_TEST, --data_file_test DATA_FILE_TEST
                        Path to dataset test set.
  -dd DATA_FILE_DEV, --data_file_dev DATA_FILE_DEV
                        Path to dataset.
  -e EMBD_FILE, --embd_file EMBD_FILE
                        Path to Embedding File of google.
  -e_mimic EMBD_FILE_MIMIC, --embd_file_mimic EMBD_FILE_MIMIC
                        Path to Embedding File of mimic.
  -e_flag EMBEDDING_FLAG, --embedding_flag EMBEDDING_FLAG
                        1 use google embedding, 2 use mimic dataset, 3 random
                        start
  -cn CLASS_NUMBER, --class_number CLASS_NUMBER
                        Number of class
  -tr TRAINING_RATIO, --training_ratio TRAINING_RATIO
                        Ratio of training set.
  -tv TEST_VAL_RATIO, --test_val_ratio TEST_VAL_RATIO
                        Ratio of testing/validation set.
  -l EMBEDDING_SIZE, --embedding_size EMBEDDING_SIZE
                        embedding size
  -opt OPTIM, --optim OPTIM
                        optimization
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch Size.
  -n NUM_ITERS, --num_iters NUM_ITERS
                        Number of iterations/epochs.
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate for optimizer.
  -wd WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
                        weight decay
  -usemb USE_EMBEDDING, --use_embedding USE_EMBEDDING
                        if we use pre-training embedding
  -tr_e TRAINING_EMBEDDING, --training_embedding TRAINING_EMBEDDING
                        If we will continue the training of embedding.
  -sp SAVE_PATH, --save_path SAVE_PATH
                        path where the model will be saved
  -pr PRINTING_LOSS, --printing_loss PRINTING_LOSS
                        whether we print the training loss in each epoch
  -multi KMULTICHANNEL, --kmultichannel KMULTICHANNEL
                        whether we use mutlichannel for Kim
  -dr DROPOUT, --dropout DROPOUT
                        dropout for cnn_text
  -fm FEATURE_MAPS, --feature_maps FEATURE_MAPS
                        size of feature map for each filter
  -fs [FILTER_SIZES [FILTER_SIZES ...]], --filter_sizes [FILTER_SIZES [FILTER_SIZES ...]]
                        size for each filter
  -z HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
                        Number of Units in LSTM layer.
  -qm QUESTION_NAME, --question_name QUESTION_NAME
                        name of the column that contain questions
  -qml QUESTION_NAME_LABEL, --question_name_label QUESTION_NAME_LABEL
```
