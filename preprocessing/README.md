#  The preprocessing folder contains files and the instructions for creating the appropriate format for the Quest CNN deep learning model

## Question identification

In order to identify the possible questions we provide 8 different question extraction method

in order to run all the question identification methods run:
```
python3 question_indetification.py -m 1 2 3 4 5 6 7 8 -f "../dataset_input/comments.csv"
```
which will create the question.csv file in questions_input folder


### Stanfordnlp parser
for methods **2** which  uses the **py-corenlp**, you first need to download the parser with:
```
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
unzip stanford-corenlp-full-2018-10-05.zip
```
and start the server:
```
cd stanford-corenlp-full-2018-10-05
java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 500000
```


## Create semantic features (optional)

### Ctakes pipeline
- Firstly, you need to download cTAKES and follow the instructions to install it from this [link](https://cwiki.apache.org/confluence/display/CTAKES/cTAKES+4.0+User+Install+Guide) and create a UMLS [account](https://uts.nlm.nih.gov/home.html#)
- Afterwards, you need to prepare the  question for the cTAKES tool by:

```
python3 create_ctakes_input.py
```
- In order to run the default pipeline of cTakes go to /usr/local/apache-ctakes-4.0.0/ and run
```
sudo bin/runPiperFile.sh  -i input_folder  --xmiOut output_folder  --user UMLS_username --pass UMLS_password
```
where
1. input_folder is the folder that contains the input files for cTAKES
2. output_folder is the folder where the cTAKES will create its output file
3. UMLS_username the username for the UMLS library account
4. UMLS_password password for the UMLS account

When the ctakes pipeline is finished run :
```
python3 ctakes_augmentation.py
```

in order to create the final file  with the name **question_complete.csv** in questions_input

## Wordnet
- We also provide an augmentation mechanism using Wordnet:
```
python3 wordnet_augmentation.py
```
will create a new file named **question_wordnet.csv** in questions_input with wordnet synonyms
