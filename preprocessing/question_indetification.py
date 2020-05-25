import csv
import re
from nltk import tokenize
from pycorenlp import StanfordCoreNLP
import pprint
from os import system
import nltk
import threading
from time import sleep
import pandas as pd
import argparse

import re
import os
import glob

W5H1 = ["who ", "what ", "where ", "when ", "why ", "how "]
aux_vers = ["am", "is", "are", "was", "were", "do", "does", "did", "have", "had", "has", "will"]
regular_expressions = ["i .*m looking for ", "i.*wonder.*", "i.*(try.*|like|need) to find",
                       "i.*(try.*|like|need) to know"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--methods", nargs='*', type=int,
                        help="which methods will be used for question indetification", default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("-f", "--file_name", type=str,
                        help="name of the initial files that we want to extract questions",
                        default="../dataset_input/comments.csv")
    parser.add_argument("-p", "--pre_de", type=bool,
                        help="whether we want to add the information from each method each sentence came", default=False)


def start_server():
    thread = threading.Thread(target=start_command, args=())
    thread.daemon = True  # Daemonize thread
    thread.start()
    return thread


def start_command():
    system('java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 150000')


class Questions:
    def __init__(self, path, path_question, path_final):
        self.text = []
        self.questions = {}
        self.word_dictionary = {}
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count = line_count + 1
                else:
                    line_count = line_count + 1
                    self.text.append(row[1])
                    self.count_unique_word(row[1])
            self.line_count = line_count
        print("Number of comments:" + str(self.line_count))
        self.question_file = path_question
        self.path_final = path_final
        return

    def count_unique_word(self, row):
        for word in row.split(" "):
            if len(word) > 0:
                if word not in self.word_dictionary:
                    self.word_dictionary[word] = 0

    def pipeline(self, question_method):
        print("using question extraction method: " + str(question_method))
        if question_method == 1:
            self.question_method1()
        elif question_method == 2:
            self.question_method2()
        elif question_method == 3:
            self.question_method3()
        elif question_method == 4:
            self.question_method4()
        elif question_method == 5:
            self.question_method5()
        elif question_method == 6:
            self.question_method6()
        elif question_method == 7:
            self.question_method7()
        elif question_method == 8:
            self.question_method8()

        self.question_printing()

    def tokenize(self, comment):
        sentences = tokenize.sent_tokenize(comment)
        return sentences

    def question_method1(self):
        """
        First method:
            if a sentence end in ? it is a question
        """

        number_sentences = 0
        for comment in self.text:
            sentences = self.tokenize(comment)
            number_sentences = number_sentences + len(sentences)
            for sentence in sentences:
                if sentence.find("?") != -1:
                    self.adding_questions(sentence)
        print("number of sentences " + str(number_sentences))

        return

    def question_method2(self):
        """
        use the stanford tree parser to discover sentences

        """
        thread = start_server()
        sleep(1)
        nlp = StanfordCoreNLP('http://localhost:9000')

        for comment in self.text:
            sentences = self.tokenize(comment)

            for temp_text in sentences:
                output = nlp.annotate(temp_text, properties={
                    'annotators': 'parse',
                    'outputFormat': 'json'
                })
                for sentence in output["sentences"]:
                    if sentence['parse'].find("SBARQ") != -1 or sentence['parse'].find("SQ") != -1:
                        self.adding_questions(temp_text)

        thread.join()
        return

    def question_method3(self):
        """
        Use nlp Dialogue Act Types
        """
        posts = nltk.corpus.nps_chat.xml_posts()[:10000]

        featuresets = [(self.dialogue_act_features(post.text), post.get('class')) for post in posts]

        train_set = featuresets
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        for comment in self.text:
            sentences = self.tokenize(comment)
            for sentence in sentences:
                feature = self.dialogue_act_features(sentence)
                tag = (classifier.classify(feature))
                if tag.find("Question") != -1:
                    self.adding_questions(sentence)
        return

    def question_method4(self):
        """
        Fourth method:
            if a sentence has  W5H1 and ? it is a question
        """
        for comment in self.text:
            sentences = self.tokenize(comment)

            for sentence in sentences:
                if self.W5H1(sentence) or sentence.find("?") != -1:
                    self.adding_questions(sentence)
        return

    def question_method5(self):
        """
        method 5:
            if a sentence has  W5H1 at the beginning of the question and ? it is a question
        """
        for comment in self.text:
            sentences = self.tokenize(comment)

            for sentence in sentences:
                if self.W5H1_first(sentence) or sentence.find("?") != -1:
                    self.adding_questions(sentence)
        return

    def adding_questions(self, sentence):
        if sentence not in self.questions:
            self.questions[sentence] = 0
        self.questions[sentence] = self.questions[sentence] + 1
        return

    def question_method6(self):
        """
        method 6:
            if a sentence has  W5H1 with an Auxiliary words
        """
        for comment in self.text:
            sentences = self.tokenize(comment)

            for sentence in sentences:
                if self.W5H1_aux(sentence) or sentence.find("?") != -1:
                    self.adding_questions(sentence)
        return

    def question_method7(self):
        """
        method 7:
            if a sentence has  W5H1 with an Auxiliary words and first word
        """
        for comment in self.text:
            sentences = self.tokenize(comment)

            for sentence in sentences:
                if self.W5H1_aux_fist(sentence) or sentence.find("?") != -1:
                    self.adding_questions(sentence)
        return

    def question_method8(self):
        """
        method 8:
        Use the regular expression from:
        Miles Efron and Megan Winget. 2010. Questions are content: a taxonomy of questions in a microblogging environment.
        """
        for comment in self.text:
            sentences = self.tokenize(comment)

            for sentence in sentences:
                if self.regular_expression_content(sentence) or sentence.find("?") != -1:
                    self.adding_questions(sentence)

        return

    def W5H1(self, sentence):
        W5H1_flag = False
        for word in W5H1:
            if sentence.lower().find(word) != -1:
                W5H1_flag = True
                break
        return W5H1_flag

    def W5H1_first(self, sentence):
        W5H1_flag = False
        for word in W5H1:
            if sentence.lower().find(word) == 0:
                W5H1_flag = True
                break
        return W5H1_flag

    def W5H1_aux(self, sentence):
        W5H1_flag_aux = False
        for word in W5H1:
            for aux in aux_vers:
                word_temp = word + aux
                if sentence.lower().find(word_temp) != -1:
                    W5H1_flag_aux = True
                    break
        return W5H1_flag_aux

    def W5H1_aux_fist(self, sentence):
        W5H1_flag_aux = False
        for word in W5H1:
            for aux in aux_vers:
                word_temp = word + aux
                if sentence.lower().find(word_temp) == 0:
                    W5H1_flag_aux = True
                    break
        return W5H1_flag_aux

    def regular_expression_content(self, sentence):
        regular_expression_flag = False
        for regular in regular_expressions:
            if re.search(regular, sentence.lower()):
                regular_expression_flag = True
                break
        return regular_expression_flag

    def question_printing(self):

        final_question_list = []

        self.number_words_vocabulary = len(self.word_dictionary)

        print("Number of comments:" + str(self.line_count))
        print("Number of questions found:" + str(len(self.questions)))

        for question in self.questions:
            features = self.calculate_features(question)
            pos_tag = nltk.pos_tag(self.text_to_word_list(question.strip()))
            pos_tag_list = self.pos_tagging_creating(pos_tag)

            label = self.find_label(question)
            final_question_list.append(
                [question, str(label), str(features[0]), str(features[1]),
                 str(features[2]), str(features[3]), pos_tag_list])
        df = pd.DataFrame(final_question_list,
                          columns=['question', 'is_good_question', "lenght", "words", "coverage",
                                   "capitalize", "pos_tag_list"])
        df.to_csv(self.question_file, index=False, sep='\t')

    def find_label(self, question):
        """
        Because there is not ground truth the label is "" otherwise it should be 0 (not question) or 1 (question) or 2 (c-question)
        """
        return ""

    def pos_tagging_creating(self, pos_tag):
        pos_tag_list = []
        for i in range(0, len(pos_tag)):
            pos_tag_list.append(pos_tag[i][1])
        pos_tag_string = " ".join(pos_tag_list)
        return pos_tag_string

    def calculate_features(self, question):
        """
        calculate metadata for question
        """
        features = []

        lenght = len(question)
        features.append(lenght)

        words = self.count_word(question)
        features.append(words)

        number_of_unique_words = self.count_unique(question)
        coverage = str(float(number_of_unique_words) / float(self.number_words_vocabulary))
        features.append(coverage)
        capitalized_words = self.count_capitalize(question)
        features.append(capitalized_words)

        return features

    def count_word(self, question):
        count_word = 0
        for word in question.split(" "):
            if len(word) > 0:
                count_word = count_word + 1
        return count_word

    def count_capitalize(self, question):
        count_capitalize = 0
        for word in question.split(" "):
            if len(word) > 0:
                if word[0].isupper():
                    count_capitalize = count_capitalize + 1
        return count_capitalize

    def count_unique(self, question):
        unique = {}
        for word in question.split(" "):
            if len(word) > 0:
                if word not in unique:
                    unique[word] = 0
        return len(unique)

    def dialogue_act_features(self, post):
        features = {}
        for word in nltk.word_tokenize(post):
            features['contains({})'.format(word.lower())] = True
        return features

    def text_to_word_list(self, text):
        ''' Pre process and convert texts to a list of words '''
        text = str(text)
        text = text.lower()

        text = text.split()

        return text

    def create_question_flags(self, vertical_stack):
        final = []
        questions_flag = self.question_feature()
        for row in vertical_stack.iterrows():
            flags = []
            for i in questions_flag:
                if row[1][0].lower() in questions_flag[i]:
                    flags.append(i)
            final.append(list(row[1]) + ["-".join(flags)])
        vertical_stack = pd.DataFrame(final, columns=list(vertical_stack.columns) + ["cluster"])
        return vertical_stack

    def question_feature(self):
        extension = 'csv'
        question_flag = {}
        result = [i for i in glob.glob('*.{}'.format(extension))]
        for i in result:
            number = i.split(".csv")[0].split("_")[1]
            if number != "2" and number != "3":
                if int(number) == 1:
                    pass
                else:
                    number = str(int(number) - 2)
                question_flag[number] = {}
                question = pd.read_csv(str(i), sep="\t")
                vertical_stack = question['question']
                vertical_stack = vertical_stack.str.lower()
                vertical_stack = vertical_stack.drop_duplicates()
                vertical_stack = vertical_stack.to_frame("question")

                vertical_stack = vertical_stack.drop_duplicates()
                for index, row in vertical_stack.iterrows():
                    question_flag[number][row[0]] = 0
        return question_flag

    def merge(self, path):
        extension = 'csv'
        os.chdir(path)
        result = [i for i in glob.glob('*.{}'.format(extension))]
        vertical_stack = pd.concat([pd.read_csv(str(i), sep="\t") for i in result], axis=0)
        vertical_stack = vertical_stack.drop_duplicates()
        if args.pre_de:
            vertical_stack = self.create_question_flags(vertical_stack)
        vertical_stack.to_csv(self.path_final, index=False)


print("start")

args = parser.parse_args()
for i in args.methods:
    extractor = Questions(args.file_name, 'questions_methods/questions_' + str(i) + '.csv',
                          '../../dataset_input/questions.csv')
    extractor.pipeline(i)
extractor.merge('questions_methods/')
