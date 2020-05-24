import pandas as pd

import re
import nltk

"""
main file for processing the CUI file from ctakes in order to crete the additional questions 
"""


class Reader:
    """
    reader to prepare the data from cTakes
    """

    def __init__(self, path, file_path_tui, path_csv, path_tui, path_initial, path_new_file):
        self.path = path

        self.file_path_tui = file_path_tui
        self.question_dictionary = {}

        self.path_csv = path_csv
        self.path_tui = path_tui
        self.path_initial = path_initial
        self.path_new_file = path_new_file

    def updating_tui(self):
        """
        for ALL the tui create a dictionary with their name
        """
        self.tui = {}
        file = open(self.path_tui, "r")
        for line in file:
            if line != "\n":
                name = line.split("|")[1].replace("\n", "").lower().split(" ")[0]
                name = name.replace(" ", "_")
                self.tui[line.split("|")[2].replace("\n", "").lower()] = name
        self.tui_read()
        return

    def create_initial_dictionary(self, text_full):
        """
        from the text extract initial sentences
        """
        text_full = text_full.replace('"question', "question")
        text_full = text_full.replace('"', "")
        questions = text_full.split("<question>")
        question_id = 0
        for question in questions:
            question = question.lower()
            question = re.sub("\s\s+", " ", question)
            self.question_dictionary[question_id] = {}
            self.question_dictionary[question_id][question] = {}
            question_id = question_id + 1
        return text_full

    def update_dictionary(self, text_full_updated):
        """
        from text extract new sentences with CUI and connect initial sentences to updated sentences

        """
        questions = text_full_updated.split("<question>")
        question_id = 0
        for question in questions:
            question = question.lower()
            question = re.sub("\s\s+", " ", question)
            if question not in self.question_dictionary[question_id]:
                self.question_dictionary[question_id][question] = {}
            question_id = question_id + 1
        return

    def create_dict(self, text_list):
        """
        create a dictionary with each concept id connecting to their phrase
        :param text_list:  list of .xmi sentences

        """
        id_dict = {}
        for i in range(0, len(text_list)):
            try:
                ids = text_list[i].split("ontologyConceptArr=")[1].split('"')[1].replace('"', '').split(" ")
                for j in range(0, len(ids)):
                    id_dict[ids[j]] = text_list[i]
            except:
                # for the case that there is any ontologyconcepts, we just go to the next section
                pass
        return id_dict

    def tui_read(self):
        """
        read the file that contains the tui that will use
        """
        self.tui_dict = {}
        with open(self.file_path_tui) as f:
            line = f.readline()
            while line:
                self.tui_dict[line.replace("\n", "")] = {}
                line = f.readline()
        return

    def augment(self):
        """
        For each patient we find the cui and we create their features from the cui
        """
        self.updating_tui()
        with open(self.path) as f:

            line = f.readline()
            text_full = line.split("sofaString=")[1].split("/>")[0]
            text_full = text_full.replace("&#10;", " ")
            text_full = text_full.replace("&quot;", "'")
            text_full = text_full.replace("&amp;", "a")
            text_full = text_full.replace("&gt;", ">")
            text_full = text_full.replace("&lt;", "<")

            text_full = self.create_initial_dictionary(text_full)
            list_temp = line.split("<refsem:UmlsConcept")
            text_list = list_temp[0].split("<")
            text_id = self.create_dict(text_list)

            self.full_text = text_full
            for i in range(1, len(list_temp)):
                ontology_id = list_temp[i].split("xmi:id=")[1].split(" ")[0]
                ontology_id = ontology_id.replace('"', '')
                tui_id = list_temp[i].split("tui=")[1].split(" ")[0].replace('"', "")
                if tui_id in self.tui_dict:
                    text = text_id[ontology_id]

                    begin = int(text.split("begin=")[1].split('"')[1])
                    end = int(text.split("end=")[1].split('"')[1])
                    name_tui = self.tui[tui_id.lower()]
                    text_replace = text_full[begin:end]

                    text_full_temp = text_full[:begin] + name_tui + text_full[end:]
                    # in order to find each position we update the terms one concept at a time
                    self.update_dictionary(text_full_temp)

        with open(self.path_csv, 'w') as f:
            f.write("new_question" + "\t" + "question" + "\t" + "new_flag" + "\t" + "new_tag" + "\n")
            for question_id in self.question_dictionary:
                initial = ""
                for question in self.question_dictionary[question_id]:
                    if initial == "":  # initial sentence
                        initial = question
                new_question, flag_new = self.create_new(self.question_dictionary[question_id], initial)
                # pos tag for new sentence
                tag_new = self.create_tags(new_question)
                f.write(new_question + "\t" + initial.strip() + "\t" + flag_new + "\t" + tag_new + "\n")

        return

    def create_tags(self, sentence):
        """
        find pos tags of the new sentence

        """
        w5list = {}
        tokens = sentence.split()
        pos_tag = nltk.pos_tag(tokens)
        tags = self.pos_tagging_creating(pos_tag, w5list)
        return tags

    def pos_tagging_creating(self, pos_tag, w5number):
        """
        create pos tags for sentences

        """
        pos_tag_list = []
        for i in range(0, len(pos_tag)):
            if pos_tag[i][0] in w5number:
                pass
            else:
                pos_tag_list.append(pos_tag[i][1])
        pos_tag_string = " ".join(pos_tag_list)
        return pos_tag_string

    def create_new(self, question_new, question):
        """
        create the new sentence and flag vector that indicates whether there is a medical concept in each position
        :param question_new: list of new sentences
        :param question: old sentence
        :return: new sentence and flag vector
        """
        initial_words = {}
        i = 0

        question = question.strip()
        for word in question.split():
            initial_words[i] = word
            i = i + 1

        final_word = {}
        for i in range(0, len(initial_words)):
            final_word[i] = initial_words[i]

        # find new sentence. update one concept per time
        for new_question in question_new:
            i = 0
            new_question = new_question.strip()
            new_list = new_question.split()

            flag = False
            word_index = 0
            while word_index < len(new_list):
                word = new_list[word_index]

                # for the case that two words are one concept we need to update both with the same concept
                if flag:
                    if new_list[word_index] != initial_words[i]:
                        word = word_temp
                        word_index = word_index - 1

                if initial_words[i] != word:
                    final_word[i] = word
                    flag = True
                    word_temp = word
                else:
                    flag = False
                i = i + 1
                word_index = word_index + 1

        # new sentence
        sentence = ""
        for i in range(0, len(final_word)):
            sentence = sentence + " " + final_word[i]
        sentence = sentence.strip()

        # flag whether we have a medical concept or not
        flag_new = []
        for i in range(0, len(initial_words)):
            if initial_words[i] == final_word[i]:
                flag_new.append("0")
            else:
                flag_new.append("1")
        flag_new_string = ",".join(flag_new)
        return sentence, flag_new_string

    def combining(self):
        df1 = pd.read_csv(self.path_initial, sep=",")
        df2 = pd.read_csv(self.path_csv, sep="\t")
        df1['question'] = df1['question'].str.lower()

        df3 = df1.merge(df2, on='question', how='left')
        df3.to_csv(self.path_new_file, sep="\t", index=False)

        return


reader = Reader('ctakes/ctakes_output/questions.txt.xmi',
                "ctakes/ctakes_output/final_tui.txt",
                "ctakes/questions_ctakes_full.csv", "ctakes/ctakes_output/tui.txt",
                "../../dataset_input/questions.csv",
                "../../dataset_input/question_complete.csv")

reader.augment()
reader.combining()

