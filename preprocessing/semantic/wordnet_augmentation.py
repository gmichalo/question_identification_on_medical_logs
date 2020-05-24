import pandas as pd
from nltk.corpus import wordnet as wn


class wordnet_hypernyms:
    def __init__(self, path1, path2):
        self.path1 = path1
        self.path2 = path2

    def text_to_word_list(self, text):
        ''' Pre process and convert texts to a list of words '''
        text = str(text)
        text = text.lower()
        text = text.split()

        return text

    def find_hyper(self, word_list):
        list_temp = []
        for word in word_list:
            try:
                word_new = wn.synsets(word)[0].hypernyms()[0].lemma_names()[0]
            except:
                word_new = word

            list_temp.append(word_new)
        return " ".join(list_temp)

    def combining(self):
        final_list = []
        df1 = pd.read_csv(self.path1, sep=",")
        for index, row in df1.iterrows():
            word_list = self.text_to_word_list(row[0])
            word_list_temp = self.find_hyper(word_list)
            final_list.append([row[0], word_list_temp])

        df2 = pd.DataFrame(final_list, columns=["question", "word_net"])
        df3 = df1.merge(df2, on='question', how='left')
        df3.to_csv(self.path2, sep="\t", index=False)
        return


add = wordnet_hypernyms("../../dataset_input/questions.csv",
                        "../../dataset_input/question_wordnet.csv")

add.combining()
