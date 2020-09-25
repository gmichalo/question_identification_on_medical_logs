import pandas as pd
from tqdm import tqdm
import re
import gensim


class create_mimic_embedding:
    def __init__(self, file_name,  embedding_size):
        self.file_name = file_name
        self.embedding_size = embedding_size

    def read_notes(self):
        notes_list = []
        df_notes = pd.read_csv(self.file_name)
        for row in tqdm(df_notes.iterrows()):
            text = row[1]["TEXT"]
            text = re.sub('\n+', '\n', text)
            text = re.sub(' +', ' ', text)

            notes_list.append(text.split(" "))
        print("model training")
        model = gensim.models.Word2Vec(notes_list, size=self.embedding_size, workers=8, iter=10, min_count=5)
        print("start saving")
        model.wv.save_word2vec_format("mimic.bin", binary=True)
        return




object1 = create_mimic_embedding("NOTEEVENTS.csv", 300)
object1.read_notes()
