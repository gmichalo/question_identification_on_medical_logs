import pandas as pd

"""
prepare dataset for the ctakes tool
"""
def read_csv(path1, path2):
    df = pd.read_csv(path1, sep=",")
    questions = []
    for row in df.iterrows():
        questions.append(row[1][0] + "<question>")
    file = open(path2, "w")
    for que in questions:
        file.write(que + "\n")


read_csv("../../dataset_input/questions.csv", "ctakes/ctakes_input/questions.txt")
