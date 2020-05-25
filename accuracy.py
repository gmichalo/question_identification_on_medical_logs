import csv
import pandas as pd
import statistics
import json


def create_accuracy_dictionary(class_names=[1, 2, 3]):
    accuracy_dictionary = {}
    for classes in class_names:
        accuracy_dictionary[classes] = {}
        accuracy_dictionary[classes]["precision"] = []
        accuracy_dictionary[classes]["recall"] = []
        accuracy_dictionary[classes]["f1"] = []
    accuracy_dictionary["accuracy"] = {}
    accuracy_dictionary["accuracy"]["result"] = []
    accuracy_dictionary["macro_avg"] = {}
    accuracy_dictionary["macro_avg"]["precision"] = []
    accuracy_dictionary["macro_avg"]["recall"] = []
    accuracy_dictionary["macro_avg"]["f1"] = []
    accuracy_dictionary["weightmacro _avg"] = {}
    accuracy_dictionary["weightmacro _avg"]["precision"] = []
    accuracy_dictionary["weightmacro _avg"]["recall"] = []
    accuracy_dictionary["weightmacro _avg"]["f1"] = []
    accuracy_dictionary["time"] = {}
    accuracy_dictionary["time"]['result'] = []

    accuracy_dictionary_val = {}
    for classes in class_names:
        accuracy_dictionary_val[classes] = {}
        accuracy_dictionary_val[classes]["precision"] = []
        accuracy_dictionary_val[classes]["recall"] = []
        accuracy_dictionary_val[classes]["f1"] = []
    accuracy_dictionary_val["accuracy"] = {}
    accuracy_dictionary_val["accuracy"]["result"] = []
    accuracy_dictionary_val["macro_avg"] = {}
    accuracy_dictionary_val["macro_avg"]["precision"] = []
    accuracy_dictionary_val["macro_avg"]["recall"] = []
    accuracy_dictionary_val["macro_avg"]["f1"] = []
    accuracy_dictionary_val["weightmacro _avg"] = {}
    accuracy_dictionary_val["weightmacro _avg"]["precision"] = []
    accuracy_dictionary_val["weightmacro _avg"]["recall"] = []
    accuracy_dictionary_val["weightmacro _avg"]["f1"] = []
    accuracy_dictionary_val["time"] = {}
    accuracy_dictionary_val["time"]['result'] = []

    return accuracy_dictionary, accuracy_dictionary_val


def write_csv(path_results, accuracy_dictionary):
    a_file = open(path_results, "a+")
    writer = csv.writer(a_file)

    for key, value in accuracy_dictionary.items():

        for key_temp, value_temp in accuracy_dictionary[key].items():
            average = 0
            temp = [float(i) for i in accuracy_dictionary[key][key_temp]]
            ds = statistics.stdev(temp)
            for i in accuracy_dictionary[key][key_temp]:
                average = average + float(i)
            average = average / len(accuracy_dictionary[key][key_temp])
            string_temp = "|".join(accuracy_dictionary[key][key_temp])
            string_temp = string_temp + "|" + str(round(average, 4)) + "+/-" + str(round(ds, 4))

            writer.writerow([str(key) + "_" + str(key_temp), string_temp, "(average and  sd the last)"])
    a_file.close()


def write_param(path_results, parameters):
    a_file = open(path_results, "a+")
    writer = csv.writer(a_file)
    writer.writerow(["parameters:" + parameters])


def write_hyper(path_results, path_results_json, f1_list, hyper_paremeters):
    a_file = open(path_results, "w+")
    writer = csv.writer(a_file)
    string_temp = ",".join(f1_list)
    writer.writerow(["f1_list:" + string_temp])
    a_file.close()
    with open(path_results_json, 'w') as fp:
        fp.write("\n")
        json.dump(hyper_paremeters, fp)


