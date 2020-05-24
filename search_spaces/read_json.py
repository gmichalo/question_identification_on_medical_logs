import json
import pandas as pd
import random


def read_hyp(json_file):
    with open(json_file) as f:
        data_strategy = json.load(f)
    return data_strategy


def assign(param):
    trial_param = {}
    for data in param:

        if data == 'optim':
            value = param[data]
        try:
            if 'sampling strategy' in param[data]:
                if param[data]['sampling strategy'] == "integer":
                    value = random.randint(param[data]['bounds'][0], param[data]['bounds'][1])
                elif param[data]['sampling strategy'] == "uniform":
                    value = random.uniform(param[data]['bounds'][0], param[data]['bounds'][1])
                elif param[data]['sampling strategy'] == "choice":
                    value = random.choice(param[data]['choice'])

        except:
            value = param[data]
        trial_param[data] = value
    return trial_param
