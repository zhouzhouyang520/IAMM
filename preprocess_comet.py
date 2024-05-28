import os
import nltk
import json
import torch
import pickle
import logging
import numpy as np
from tqdm.auto import tqdm
from src.utils import config
import torch.utils.data as data
from src.utils.common import save_config
from nltk.corpus import wordnet, stopwords
from src.utils.constants import DATA_FILES
from src.utils.constants import EMO_MAP as emo_map
from src.utils.constants import WORD_PAIRS as word_pairs
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.utils.data.loader import load_comet_data

import datetime
import time

import os

os.environ["CUDA_VISOBLE_DEVICES"] = config.device_id
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if torch.cuda.is_available():
    torch.cuda.set_device(int(config.device_id))


relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]
emotion_lexicon = json.load(open("data/NRCDict.json"))[0]
stop_words = stopwords.words("english")

def convert(n):
    return str(datetime.timedelta(seconds = n)) 

def wrapper_calc_time(print_log=True):
    """ 
    :param print_log: print logs or not.
    :return:
    """

    def wrapper(func):
        def inner_wrapper(*args, **kwargs):
            start_time = time.time()
            func_re = func(*args, **kwargs)
            run_time = time.time() - start_time
            converted_time = convert(run_time)
            if print_log:
                print(f"{func.__name__} time:", run_time, converted_time)
            return func_re

        return inner_wrapper

    return wrapper


def process_sent(sentence):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    sentence = nltk.word_tokenize(sentence)
    return sentence

def get_commonsense(comet, input_event):
    cs_list = []
    for rel in relations:
        cs_res = comet.generate(input_event, rel)
        cs_res = [process_sent(item) for item in cs_res]
        cs_list.append(cs_res)
    return cs_list

def encode(data_dict, comet, enc_type=""): # enc_type: context, target, situation
    
    res_dict = {}
    res_dict[enc_type + "_comet"] = []
    
    length = len(data_dict["context"])
    print(f"Process data length: {length}")
     
    for i in tqdm(range(length)):
        context = data_dict["context"][i]
        target = data_dict["target"][i]
        situation = data_dict["situation"][i]
        # context
        if enc_type == "context":
            contex_cs_list = []
            for c in context:
                item = " ".join(c)
                cs_list = get_commonsense(comet, item)
                contex_cs_list.append(cs_list)
            res_dict["context_comet"].append(contex_cs_list)

        elif enc_type == "target":
            # target
            item = " ".join(target)
            target_cs_list = get_commonsense(comet, item)
            res_dict["target_comet"].append(target_cs_list)
        
        elif enc_type == "situation":
            # situation
            item = " ".join(situation)
            situation_cs_list = get_commonsense(comet, item)
            res_dict["situation_comet"].append(situation_cs_list)
    print("length:", len(res_dict[enc_type + "_comet"]))
    return res_dict

@wrapper_calc_time(print_log=True)
def generate_comet_text(data_dict):
    from src.utils.comet import Comet
    comet = Comet("data/Comet", config.device)
    res_dict = encode(data_dict, comet, config.enc_type)
    return res_dict

def load_dataset(data_dir):
    cache_file = f"{data_dir}/dataset_preproc.p"

    print("LOADING empathetic_dialogue")
    with open(cache_file, "rb") as f:
        [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    return data_tra, data_val, data_tst, vocab

def save_data(data_name, data):
    with open(data_name, "wb") as f:
        pickle.dump(data, f)
        print("Saved PICKLE")

#@wrapper_calc_time(print_log=True)
def process_data():
    data_dir = config.data_dir
    data_tra, data_val, data_tst, vocab = load_dataset(data_dir)

    print("Building dataset...")
    data_train = generate_comet_text(data_tra)
    data_dev = generate_comet_text(data_val)
    data_test = generate_comet_text(data_tst)

    save_cache_file = f"{data_dir}/comet_"+ config.enc_type + "_preproc.p"
    data = [data_train, data_dev, data_test]
    save_data(save_cache_file, data)

def assign(data_dict, comet_cxt, comet_sit, comet_trg):
    print(f"data_dict cxt length check: ", len(data_dict["context"]), len(comet_cxt["context_comet"]))
    data_dict["comet_cxt"] = comet_cxt["context_comet"]
    print(f"data_dict sit length check: ", len(data_dict["context"]), len(comet_sit["situation_comet"]))
    data_dict["comet_sit"] = comet_sit["situation_comet"]
    print(f"data_dict trg length check: ", len(data_dict["context"]), len(comet_trg["target_comet"]))
    data_dict["comet_trg"] = comet_trg["target_comet"]

def merge_data():
    data_dir = config.data_dir
    print(f"Loading COMET dataset.")
    comet_cxt = load_comet_data("comet_context_preproc.p")
    comet_sit = load_comet_data("comet_situation_preproc.p")
    comet_trg = load_comet_data("comet_target_preproc.p")
    print(f"comet_cxt type: {type(comet_cxt[0])}")

    print(f"Loading empathetic dialogue dataset.")
    data_tra, data_val, data_tst, vocab = load_dataset(data_dir)
    # data = [data_train, data_dev, data_test]
    assign(data_tra, comet_cxt[0], comet_sit[0], comet_trg[0])
    assign(data_val, comet_cxt[1], comet_sit[1], comet_trg[1])
    assign(data_tst, comet_cxt[2], comet_sit[2], comet_trg[2])

    cache_file = f"{data_dir}/dataset_preproc.p"
    save_data(cache_file, [data_tra, data_val, data_tst, vocab])

# Step 1: process data.
#process_data()

# Step 2: merge context, situation, and target data.
merge_data()
