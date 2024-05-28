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


relations = ["xIntent", "xNeed", "xWant", "xEffect", "xReact"]
emotion_lexicon = json.load(open("data/NRCDict.json"))[0]
stop_words = stopwords.words("english")


class Lang:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return None


def process_sent(sentence):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    sentence = nltk.word_tokenize(sentence)
    return sentence


def get_commonsense(comet, item, data_dict):
    cs_list = []
    input_event = " ".join(item)
    for rel in relations:
        cs_res = comet.generate(input_event, rel)
        cs_res = [process_sent(item) for item in cs_res]
        cs_list.append(cs_res)

    data_dict["utt_cs"].append(cs_list)


def encode_ctx(vocab, items, data_dict, comet):
    for ctx in tqdm(items):
        ctx_list = []
        e_list = []
        for i, c in enumerate(ctx):
            item = process_sent(c)
            ctx_list.append(item)
            vocab.index_words(item)
            ws_pos = nltk.pos_tag(item)  # pos
            for w in ws_pos:
                w_p = get_wordnet_pos(w[1])
                if w[0] not in stop_words and (
                    w_p == wordnet.ADJ or w[0] in emotion_lexicon
                ):
                    e_list.append(w[0])
            if i == len(ctx) - 1:
                get_commonsense(comet, item, data_dict)

        data_dict["context"].append(ctx_list)
        data_dict["emotion_context"].append(e_list)


def encode(vocab, files):
    from src.utils.comet import Comet

    data_dict = {
        "context": [],
        "target": [],
        "emotion": [],
        "situation": [],
        "emotion_context": [],
        "utt_cs": [],
    }
    comet = Comet("data/Comet", config.device)

    for i, k in enumerate(data_dict.keys()):
        items = files[i]
        if k == "context":
            encode_ctx(vocab, items, data_dict, comet)
        elif k == "emotion":
            data_dict[k] = items
        else:
            for item in tqdm(items):
                item = process_sent(item)
                data_dict[k].append(item)
                vocab.index_words(item)
        if i == 3:
            break
    assert (
        len(data_dict["context"])
        == len(data_dict["target"])
        == len(data_dict["emotion"])
        == len(data_dict["situation"])
        == len(data_dict["emotion_context"])
        == len(data_dict["utt_cs"])
    )

    return data_dict


def read_files(vocab):
    files = DATA_FILES(config.data_dir)
    train_files = [np.load(f, allow_pickle=True) for f in files["train"]]
    dev_files = [np.load(f, allow_pickle=True) for f in files["dev"]]
    test_files = [np.load(f, allow_pickle=True) for f in files["test"]]

    data_train = encode(vocab, train_files)
    data_dev = encode(vocab, dev_files)
    data_test = encode(vocab, test_files)

    return data_train, data_dev, data_test, vocab

#def load_situation_vec():
#    data_dir = config.data_dir
#    cache_file = f"{data_dir}/situation_vector_out.p"
#    print("LOADING situation of empathetic dialogue")
#    with open(cache_file, "rb") as f:
#        [sit_tra, sit_val, sit_tst] = pickle.load(f)
#    return sit_tra, sit_val, sit_tst

def load_comet_data(file_name):
    data_dir = config.data_dir
    cache_file = f"{data_dir}/{file_name}"
    print(f"LOADING COMET data of empathetic dialogue: {file_name}")
    with open(cache_file, "rb") as f:
        [comet_tra, comet_val, comet_tst] = pickle.load(f)
    return comet_tra, comet_val, comet_tst

def load_dataset():
    data_dir = config.data_dir
    cache_file = f"{data_dir}/dataset_preproc.p"
    if os.path.exists(cache_file):
        print("LOADING empathetic_dialogue")
        with open(cache_file, "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    else:
        print("Building dataset...")
        data_tra, data_val, data_tst, vocab = read_files(
            vocab=Lang(
                {
                    config.UNK_idx: "UNK",
                    config.PAD_idx: "PAD",
                    config.EOS_idx: "EOS",
                    config.SOS_idx: "SOS",
                    config.USR_idx: "USR",
                    config.SYS_idx: "SYS",
                    config.CLS_idx: "CLS",
                }
            )
        )
        with open(cache_file, "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab], f)
            print("Saved PICKLE")

    for i in range(10):
        print("[situation]:", " ".join(data_tra["situation"][i]))
        print("[emotion]:", data_tra["emotion"][i])
        print("[context]:", [" ".join(u) for u in data_tra["context"][i]])
        print("[target]:", " ".join(data_tra["target"][i]))
        print(" ")
    return data_tra, data_val, data_tst, vocab


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, vocab):#, situation_vec):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data
        self.emo_map = emo_map
        self.analyzer = SentimentIntensityAnalyzer()
        #self.situation_vec = situation_vec
        #self.oovs = []

    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["context_text"] = self.data["context"][index]
        item["situation_text"] = self.data["situation"][index]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = self.data["emotion"][index]
        item["emotion_context"] = self.data["emotion_context"][index]

        item["context_emotion_scores"] = self.analyzer.polarity_scores(
            " ".join(self.data["context"][index][0])
        )

        item["context"], item["context_mask"], item["context_ext"] = self.preprocess(item["context_text"])
        item["target"] = self.preprocess(item["target_text"], anw=True)
        item["emotion"], item["emotion_label"] = self.preprocess_emo(
            item["emotion_text"], self.emo_map
        )

        # The process is same with situaion, so set sit to True.
        item["emotion_context"], item["emotion_context_mask"] = self.preprocess(item["emotion_context"], sit=True)

        item["cs_text"] = self.data["utt_cs"][index]
        item["x_intent_txt"] = item["cs_text"][0]
        item["x_need_txt"] = item["cs_text"][1]
        item["x_want_txt"] = item["cs_text"][2]
        item["x_effect_txt"] = item["cs_text"][3]
        item["x_react_txt"] = item["cs_text"][4]

        item["x_intent"] = self.preprocess([item["x_intent_txt"]], cs=True)
        item["x_need"] = self.preprocess([item["x_need_txt"]], cs=True)
        item["x_want"] = self.preprocess([item["x_want_txt"]], cs=True)
        item["x_effect"] = self.preprocess([item["x_effect_txt"]], cs=True)
        item["x_react"] = self.preprocess([item["x_react_txt"]], cs="react")

        item["situation"], item["situation_ext"] = self.preprocess(item["situation_text"], sit=True)
        #item["situation_vec"] = self.situation_vec[index]

        self.relations = ["intent", "need", "want", "effect", "react"]
        self.valid_data_type = ["c", "s"]
        self.process_comet_data(item, self.data["comet_cxt"][index], data_type="c")
        self.process_comet_data(item, self.data["comet_sit"][index], data_type="s")

        return item

    def process_comet_data(self, item, data, data_type): # data_type: c(context), s(situation), t(target)
        # get text data

        for i, r in enumerate(self.relations):
            if data_type == "c":
                r_data = [d[i] for d in data] # if preprocess context data, obtain the relation data of last history sentences.
            else:
                r_data = [data[i]]
            item[f"{data_type}_{r}_txt"] = r_data
            flag = r if r == "react" else True
            item[f"{data_type}_{r}"] = self.preprocess(item[f"{data_type}_{r}_txt"], cs=flag)

    def process_oov(self, sentence, ids, oovs):
        for w in sentence:
            if w in self.vocab.word2index:
                i = self.vocab.word2index[w]
                ids.append(i)
            else:
                if w not in oovs:
                    oovs.append(w)
                oov_num = oovs.index(w)
                ids.append(len(self.vocab.word2index) + oov_num)

    def process_context_oov(self, context):  #
        ids, oovs = [], []
        for si, sentence in enumerate(context):
            self.process_oov(sentence, ids, oovs)
        return ids 

    def preprocess(self, arr, anw=False, cs=None, emo=False, sit=False):
        """Converts words to ids."""
        if anw:
            sequence = [
                self.vocab.word2index[word]
                if word in self.vocab.word2index
                else config.UNK_idx
                for word in arr
            ] + [config.EOS_idx]

            return torch.LongTensor(sequence)
        elif cs: 
            res = []
            for utter in arr:
                sequence = [config.CLS_idx] if cs != "react" else []
                for sent in utter:
                    sequence += [
                        self.vocab.word2index[word]
                        for word in sent
                        if word in self.vocab.word2index and word not in ["to", "none"]
                    ]   
                res.append(torch.LongTensor(sequence))
            return res 
        elif emo:
            x_emo = [config.CLS_idx]
            x_emo_mask = [config.CLS_idx]
            for i, ew in enumerate(arr):
                x_emo += [
                    self.vocab.word2index[ew]
                    if ew in self.vocab.word2index
                    else config.UNK_idx
                ]
                x_emo_mask += [self.vocab.word2index["CLS"]]

            assert len(x_emo) == len(x_emo_mask)
            return torch.LongTensor(x_emo), torch.LongTensor(x_emo_mask)

        elif sit:
            ids, oovs = [], []
            self.process_oov(arr, ids, oovs)
            ids = [config.CLS_idx] + ids
            sequence = [config.CLS_idx] + [
                self.vocab.word2index[word]
                if word in self.vocab.word2index
                else config.UNK_idx
                for word in arr
            ]

            return torch.LongTensor(sequence), torch.LongTensor(ids)

        else:
            x_dial = []
            x_mask = []
            x_dial_ext = []
            for i, sentence in enumerate(arr):
                dial = [config.CLS_idx] + [
                    self.vocab.word2index[word]
                    if word in self.vocab.word2index
                    else config.UNK_idx
                    for word in sentence
                ]
                ext_ids, oovs = [], []
                self.process_oov(sentence, ext_ids, oovs)
                ext = [config.CLS_idx] + ext_ids 
                x_dial_ext.append(torch.LongTensor(ext)) 

                spk = (
                    self.vocab.word2index["USR"]
                    if i % 2 == 0
                    else self.vocab.word2index["SYS"]
                )
                mask = [config.CLS_idx] + [spk for _ in range(len(sentence))]
                x_dial.append(torch.LongTensor(dial))
                x_mask.append(torch.LongTensor(mask))
            assert len(x_dial) == len(x_mask)

            return x_dial, x_mask, x_dial_ext

    def preprocess_emo(self, emotion, emo_map):
        program = [0] * len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion]


def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(
            len(sequences), max(lengths)
        ).long()  ## padding index 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def merge_extra(sequences):
        lengths = [len(seq) for seq in sequences]
        max_dims = lengths[0] 
        padded_seqs = torch.zeros(
            len(sequences), max_dims
        ).float()  ## padding values 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def context_merge(sequences):
        lengths = [[len(seq) for seq in ss] for ss in sequences]
        bz = len(lengths)
        max_len = max([max(s) for s in lengths])
        sent_num = max([len(s) for s in lengths])
        padded_seqs = torch.ones(bz, sent_num, max_len).long()
        for i, seq in enumerate(sequences):
            for j, ss in enumerate(seq):
                ## padding index 1
                end = lengths[i][j]
                padded_seqs[i, j, :end] = ss[:end]
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x["context"]), reverse=True)  ## sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    input_batch, input_lengths = context_merge(item_info["context"])

    mask_input, mask_input_lengths = context_merge(item_info["context_mask"])
    emotion_batch, emotion_lengths = merge(item_info["emotion_context"])
    
    input_ext_batch, _ = context_merge(item_info["context_ext"])

    ## Target
    target_batch, target_lengths = merge(item_info["target"])
    situation_batch, situation_lengths = merge(item_info["situation"])
    situation_ext_batch, situation_ext_lengths = merge(item_info["situation_ext"])

    #situation_vec_batch = torch.from_numpy(np.array(item_info["situation_vec"]))

    input_batch = input_batch.to(config.device)
    mask_input = mask_input.to(config.device)
    target_batch = target_batch.to(config.device)
    situation_batch = situation_batch.to(config.device)
    #situation_vec_batch = situation_vec_batch.to(config.device)
    situation_ext_batch = situation_ext_batch.to(config.device)
    input_ext_batch = input_ext_batch.to(config.device)
    d = {}
    d["input_batch"] = input_batch
    d["input_ext_batch"] = input_ext_batch
    d["input_lengths"] = input_lengths
    d["mask_input"] = mask_input
    d["target_batch"] = target_batch
    d["target_lengths"] = torch.LongTensor(target_lengths)
    d["emotion_context_batch"] = emotion_batch.to(config.device)

    d["situation_batch"] = situation_batch
    d["situation_lengths"] = torch.LongTensor(situation_lengths)
    #d["situation_vec_batch"] = situation_vec_batch
    d["situation_ext_batch"] = situation_ext_batch

    ##program
    d["target_program"] = item_info["emotion"]
    d["program_label"] = item_info["emotion_label"]

    ##text
    d["input_txt"] = item_info["context_text"]
    d["target_txt"] = item_info["target_text"]
    d["program_txt"] = item_info["emotion_text"]
    d["situation_txt"] = item_info["situation_text"]

    d["context_emotion_scores"] = item_info["context_emotion_scores"]

#    relations = ["x_intent", "x_need", "x_want", "x_effect", "x_react"]
#    for r in relations:
#        pad_batch, _ = merge(item_info[r])
#        pad_batch = pad_batch.to(config.device)
#        d[r] = pad_batch
#        d[f"{r}_txt"] = item_info[f"{r}_txt"]

    # comet data
    relations = ["intent", "need", "want", "effect", "react"]
    valid_data_type = ["c", "s"]
    for prefix in valid_data_type:
        for r in relations:
            r =  f"{prefix}_{r}"
            #print(f"batch r1: {r}")
            #print(f"relations: {r}", item_info[r])
            pad_batch, _ = context_merge(item_info[r])
            #print(f"pad_batch: {pad_batch.shape}, {pad_batch}")
            pad_batch = pad_batch.to(config.device)
            d[r] = pad_batch
            r =  f"{r}_txt"
            #print(f"batch r2: {r}")
            d[r] = item_info[r]

#    print("batch x_intent:", item_info["x_intent"])
#    print("batch x_need:", item_info["x_need"])
#    print("batch x_want:", item_info["x_want"])
#    print("batch x_effect:", item_info["x_effect"])
#    print("batch x_react:", item_info["x_react"])
#    print("batch =================================")
#
#    print("batch c_intent:", d["c_intent"])
#    print("batch c_need:", d["c_need"])
#    print("batch c_want:", d["c_want"])
#    print("batch c_effect:", d["c_effect"])
#    print("batch c_react:", d["c_react"])
#    print("batch =================================")
#
#    print("batch s_intent:", d["s_intent"])
#    print("batch s_need:", d["s_need"])
#    print("batch s_want:", d["s_want"])
#    print("batch s_effect:", d["s_effect"])
#    print("batch s_react:", d["s_react"])
#    print("batch =================================")

    return d

def load_idf(load_path="data/data/updated_vocab_idf.json"):
    with open(load_path, 'r') as f:
        print("LOADING vocabulary idf")
        idf_json = json.load(f)
    max_idf = 0.
    mean_idf = 0.0 
    min_idf = 99.0
    for key in idf_json:
        idf = idf_json[key]
        if max_idf < idf:
            max_idf = idf 
        if min_idf > idf:
            min_idf = idf 
        mean_idf += idf 
    print(f"Max idf: {max_idf}, Mean idf: {mean_idf / len(idf_json)}, Min idf: {min_idf}")
    return idf_json 

def prepare_data_seq(batch_size=32):

    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()
    #sit_tra, sit_val, sit_tst = load_situation_vec()

    logging.info("Vocab  {} ".format(vocab.n_words))

    dataset_train = Dataset(pairs_tra, vocab)
    data_loader_tra = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

#    for key in pairs_tst:
#        pairs_tst[key] = pairs_tst[key][:10]

    dataset_valid = Dataset(pairs_val, vocab)
    data_loader_val = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dataset_test = Dataset(pairs_tst, vocab)
    data_loader_tst = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    save_config()
    return (
        data_loader_tra,
        data_loader_val,
        data_loader_tst,
        vocab,
        len(dataset_train.emo_map),
    )
