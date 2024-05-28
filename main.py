from tqdm import tqdm
from copy import deepcopy
from tensorboardX import SummaryWriter
from torch.nn.init import xavier_uniform_

from src.utils import config
from src.utils.common import set_seed
from src.models.MOEL.model import MOEL
from src.models.MIME.model import MIME
from src.models.EMPDG.model import EMPDG
from src.models.IAMM.model import IAMM
from src.models.Transformer.model import Transformer
from src.utils.data.loader import prepare_data_seq
from src.models.common import evaluate, count_parameters, make_infinite

import os
import torch
from src.scripts.evaluate import eval_one

def make_model(vocab, dec_num):
    is_eval = config.test
    if config.model == "trs":
        model = Transformer(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            model_file_path=config.model_path if is_eval else None,
        )
    if config.model == "multi-trs":
        model = Transformer(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            is_multitask=True,
            model_file_path=config.model_path if is_eval else None,
        )
    elif config.model == "moel":
        model = MOEL(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            model_file_path=config.model_path if is_eval else None,
        )
    elif config.model == "mime":
        model = MIME(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            model_file_path=config.model_path if is_eval else None,
        )
    elif config.model == "empdg":
        model = EMPDG(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            model_file_path=config.model_path if is_eval else None,
        )
    elif config.model == "iamm":
        model = IAMM(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            model_file_path=config.model_path if is_eval else None,
        )

    model.to(config.device)

    # Intialization
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
            xavier_uniform_(p)

    print("# PARAMETERS", count_parameters(model))

    return model


def train(model, train_set, dev_set):
    min_iter_num = 10000
    max_iter_num = 17000
    check_iter = 800

    #min_iter_num = 10
    #max_iter_num = 30
    #check_iter = 10
    try:
        model.train()
        best_ppl = 100
        patient = 0
        writer = SummaryWriter(log_dir=config.save_path)
        weights_best = deepcopy(model.state_dict())
        data_iter = make_infinite(train_set)
        for n_iter in tqdm(range(1000000)):
            if "iamm" in config.model:
                loss, ppl, bce, acc, _, _ = model.train_one_batch(
                    next(data_iter), n_iter
                )
            else:
                loss, ppl, bce, acc = model.train_one_batch(next(data_iter), n_iter)

            writer.add_scalars("loss", {"loss_train": loss}, n_iter)
            writer.add_scalars("ppl", {"ppl_train": ppl}, n_iter)
            writer.add_scalars("bce", {"bce_train": bce}, n_iter)
            writer.add_scalars("accuracy", {"acc_train": acc}, n_iter)
            if config.noam:
                writer.add_scalars(
                    "lr", {"learning_rata": model.optimizer._rate}, n_iter
                )

            if n_iter < min_iter_num:
                continue

            if (n_iter + 1) % check_iter == 0:
                model.eval()
                model.is_eval = True
                model.epoch = n_iter
                loss_val, ppl_val, bce_val, acc_val, _ = evaluate(
                    model, dev_set, ty="valid", max_dec_step=50
                )
                writer.add_scalars("loss", {"loss_valid": loss_val}, n_iter)
                writer.add_scalars("ppl", {"ppl_valid": ppl_val}, n_iter)
                writer.add_scalars("bce", {"bce_valid": bce_val}, n_iter)
                writer.add_scalars("accuracy", {"acc_train": acc_val}, n_iter)
                model.train()
                model.is_eval = False
                #if ppl_val <= best_ppl:
                if ppl_val <= best_ppl or (ppl_val < 38 and acc_val > 0.50):
                    best_ppl = ppl_val
                    patient = 0
                    model.save_model(best_ppl, n_iter, acc_val)
                    weights_best = deepcopy(model.state_dict())
                else:
                    patient += 1
                if patient > 10000000 or n_iter > max_iter_num: break

#                if patient > 2:
#                    break

    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")
        model.save_model(best_ppl, n_iter, acc_val)
        weights_best = deepcopy(model.state_dict())

    return weights_best


def test(model, test_set, test_file_name=None):
    model.eval()
    model.is_eval = True
    loss_test, ppl_test, bce_test, acc_test, results = evaluate(
        model, test_set, ty="test", max_dec_step=50
    )
    if test_file_name is None:
        test_file_name = "results.txt"
    file_summary = config.save_path + "/" + test_file_name
    with open(file_summary, "w") as f:
        f.write("EVAL\tLoss\tPPL\tAccuracy\n")
        f.write(
            "{}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
                loss_test, ppl_test, bce_test, acc_test
            )
        )
        for r in results:
            f.write(r)

def get_filenamea(path, filetype):
    filetype1=filetype.upper()
    #print(filetype)
    name =[] 
    final_name = []
    for files in os.listdir(path):
        if files.startswith(filetype) or files.startswith(filetype1):
            final_name.append(files)
    return final_name

def test_model_list(vocab, dec_num, test_set):
    base_dir = "save/test/"
    #models = get_filenamea(base_dir, "IAMM")
    #print("Valid models:", models)
    result_path = base_dir + config.result_name + ".txt"
    model_name = config.test_model_name
    with open(result_path, "a") as f:
        #for model_name in models:
        config.model_path = base_dir + model_name

        print("")
        print("Current model path:", config.model_path)
        model = make_model(vocab, dec_num)
        test(model, test_set, model_name + ".txt")
        ppl, acc, d1, d2 = eval_one(model_name)
        save_title = model_name + "\tEVAL\tPPL\tAccuracy\tDist-1\tDist-2\n"
        save_context = "{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ppl, acc, d1, d2)
        print("Save title and context")
        print(save_title)
        print(save_context)
        print("")
        
        f.write(save_title)
        f.write(save_context)
        f.write("\n")

def main():
    set_seed()  # for reproducibility

    train_set, dev_set, test_set, vocab, dec_num = prepare_data_seq(
        batch_size=config.batch_size
    )

    if config.test:
        test_model_list(vocab, dec_num, test_set)
    else:
        model = make_model(vocab, dec_num)
        weights_best = train(model, train_set, dev_set)
        model.epoch = 1
        model.load_state_dict({name: weights_best[name] for name in weights_best})
        # Set batch size
        config.batch_size = 96
        model = None
        test_model_list(vocab, dec_num, test_set)
        #test(model, test_set)

os.environ["CUDA_VISOBLE_DEVICES"] = config.device_id
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if torch.cuda.is_available():
    torch.cuda.set_device(int(config.device_id))

if __name__ == "__main__":
    main()
