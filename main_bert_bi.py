# %%
import argparse
import os
import pickle
import random

import numpy as np
import torch
import transformers
from torch import nn
from torch.nn import DataParallel
from torch.optim import AdamW, Adam
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import models
from utils.stuff import get_dataset_stats, get_bert_tokenizer, get_directories, model_name, get_gold_bincount, \
    get_metrics, get_table_for_response_retrieval, draw_plot_for_response_retrieval, load_data_loaders, \
    get_bert_bi_hyperparameter_string

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='ubuntu_data')
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--batch', default=8, type=int)
parser.add_argument('--load', default=0, type=int)
parser.add_argument('--aug', action='store_true')
parser.add_argument('--device', default='cuda')
parser.add_argument('--nmodel', default='MolyEncoderAggP2')
parser.add_argument('--ntransformer', default='distilbert')
parser.add_argument('--ntrain', type=int)
parser.add_argument('--ntest', type=int)
parser.add_argument('--skip_train', action='store_true')
parser.add_argument('--skip_test', action='store_true')
parser.add_argument('--freeze_transformer', action='store_true')
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--lr_normal', default=1e-3, type=float)
parser.add_argument('--no_save_weights', action='store_true')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--shuffle_train', action='store_true')
parser.add_argument('--nname')

args = parser.parse_known_args()
print(args)

# %%

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SEED = args[0].seed
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# %%
device = torch.device(args[0].device if torch.cuda.is_available() else "cpu")
BATCH_SIZE = args[0].batch
LR_FINE_TUNE = args[0].lr
LR_NORMAL = args[0].lr_normal
N_EPOCHS = args[0].epochs
N_TRAIN = args[0].ntrain
N_TEST = args[0].ntest
BERT_TYPE = 'bert-bi'
INF = 1e13
# EXP_LR = 0.6
AUG = args[0].aug
START_EPOCH = args[0].load
MODEL_NAME = args[0].nmodel
TRANSFORMER_NAME = args[0].ntransformer
DATASET = args[0].dataset
EPSILON = 1e-13
SKIP_TRAIN = args[0].skip_train
SKIP_TEST = args[0].skip_test
DEVICE_NAME = torch.cuda.get_device_name()
FREEZE_TRANSFORMER = args[0].freeze_transformer
SHUFFLE_TRAIN = args[0].shuffle_train
folder = 'Datasets/'
PATH = folder + DATASET
T_10, T_50, NUM_CANDIDATES, _, _ = get_dataset_stats(DATASET)


# %%

train_data_loader, valid_data_loader, test_data_loader = load_data_loaders(PATH, TRANSFORMER_NAME, AUG, BATCH_SIZE,
                                                                           N_TRAIN, N_TEST, SKIP_TEST, SHUFFLE_TRAIN)
# %%
tokenizer, PAD, SEP, CLS, vocab_size = get_bert_tokenizer(TRANSFORMER_NAME)

# %%
print(f'Building model {MODEL_NAME} ...')
Model = getattr(models, MODEL_NAME)
model = Model(TRANSFORMER_NAME, vocab_size, PAD)
# if args[0].dp:
#     model = DataParallel(model)
model = model.to(device)
try:
    model.bert1.requires_grad_(not FREEZE_TRANSFORMER)
except AttributeError:
    pass

NAME = model_name(model) if args[0].nname is None else args[0].nname
# %%

# optimizer = AdamW([
#     {'params': [param for name, param in model.named_parameters() if 'bert' in name], 'lr': LR_FINE_TUNE},
#     {'params': [param for name, param in model.named_parameters() if 'bert' not in name], 'lr': LR_NORMAL},
# ])

if not FREEZE_TRANSFORMER:
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FINE_TUNE)
else:
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_NORMAL)

# optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FINE_TUNE)
# optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_NORMAL)
# scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=train_data_loader.__len__() * N_EPOCHS)
# %%

hyper_parameters_string = get_bert_bi_hyperparameter_string(args, DEVICE_NAME)

print('Output directory:', hyper_parameters_string)

weights_dir, metrics_dir, history_dir, plot_dir = get_directories(BERT_TYPE, hyper_parameters_string)

if START_EPOCH > 0:
    weight_path = os.path.join(weights_dir, NAME + '.state_dict.epoch=' + str(START_EPOCH) + '.bin')
    print('loading weights of previous epoch', START_EPOCH)
    model.load_state_dict(torch.load(weight_path), strict=False)


# %%


def calc_logits(c, r):
    logitses = []
    for r_i in r.transpose(0, 1):
        logits = model(c, r_i)
        logitses.append(logits)
    logitses = torch.stack(logitses).transpose(0, 1)
    # logitses = model.forward_with_candidates(c, r)
    return logitses


# %%
metrics_history = []

criterion = nn.BCEWithLogitsLoss(reduction='sum',
                                 pos_weight=torch.tensor([BATCH_SIZE - 1]).float().to(device)
                                 ).to(device)

for i_epoch in range(1 + START_EPOCH, 1 + N_EPOCHS + START_EPOCH):

    if not SKIP_TRAIN:
        loss_train = 0
        acc = 0
        total_train, nb_tr_steps = 0, 0

        p_bar = tqdm(train_data_loader)
        model.train()
        for step, batch in enumerate(p_bar):
            optimizer.zero_grad()

            batch = tuple(t.to(device) for t in batch)
            c, r = batch
            b_size = c.shape[0]

            c_max_len = (~c.eq(PAD)).sum(-1).max()
            c = c[:, :c_max_len]
            r_max_len = (~r.eq(PAD)).sum(-1).max()
            r = r[:, :r_max_len]

            logits, labels = model.forward_with_negatives(c, r)  # (BB)

            # logits = torch.cat([logits[:b_size], logits[b_size:b_size + 1], logits.reshape(b_size, -1).diag()[1:]])

            loss = criterion(logits, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # scheduler.step()
            loss_train += loss.item()
            with torch.no_grad():
                acc += logits.gt(0).eq(labels.bool()).long().sum().item()
            total_train += logits.shape[0]

            metrics_train = (
                loss_train / total_train,
                acc / total_train
            )

            p_bar.set_description('[ TRN ][ EP {:02d} ][ Lss {:.4f} Acc {:.4f} ]'.format(i_epoch, *metrics_train))
            # break

        if not args[0].no_save_weights:
            torch.save(model.state_dict(),
                       os.path.join(weights_dir, NAME + '.state_dict.epoch=' + str(i_epoch) + '.bin'))

    metrics_dev_and_test = []
    with torch.no_grad():
        model.eval()

        for n, data_loader in [('DEV', valid_data_loader)] + ([] if SKIP_TEST else [('TST', test_data_loader)]):
            total_valid = 0
            gold_bincounts_all = torch.zeros(NUM_CANDIDATES).long()
            p_bar_d = tqdm(data_loader)
            for step, batch in enumerate(p_bar_d):
                batch = tuple(t.to(device) for t in batch)
                c, r = batch

                c_max_len = (~c.eq(PAD)).sum(-1).max()
                c = c[:, :c_max_len]
                r_max_len = (~r.eq(PAD)).sum(-1).max()
                r = r[:, :, :r_max_len]

                logitses = calc_logits(c, r)

                a, b = get_gold_bincount(logitses.cpu())
                gold_bincounts_all += a
                total_valid += c.shape[0]

                metrics_dev = get_metrics(gold_bincounts_all, T_10, T_50)

                p_bar_d.set_description(
                    '[ {:} ][ EP {:02d} ][ R@1 {:.4f} R@- {:.4f} R@= {:.4f} MRR {:.4f} ]'.format(n, i_epoch,
                                                                                                 *metrics_dev))

            metrics_dev_and_test.extend(metrics_dev)

    if not SKIP_TRAIN:
        metrics_history.append((*metrics_train, *metrics_dev_and_test))
        with open(os.path.join(history_dir, 'temp_history'), 'wb') as f:
            pickle.dump(metrics_history, f)

# %%


if os.path.exists(os.path.join(history_dir, f"{DATASET}.ntest={args[0].ntest}.pkl")):
    with open(os.path.join(history_dir, f"{DATASET}.ntest={args[0].ntest}.pkl"), 'rb') as f:
        baselines = pickle.load(f)
else:
    baselines = dict()

if START_EPOCH == 0:
    baselines[NAME] = metrics_history
else:
    metrics_history = baselines[NAME][:START_EPOCH] + metrics_history
    baselines[NAME] = metrics_history

metrics_history_all_t = [np.array(i).T for i in baselines.values()]
model_names = list(baselines.keys())

t = get_table_for_response_retrieval(model_names, metrics_history_all_t, T_10, T_50)
print(t.draw())

# %%
if not SKIP_TRAIN:
    with open(f'{metrics_dir}/{DATASET}.ntest={args[0].ntest}.txt', 'w') as f:
        print(t.draw(), file=f)

    with open(f'{history_dir}/{DATASET}.ntest={args[0].ntest}.pkl', 'wb') as f:
        pickle.dump(baselines, f)

    draw_plot_for_response_retrieval(metrics_history_all_t, model_names, plot_dir, f"{DATASET}.ntest={args[0].ntest}")

    print('Results saved to', hyper_parameters_string)
