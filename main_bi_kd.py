# %%
import argparse
import os
import pickle
import random

import numpy as np
import torch
from texttable import Texttable
from torch import nn
from torch.nn import DataParallel
from torch.optim import AdamW, Adam
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import models
from utils.stuff import get_dataset_stats, get_bert_tokenizer, get_directories, model_name, get_gold_bincount, \
    get_metrics, draw_plot_for_response_retrieval, get_bert_bi_hyperparameter_string, load_data_loaders, \
    get_student_hyperparameter_string

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='ubuntu_data')

parser.add_argument('--epochs', default=1, type=int)

parser.add_argument('--batch', default=8, type=int)
parser.add_argument('--student_lr', default=5e-5, type=float)

parser.add_argument('--student_batch', default=16, type=int)

parser.add_argument('--load', default=2, type=int)
parser.add_argument('--load_student', default=0, type=int)
parser.add_argument('--aug', default=False)
parser.add_argument('--device', default='cuda')

parser.add_argument('--nmodel', default='MolyEncoderAggP2')
parser.add_argument('--nstudentmodel', default='BiEncoderDot2')

parser.add_argument('--ntransformer', default='distilbert')
parser.add_argument('--alpha', default=0.5, type=float)

parser.add_argument('--ntrain', type=int)

parser.add_argument('--ntest', type=int)
parser.add_argument('--skip_train', action='store_true')
parser.add_argument('--skip_test', action='store_true')
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--lr_normal', default=1e-3, type=float)
parser.add_argument('--use_weights', action='store_true')


parser.add_argument('--teacher_device', default=None)
parser.add_argument('--no_save_weights', action='store_true')
parser.add_argument('--use_cached_logits', action='store_true')

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
STUDENT_BATCH_SIZE = args[0].student_batch
LR = args[0].lr
LR = args[0].lr
N_EPOCHS = args[0].epochs
N_TRAIN = args[0].ntrain
N_TEST = args[0].ntest
BERT_TYPE = 'bert-bi'
INF = 1e13
# EXP_LR = 0.6
AUG = args[0].aug
TEACHER_LOAD_EPOCH = args[0].load
STUDENT_LOAD_EPOCH = args[0].load_student
MODEL_NAME = args[0].nmodel
MODEL_STUDENT_NAME = args[0].nstudentmodel
TRANSFORMER_NAME = args[0].ntransformer
DATASET = args[0].dataset
EPSILON = 1e-13
SKIP_TRAIN = args[0].skip_train
SKIP_TEST = args[0].skip_test
DEVICE_NAME = torch.cuda.get_device_name()
USE_WEIGHTS = args[0].use_weights
TEACHER_DEVICE = args[0].teacher_device
ALPHA = args[0].alpha
USE_CACHED_LOGITS = args[0].use_cached_logits
SHUFFLE_TRAIN = args[0].shuffle_train
if ALPHA != 1 and TEACHER_LOAD_EPOCH == 0:
    raise Exception('You forgot to set load.')

folder = 'Datasets/'
PATH = folder + DATASET
T_10, T_50, NUM_CANDIDATES, _, _ = get_dataset_stats(DATASET)

# %%

train_data_loader, valid_data_loader, test_data_loader = load_data_loaders(PATH, TRANSFORMER_NAME, AUG, STUDENT_BATCH_SIZE,
                                                                           N_TRAIN, N_TEST, SKIP_TEST, True)

# %%
tokenizer, PAD, SEP, CLS, vocab_size = get_bert_tokenizer(TRANSFORMER_NAME)

# %%


teacher_hyper_parameters = get_bert_bi_hyperparameter_string(args,
                                                             DEVICE_NAME if TEACHER_DEVICE is None else TEACHER_DEVICE)

teacher_weights_dir, _, teacher_hist_dir, _ = get_directories(BERT_TYPE, teacher_hyper_parameters)

if ALPHA != 1.0 and not USE_CACHED_LOGITS:
    print(f'Building model {MODEL_NAME} ...')
    Model = getattr(models, MODEL_NAME)
    teacher_model = Model(TRANSFORMER_NAME, vocab_size, PAD)
    # if args[0].dp:
    #     model = DataParallel(model)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    if TEACHER_LOAD_EPOCH > 0:
        print(f'Load epoch {TEACHER_LOAD_EPOCH} weights for {MODEL_NAME} ...')
        weight_path = os.path.join(teacher_weights_dir,
                                   model_name(teacher_model) + '.state_dict.epoch=' + str(TEACHER_LOAD_EPOCH) + '.bin')
        teacher_model.load_state_dict(torch.load(weight_path), strict=False)

if ALPHA != 1.0 and USE_CACHED_LOGITS:
    print(f'Loading logits from {MODEL_NAME} ...')
    teacher_logits_all = torch.load(f'{teacher_hist_dir}/{MODEL_NAME}.logits.epoch={TEACHER_LOAD_EPOCH}.batch={STUDENT_BATCH_SIZE}.bin')
    teacher_logits_data_loader = DataLoader(TensorDataset(teacher_logits_all), batch_size=STUDENT_BATCH_SIZE * STUDENT_BATCH_SIZE)

# %%

print(f'Building model {MODEL_STUDENT_NAME} ...')
StudentModel = getattr(models, MODEL_STUDENT_NAME)
# student_model = models.DualPieceRNN(teacher_model.bert1.embeddings, PAD)
# student_model = models.DualTransformerStudent(teacher_model.bert1, PAD)

student_model = StudentModel(TRANSFORMER_NAME, vocab_size, PAD)

if args[0].use_weights:
    pass

student_model = student_model.to(device)

student_hyper_parameters = get_student_hyperparameter_string(DEVICE_NAME, TEACHER_DEVICE, args[0])

student_weights_dir, student_metrics_dir, student_history_dir, student_plot_dir = get_directories(
    f'kd/{teacher_hyper_parameters}', student_hyper_parameters)

NAME = model_name(student_model) if args[0].nname is None else args[0].nname

if STUDENT_LOAD_EPOCH > 0:
    print(f'Load epoch {STUDENT_LOAD_EPOCH} weights for {MODEL_STUDENT_NAME} ...')
    weight_path = os.path.join(student_weights_dir,
                               NAME + '.state_dict.epoch=' + str(STUDENT_LOAD_EPOCH) + '.bin')
    student_model.load_state_dict(torch.load(weight_path), strict=False)


# %%

# optimizer = Adam(student_model.parameters(), lr=args[0].student_lr)
optimizer = AdamW(filter(lambda p: p.requires_grad, student_model.parameters()), lr=args[0].student_lr)


# %%

def calc_logits(c, r):
    logitses = []
    for r_i in r.transpose(0, 1):
        logits = student_model(c, r_i)
        # logits = forward(student_model, c, r_i)
        logitses.append(logits)
    logitses = torch.stack(logitses).transpose(0, 1)
    return logitses


# %%
metrics_history = []

criterion_bce = nn.BCEWithLogitsLoss(reduction='sum',
                                     pos_weight=torch.tensor([BATCH_SIZE - 1]).float().to(device)
                                     ).to(device)

criterion_mse = lambda a, b, l: ((a - b).pow(2) * (l * (l.sum() - 1) + 1) / l.sum()).sum()


if ALPHA != 1.0 and not USE_CACHED_LOGITS:
    teacher_model.train()

for i_epoch in range(1 + STUDENT_LOAD_EPOCH, 1 + N_EPOCHS + STUDENT_LOAD_EPOCH):
    student_model.train()

    loss_mse_train = 0
    loss_bce_train = 0
    acc = 0
    total_train = 0
    if USE_CACHED_LOGITS:
        teacher_logits_iter = teacher_logits_data_loader.__iter__()
    p_bar = tqdm(train_data_loader)
    for step, batch in enumerate(p_bar):

        optimizer.zero_grad()

        batch = tuple(t.to(device) for t in batch)
        c, r = batch
        b_size = c.shape[0]

        c_max_len = (~c.eq(PAD)).sum(-1).max()
        c = c[:, :c_max_len]
        r_max_len = (~r.eq(PAD)).sum(-1).max()
        r = r[:, :r_max_len]

        student_logits, labels = student_model.forward_with_negatives(c, r)  # (BB)
        loss_bce = criterion_bce(student_logits, labels)
        if 0 <= ALPHA < 1:
            if USE_CACHED_LOGITS:
                teacher_logits = next(teacher_logits_iter)[0].to(device)
            else:
                with torch.no_grad():
                    teacher_logits, _ = teacher_model.forward_with_negatives(c, r)  # (BB)
            loss_mse = criterion_mse(student_logits, teacher_logits, labels)
            combined_loss = ALPHA * loss_bce + (1 - ALPHA) * loss_mse
            loss_mse_train += loss_mse.item()
        elif ALPHA == 1:
            combined_loss = loss_bce
        else:
            raise Exception('alpha')

        combined_loss.backward()

        optimizer.step()
        loss_bce_train += loss_bce.item()

        with torch.no_grad():
            acc += student_logits.gt(0).eq(labels.bool()).long().sum().item()
        total_train += student_logits.shape[0]

        metrics_train = (
            loss_bce_train / total_train,
            loss_mse_train / total_train,
            acc / total_train
        )

        p_bar.set_description(
            '[ TRN ][ EP {:02d} ][ BCE {:.4f} MSE {:.4f} Acc {:.4f} ]'.format(i_epoch, *metrics_train))

    if not args[0].no_save_weights:
        torch.save(student_model.state_dict(),
                   os.path.join(student_weights_dir,
                                NAME + '.state_dict.epoch=' + str(i_epoch) + '.bin'))

    metrics_dev_and_test = []
    with torch.no_grad():
        student_model.eval()

        for n, data_loader in [('DEV', valid_data_loader)] + ([('TST', test_data_loader) if not SKIP_TEST else []]):
            total_valid = 0
            gold_bincounts_all = torch.zeros(NUM_CANDIDATES).long()
            p_bar = tqdm(data_loader)
            for step, batch in enumerate(p_bar):
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

                p_bar.set_description(
                    '[ {:} ][ EP {:02d} ][ R@1 {:.4f} R@- {:.4f} R@= {:.4f} MRR {:.4f} ]'.format(n, i_epoch,
                                                                                                 *metrics_dev))

            metrics_dev_and_test.extend(metrics_dev)

    metrics_history.append((*metrics_train, *metrics_dev_and_test))

    # with open(os.path.join(history_dir, 'temp_history'), 'wb') as f:
    #     pickle.dump(metrics_history, f)

# %%

#
if os.path.exists(f'{student_history_dir}/{DATASET}.ntest={args[0].ntest}.pkl'):
    with open(f'{student_history_dir}/{DATASET}.ntest={args[0].ntest}.pkl', 'rb') as f:
        baselines = pickle.load(f)
else:
    baselines = dict()

if STUDENT_LOAD_EPOCH == 0:
    baselines[NAME] = metrics_history
else:
    metrics_history = baselines[NAME][:STUDENT_LOAD_EPOCH] + metrics_history
    baselines[NAME] = metrics_history

metrics_history_all_t = [np.array(i).T for i in baselines.values()]
model_names = list(baselines.keys())


def get_table_for_response_retrieval(model_names, metrics_history_all_t, t_10, t_50):
    rows = []
    for name, values in zip(model_names, metrics_history_all_t):
        best_index = values[3].argmax()  # sort by dev r@1
        row = [name, *values[:, best_index].tolist(), f'{best_index + 1}/{values.shape[1]}']
        rows.append(row)

    rows = sorted(rows, key=lambda x: x[4])  # sort by dev r@1
    t = Texttable(max_width=0)
    t.add_rows(rows, header=False)
    t.header(('Model', 'BCE Loss', 'MSE Loss', 'Trn Acc',
              'Dev R@1', f'Dev R@{t_10}', f'Dev R@{t_50}', 'Dev MRR',
              'Tst R@1', f'Tst R@{t_10}', f'Tst R@{t_50}', 'Tst MRR',
              'Ep'))
    return t


t = get_table_for_response_retrieval(model_names, metrics_history_all_t, T_10, T_50)
print(t.draw())

# %%

with open(f'{student_metrics_dir}/{DATASET}.ntest={args[0].ntest}.txt', 'w') as f:
    print(t.draw(), file=f)

with open(f'{student_history_dir}/{DATASET}.ntest={args[0].ntest}.pkl', 'wb') as f:
    pickle.dump(baselines, f)

draw_plot_for_response_retrieval(metrics_history_all_t, model_names, student_plot_dir,
                                 f"{DATASET}.ntest={args[0].ntest}", dev_idx=3, test_idx=7)

print('Results saved to', student_hyper_parameters)
