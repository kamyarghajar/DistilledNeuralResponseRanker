import os
import pickle

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from transformers import BertTokenizer, AlbertTokenizer, RobertaTokenizer
from texttable import Texttable
import seaborn as sns

sns.set()


def get_gold_bincount(logits_s):
    predictions_sorted = torch.argsort(logits_s.cpu(), descending=True)

    gold_ranks = torch.nonzero(predictions_sorted == 0)[:, 1]

    gold_bincount = torch.bincount(gold_ranks, minlength=logits_s.shape[1])

    return gold_bincount, gold_ranks


def get_metrics(gold_bincount, t_10, t_50):
    total = gold_bincount.sum().item()
    recall_at_1 = gold_bincount[0].item() / total
    recall_at_10 = gold_bincount[0:t_10].sum().item() / total
    recall_at_50 = gold_bincount[0:t_50].sum().item() / total
    mrr = ((1.0 / (torch.arange(len(gold_bincount)).float() + 1)) * gold_bincount.float()).sum().item() / total
    return recall_at_1, recall_at_10, recall_at_50, mrr


def model_name(model):
    try:
        a = model.name
    except AttributeError:
        a = type(model).__name__
        if a == 'DataParallel':
            a = model_name(model.module)
    return a


def get_dataset_stats(dataset):
    if 'ubuntu_data' in dataset:
        t_10 = 2
        t_50 = 5
        num_cand = 10
        max_len = 200
        max_sent = 60
    elif 'dstc7' in dataset:
        t_10 = 10
        t_50 = 50
        num_cand = 100
        max_len = 200
        max_sent = 60
    elif dataset == 'msdialog':
        t_10 = 2
        t_50 = 5
        num_cand = 10
        max_len = 300
        max_sent = 100

    elif dataset == 'convai2':
        t_10 = 2
        t_50 = 5
        num_cand = 20
        max_len = 300
        max_sent = 70

    elif dataset == 'mantis':
        t_10 = 2
        t_50 = 5
        num_cand = 11
        max_len = 300
        max_sent = 80
    else:
        raise Exception('no such dataset')

    return t_10, t_50, num_cand, max_len, max_sent


def get_bert_tokenizer(transformer_name):
    if transformer_name == 'distilbert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        pad, sep, cls = tokenizer.vocab['[PAD]'], tokenizer.vocab['[SEP]'], tokenizer.vocab['[CLS]']
    elif transformer_name == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True)
        pad, sep, cls = tokenizer.pad_token_id, tokenizer.sep_token_id, tokenizer.cls_token_id
    elif transformer_name == 'distilroberta':
        tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base', do_lower_case=True)
        pad, sep, cls = tokenizer.pad_token_id, tokenizer.sep_token_id, tokenizer.cls_token_id
    else:
        raise Exception('No such transformer')
    tokenizer.add_special_tokens({
        tokenizer.SPECIAL_TOKENS_ATTRIBUTES[-1]: ['[EOU]', '[NUM]', '[PATH]', '[URL]'],
    })

    vocab_size = tokenizer.vocab_size + len(tokenizer.additional_special_tokens)
    return tokenizer, pad, sep, cls, vocab_size


def get_directories(subdir, hyper_parameters):
    weights_dir = os.path.join('outputs', subdir, hyper_parameters, 'weights')
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    metrics_dir = os.path.join('outputs', subdir, hyper_parameters, 'metrics')
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    history_dir = os.path.join('outputs', subdir, hyper_parameters, 'histories')
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    plot_dir = os.path.join('outputs', subdir, hyper_parameters, 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    return weights_dir, metrics_dir, history_dir, plot_dir


def draw_plot_for_response_retrieval(metrics_history_all_t, model_names, plot_dir, dataset, dev_idx=2, test_idx=6):
    plt.clf()
    ax = plt.gca()
    for history in metrics_history_all_t:
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(range(len(history[dev_idx])), history[dev_idx], color=color, marker='o', alpha=0.7)
        plt.plot(range(len(history[test_idx])), history[test_idx], color=color, marker='^', linestyle='dashed', alpha=0.7)

    plt.ylabel('Recall@1')
    plt.xlabel('Epoch')

    legends = []
    for i in model_names:
        for j in ['valid', 'test']:
            legends.append(' '.join((i, j)))

    plt.legend(legends, loc='upper left',
               bbox_to_anchor=(0, -0.2),
               fancybox=True, shadow=True, ncol=2)
    plt.title(f"Comparison of ANN Models for {dataset}")

    plt.savefig(f'{plot_dir}/{dataset}.png', dpi=400, bbox_inches='tight')
    plt.savefig(f'{plot_dir}/{dataset}.svg', bbox_inches='tight')




def get_table_for_response_retrieval(model_names, metrics_history_all_t, t_10, t_50, dev_r1_idx=2):
    rows = []
    for name, values in zip(model_names, metrics_history_all_t):
        best_index = values[dev_r1_idx].argmax()  # sort by dev r@1
        row = [name, *values[:, best_index].tolist(), f'{best_index + 1}/{values.shape[1]}']
        rows.append(row)

    rows = sorted(rows, key=lambda x: x[dev_r1_idx + 1])  # sort by dev r@1
    t = Texttable(max_width=0)
    t.add_rows(rows, header=False)
    t.header(('Model', 'Trn Loss', 'Trn Acc',
              'Dev R@1', f'Dev R@{t_10}', f'Dev R@{t_50}', 'Dev MRR',
              'Tst R@1', f'Tst R@{t_10}', f'Tst R@{t_50}', 'Tst MRR',
              'Ep'))
    return t


class Config:
    def __init__(self, pad, unk, eps, inf, max_len, max_sent, max_seq):
        self.PAD = pad
        self.UNK = unk
        self.EPS = eps
        self.INF = inf
        self.max_len = max_len
        self.max_sent = max_sent
        self.MAX_SEQ = max_seq


def load_data_loaders(path, transformer_name, aug, batch_size, n_train, n_test, skip_test, shuffle_train=False, sequential=False):
    def load_data(fname, b_size, shuffle, n=None):
        with open(fname, 'rb') as f:
            context, response = pickle.load(f)

        dataset = TensorDataset(torch.tensor(context[:n]), torch.tensor(response[:n]))

        # dataloader = DataLoader(dataset, batch_size=b_size, shuffle=False)
        dataloader = DataLoader(dataset, batch_size=b_size, shuffle=shuffle)

        return dataloader, dataset

    train_data_loader, train_dataset = load_data(
        f"{path}/train-bert-bi{'-utterances' if sequential else ''}--ntransformer={transformer_name}--aug={aug}.bin",
        b_size=batch_size, shuffle=shuffle_train, n=n_train)
    valid_data_loader, valid_dataset = load_data(f"{path}/valid-bert-bi{'-utterances' if sequential else ''}--ntransformer={transformer_name}.bin",
                                                 b_size=batch_size * 2, shuffle=False, n=n_test)
    if not skip_test:
        test_data_loader, test_dataset = load_data(f"{path}/test-bert-bi{'-utterances' if sequential else ''}--ntransformer={transformer_name}.bin",
                                                   b_size=batch_size * 2, shuffle=False, n=n_test)
    else:
        test_data_loader = None

    return train_data_loader, valid_data_loader, test_data_loader




def get_bert_bi_hyperparameter_string(args, device_name):
    hyper_parameters_string = '--'.join(
        [k + '=' + str(v) for k, v in vars(args[0]).items() if k in
         ['dataset', 'batch', 'lr', 'aug', 'ntransformer']
         + ['lr_normal']
         + ['ntrain'] * (args[0].ntrain is not None)
         + ['seed'] * (args[0].seed != 1)
         + ['shuffle_train'] * int(args[0].shuffle_train)
         ]
        + ([f"gpu={device_name.replace(' ', '_').replace('-', '_')}"] if device_name != 'GeForce GTX 1080 Ti' else [])
    )
    return hyper_parameters_string

def get_student_hyperparameter_string(DEVICE_NAME, TEACHER_DEVICE, args):
    return '--'.join(
        [k + '=' + str(v) for k, v in vars(args).items() if k in
         [
             'nstudentmodel', 'student_lr', 'alpha', 'load'
         ] +
         (['student_batch'] if args.batch != args.student_batch else []) +
         (['use_weights'] if args.use_weights else []) + (['use_cached_logits'] if args.use_cached_logits else [])]
        + ([f"gpu={DEVICE_NAME.replace(' ', '_').replace('-', '_')}"] if DEVICE_NAME != 'GeForce GTX 1080 Ti' else [])
        + (
            [f"tgpu={(DEVICE_NAME if TEACHER_DEVICE is None else TEACHER_DEVICE).replace(' ', '_').replace('-', '_')}"]
            if DEVICE_NAME != 'GeForce GTX 1080 Ti' else [])
    )