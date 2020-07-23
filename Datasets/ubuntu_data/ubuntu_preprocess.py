import os
import ubuntu_uf
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
import pickle


# %%

def preprocess(name, is_train):
    with open('./' + name + '.txt', 'r', encoding='utf-8-sig') as f:
        s = f.readlines()

    n_samp = 2 if is_train else 10

    contexts = [' [EOU] '.join(x.split('\t')[1:-1])
                    .replace('__number__', '[NUM]')
                    .replace('__path__', '[PATH]')
                    .replace('__url__', '[URL]')
                    .strip()
                for i, x in enumerate(s) if i % n_samp == 0]

    responses = [(x.split('\t')[-1])
                     .replace('__number__', '[NUM]')
                     .replace('__path__', '[PATH]')
                     .replace('__url__', '[URL]')
                     .strip()
                 for x in s]

    with Pool(os.cpu_count() // 2) as p:
        token_ids_context = p.map(ubuntu_uf.process_context, tqdm(contexts))

    with Pool(os.cpu_count() // 2) as p:
        token_ids_response = p.map(ubuntu_uf.process_response, tqdm(responses))

    token_ids_matrix_context = np.array(token_ids_context).astype('int64')
    if is_train:
        token_ids_matrix_response = np.array(token_ids_response).astype('int64')[::2]
    else:
        token_ids_matrix_response = np.array(token_ids_response).astype('int64').reshape(len(token_ids_response) // 10,
                                                                                         10, -1)

    with open(f"{name}-bert-bi--ntransformer=distilbert{'--aug=False' if is_train else ''}.bin",
              'wb') as f:
        pickle.dump((token_ids_matrix_context, token_ids_matrix_response), f)


# %%
if __name__ == '__main__':
    preprocess('train', True)

    preprocess('valid', False)

    preprocess('test', False)
