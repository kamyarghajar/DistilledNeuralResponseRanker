# Distilled Neural Response Ranker 

Implementation of "Distilling Knowledge for Fast Retrieval-based Chat-bots" (SIGIR 2020) using deep matching transformer networks and knowledge distillation for response retrieval in information-seeking conversational systems.

Prepocessing a dataset:
```bash
cd ./Datasets/ubuntu_data/
ipython ./ubuntu_preprocess.py
```

Training a Bi-Encoder:
```bash
ipython ./main_bert_bi.py -- --dataset ubuntu_data --nmodel BiEncoderDot2 --epochs 1 --ntrain 100000
```

Training an enchanced Cross-Encoder (BECA):
```bash
ipython ./main_bert_bi.py -- --dataset ubuntu_data --nmodel MolyEncoderAggP2 --epochs 1 --ntrain 100000
```


You can cite the paper as:

```
Amir Vakili Tahami, Kamyar Ghajar, Azadeh Shakery. Distilling Knowledge for Fast Retrieval-based Chat-bots. In Proceedings of the
43th International ACM SIGIR Conference on Research & Development in Information Retrieval (SIGIR 2020).

Bibtext
 @inproceedings{tahami2020distilling,
    title={Distilling Knowledge for Fast Retrieval-based Chat-bots},
    author={Amir Vakili Tahami and Kamyar Ghajar and Azadeh Shakery},
    year={2020},
    eprint={2004.11045},
    archivePrefix={arXiv},
    booktitle = {SIGIR '20}
}
```

