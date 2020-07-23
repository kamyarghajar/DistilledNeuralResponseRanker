# Distilling Knowledge for Fast Retrieval-based Chat-bots

Implementation of "Distilling Knowledge for Fast Retrieval-based Chat-bots" (SIGIR 2020) using deep matching transformer networks and knowledge distillation for response retrieval in information-seeking conversational systems.

Prepocessing a dataset:
```bash
cd ./Datasets/ubuntu_data/
ipython ./ubuntu_preprocess.py
```
The ubuntu dataset can be found [here](https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip?dl=0&file_subpath=%2Fubuntu_data).


Training a Bi-Encoder using cross entropy loss:
```bash
ipython ./main_bert_bi.py -- --dataset ubuntu_data --batch 16 --nmodel BiEncoderDot2 --epochs 1 
```

Training an enchanced Cross-Encoder (BECA) using cross entropy loss:
```bash
ipython ./main_bert_bi.py -- --dataset ubuntu_data --batch 8 --nmodel MolyEncoderAggP2 --epochs 1
```

Training a Bi-Encoder using cross-entropy loss and knowledge distillation:
```bash
 ipython ./main_bi_kd.py -- --dataset ubuntu_data --batch 8 --student_batch 16 --alpha 0.5  --epochs 1  --load 1 --nmodel MolyEncoderAggP2 --nstudentmodel BiEncoderDot2
```
This uses the outputs of a previously trained enhanced cross encoder (BECA) when training the Bi-Encoder.
Be careful to match the hyper-parameters between the two runs otherwise you'll get file not found errors. 


You can cite the paper as:

```
Amir Vakili Tahami, Kamyar Ghajar, Azadeh Shakery. Distilling Knowledge for Fast Retrieval-based Chat-bots. In Proceedings of the
43th International ACM SIGIR Conference on Research & Development in Information Retrieval (SIGIR 2020).

Bibtext
 @article{tahami2020distilling,
    title={Distilling Knowledge for Fast Retrieval-based Chat-bots},
    author={Amir Vakili Tahami and Kamyar Ghajar and Azadeh Shakery},
    year={2020},
    eprint={2004.11045},
    archivePrefix={arXiv},
}
```

