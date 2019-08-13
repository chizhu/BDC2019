import os
import pandas as pd
import numpy as np
import random as rn
from tqdm import tqdm, tqdm_notebook
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
import gc
import time
from gensim.models import Word2Vec
import fasttext
from gensim.models import Word2Vec
import scipy.spatial.distance as ssd
tqdm.pandas()
input_path = "/home/kesci/input/bytedance/"
out_work_path = '/home/kesci/work/zhifeng/'
out_path = '/home/kesci/zhifeng/'

w2v = Word2Vec.load('/home/kesci/work/chizhu/skip_w2v_all_300.model')


def get_sentence_embeddings(text, sep=' ', dim=300):
    v = np.zeros(dim)
    words = text.strip().split(sep)
    cnt = 0
    for word in words:
        if word in w2v:
            v += w2v[word]
            cnt += 1
    return v/cnt if cnt != 0 else v


train_cosine_list = []
with open(out_path+'train.smaller.csv', 'r') as fin:
    for line in tqdm(fin):
        _, q, _, a, _ = line.strip().split(',')
        v1 = get_sentence_embeddings(q)
        v2 = get_sentence_embeddings(a)
        train_cosine_list.append(ssd.cosine(v1, v2))
pd.to_pickle(np.array(train_cosine_list), out_work_path+'train.cosine.w2v.pkl')
val_cosine_list = []
with open(out_path+'val.csv', 'r') as fin:
    for line in tqdm(fin):
        _, q, _, a, _ = line.strip().split(',')
        v1 = get_sentence_embeddings(q)
        v2 = get_sentence_embeddings(a)
        val_cosine_list.append(ssd.cosine(v1, v2))
pd.to_pickle(np.array(val_cosine_list), out_work_path+'val.cosine.w2v.pkl')
test_cosine_list = []
with open(input_path+'test_final_part1.csv', 'r') as fin:
    for line in tqdm(fin):
        _, q, _, a = line.strip().split(',')
        v1 = get_sentence_embeddings(q)
        v2 = get_sentence_embeddings(a)
        test_cosine_list.append(ssd.cosine(v1, v2))
pd.to_pickle(np.array(test_cosine_list), out_path+'test.cosine.w2v.pkl')
