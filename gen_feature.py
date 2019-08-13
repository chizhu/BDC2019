from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import StratifiedKFold, GroupKFold
import numpy as np
import os
import Levenshtein
import logging
from gensim.models import Word2Vec
import time
import gc
import keras
from keras.initializers import *
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.engine.topology import Layer
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import *
from keras import backend as K
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import random as rn
import pandas as pd
tqdm.pandas()
np.random.seed(1017)
rn.seed(1017)
tf.set_random_seed(1017)
path = "/home/kesci/input/bytedance/"
out = '/home/kesci/work/chizhu/'
print(os.listdir(path))

train = pd.read_csv(path+"train_final.csv",skiprows=900000000,nrows=100000000,names=['query_id','query','query_title_id','title','label'])

testa = pd.read_csv(path+"test_final_part1.csv",names=['query_id','query','query_title_id','title'])
testb = pd.read_csv(path+"bytedance_contest.final_2.csv",names=['query_id','query','query_title_id','title'])

testa['label']=-1
testb['label']=-2
test=pd.concat([testa,testb],ignore_index=True)
del testa,testb
gc.collect()

train['title']=train['title'].apply(lambda x:str(x).replace("\t",""),1)
test['title']=test['title'].apply(lambda x:str(x).replace("\t",""),1)
data_all=pd.concat([train,test],ignore_index=True)
del train,test
gc.collect()

# 构造特征集 f1
def get_union_data(row):
    title_list = row['title'].split(' ')
    query_list = row['query'].split(' ')
    return len(list(set(title_list).intersection(set(query_list))))

def same_1(row):
    title_list = row['title'].split(' ')
    query_list = row['query'].split(' ')
    if title_list[0] ==  query_list[0]:
        return 1
    else:
        return 0

def same_2(row):
    title_list = row['title'].split(' ')
    query_list = row['query'].split(' ')
    if ' '.join(title_list[:2]) ==  ' '.join(query_list[:2]):
        return 1
    else:
        return 0

def same_3(row):
    title_list = row['title'].split(' ')
    query_list = row['query'].split(' ')
    if ' '.join(title_list[:3]) ==  ' '.join(query_list[:3]):
        return 1
    else:
        return 0

def is_all_in(row):
    if row['query'] in row['title']:
        return 1
    else:
        return 0

feature = pd.DataFrame()
feature['问题长度'] = data_all['query'].progress_apply(lambda row:len(row.split(' ')))
feature['标题长度'] = data_all['title'].progress_apply(lambda row:len(row.split(' ')))
feature['标题长度-问题长度'] = feature['标题长度'] - feature['问题长度']
feature['问题是否全部在标题里面'] = data_all.progress_apply(lambda row:is_all_in(row), axis=1)
feature['标题和问题的交集个数'] = data_all.progress_apply(lambda row:get_union_data(row), axis=1)
feature['标题问题词语的交集个数/问题长度'] = np.around(np.divide(feature['标题和问题的交集个数'], feature['问题长度']), 8)
feature['标题问题词语的交集个数/标题长度'] = np.around(np.divide(feature['标题和问题的交集个数'], feature['标题长度']), 8)
feature['编辑距离'] = data_all.progress_apply(lambda row:Levenshtein.distance(row['query'], row['title']), axis=1)
feature['前一个词语是否相同'] = data_all.progress_apply(lambda row:same_1(row), axis=1)
feature['前两个词语是否相同'] = data_all.progress_apply(lambda row:same_2(row), axis=1)
feature['前三个词语是否相同'] = data_all.progress_apply(lambda row:same_3(row), axis=1)
feature.to_csv(out + 'f1.csv', index=False)

# 构造特征集 f2
def pos_1(row):
    title_list = row['title'].split(' ')
    query_list = row['query'].split(' ')
    value = -1
    try:
        value = title_list.index(query_list[0])
    except Exception:
        value = -1
    return value

def pos_2(row):
    title_list = row['title'].split(' ')
    query_list = row['query'].split(' ')
    if len(query_list) <=1 :
        return -1
    try:
        value = title_list.index(query_list[1])
    except Exception:
        value = -1
    return value

def pos_3(row):
    title_list = row['title'].split(' ')
    query_list = row['query'].split(' ')
    if len(query_list) <=2 :
        return -1
    try:
        value = title_list.index(query_list[2])
    except Exception:
        value = -1
    return value

feature = pd.DataFrame()
feature['第一个词语在标题里面出现位置'] = data_all.progress_apply(lambda row:pos_1(row), axis=1)
feature['第二个词语在标题里面出现位置'] = data_all.progress_apply(lambda row:pos_2(row), axis=1)
feature['第三个词语在标题里面出现位置'] = data_all.progress_apply(lambda row:pos_3(row), axis=1)
feature.to_csv(out + 'f2.csv', index=False)

feature = pd.DataFrame()
feature['标题求组合后词语'] = data_all.groupby('title').query.transform('nunique')
feature['词语求组合后标题'] = data_all.groupby('query').title.transform('nunique')
feature.to_csv(out + 'f3.csv', index=False)


# data_all = data_all.fillna(-1)
# data_all.to_pickle(out+"data.pickle")

# data_all = pd.read_pickle(out+"data.pickle")
# f5 word2vec本身相似度
from gensim.models import Word2Vec
import gensim
import logging
feature = pd.DataFrame()
w2v = Word2Vec.load(out + 'new_skip_w2v_all_300.model')
def get_new_w2v(seq1, seq2):
    seq1 = seq1.split(' ')
    seq2 = seq2.split(' ')
    try:
        return w2v.n_similarity(seq1, seq2)
    except:
        return -1

f3 = pd.read_csv(out + 'f3.csv')
f3['w2v本身相似度'] = data_all.progress_apply(lambda row:get_new_w2v(row['query'], row['title']), axis=1)
f3.to_csv(out + 'f3.csv', index=False)


