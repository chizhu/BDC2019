from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import numpy as np
import random as rn
from tqdm import tqdm, tqdm_notebook
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras import backend as K
from keras.optimizers import *
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.initializers import *
import keras
from sklearn.model_selection import StratifiedKFold, GroupKFold
import gc
import time
from gensim.models import Word2Vec
import logging
import Levenshtein
tqdm.pandas()
np.random.seed(1017)
rn.seed(1017)
tf.set_random_seed(1017)
path = "/home/kesci/input/bytedance/"
out = '/home/kesci/work/chizhu/'
print(os.listdir(path))

train = pd.read_csv(path+"train_final.csv",skiprows=900000000,nrows=100000000,names=['query_id','query','query_title_id','title','label'])
test = pd.read_csv(path+"test_final_part1.csv",names=['query_id','query','query_title_id','title'])

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
# feature['词语求组合后标题'] = data_all.groupby('query').title.transform('nunique')
feature.to_csv(out + 'f3.csv', index=False)

# data_all = data_all.fillna(-1)
# data_all.to_csv(out+"data.csv", index=False)

# data_all = pd.read_csv(out+"data.csv")

# f5 word2vec本身相似度
from gensim.models import Word2Vec
import gensim
import logging
feature = pd.DataFrame()
w2v = Word2Vec.load(out + 'w2v.model')
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

f1 = pd.read_csv(out + 'f1.csv')
f2 = pd.read_csv(out + 'f2.csv')
f3 = pd.read_csv(out + 'f3.csv')
feature = pd.concat([f1, f2, f3], sort=False, axis=1)
del f1, f2, f3
gc.collect()

train = data_all[data_all['label'] != -1]
test = data_all[data_all['label'] == -1]
del data_all
gc.collect()
train_feature = feature[:len(train)]
test_feature = feature[len(train):]
train.index = range(len(train))
test.index = range(len(test))
train_feature.index = range(len(train_feature))
test_feature.index = range(len(test_feature))
del feature
gc.collect()

embed_size = 300  # how big is each word vector
# how many unique words to use (i.e num rows in embedding vector)
max_features = None
maxlen1 = 8
maxlen2 = 20  # max number of words in a question to use

train_X1 = train["query"].fillna("0").values
test_X1 = test["query"].fillna("0").values

train_X2 = train["title"].fillna("0").values
test_X2 = test["title"].fillna("0").values
print("token...")
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X1)+list(test_X1) +
                       list(train_X2)+list(test_X2))
train_X1 = tokenizer.texts_to_sequences(train_X1)
test_X1 = tokenizer.texts_to_sequences(test_X1)
## Pad the sentences
print("padding")
train_X1 = pad_sequences(train_X1, maxlen=maxlen1)
test_X1 = pad_sequences(test_X1, maxlen=maxlen1)

train_X2 = tokenizer.texts_to_sequences(train_X2)
test_X2 = tokenizer.texts_to_sequences(test_X2)
## Pad the sentences
train_X2 = pad_sequences(train_X2, maxlen=maxlen2)
test_X2 = pad_sequences(test_X2, maxlen=maxlen2)
## Get the target values

train_y = train['label'].values

word_index = tokenizer.word_index
gc.collect()

text_list = train['query'].values.tolist()
text_list.extend(test['query'].values.tolist())
text_list.extend(train['title'].values.tolist())
text_list.extend(test['title'].values.tolist())
del train,test
gc.collect()
import time
time.sleep(10)
text_list = [[word for word in str(document).split(' ') ] for document in text_list]
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
w2v = Word2Vec(text_list, size=300, window=7, iter=30, seed=10, workers=4, min_count=3)
w2v.save(out+"w2v.model")
w2v.wv.save_word2vec_format(out+'new_w2v_300.txt')
print("w2v model done")
del w2v, text_list, texts
gc.collect()


def get_embedding_matrix(word_index, embed_size=embed_size, Emed_path=out+"new_w2v_300.txt"):
    embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(
        Emed_path, binary=False)
    nb_words = len(word_index)+1
    embedding_matrix = np.zeros((nb_words, embed_size))
    count = 0
    for word, i in tqdm(word_index.items()):
        if i >= nb_words:
            continue
        try:
            embedding_vector = embeddings_index[word]
        except:
            embedding_vector = np.zeros(embed_size)
            count += 1
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print("null cnt", count)
    return embedding_matrix


embedding_matrix = get_embedding_matrix(word_index)


class AdamW(Optimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4,  # decoupled weight decay (1/4)
                 epsilon=1e-8, decay=0., **kwargs):
        super(AdamW, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            # decoupled weight decay (2/4)
            self.wd = K.variable(weight_decay, name='weight_decay')
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        wd = self.wd  # decoupled weight decay (3/4)

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            # decoupled weight decay (4/4)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon) - lr * wd * p

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'weight_decay': float(K.get_value(self.wd)),
                  'epsilon': self.epsilon}
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

# AUC for a binary classifier
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P


val = train[99000000:]
train = train[:99000000]
val_X1 = train_X1[99000000:]
val_X2 = train_X2[99000000:]
train_X1 = train_X1[:99000000]
train_X2 = train_X2[:99000000]
val_feature = train_feature[99000000:]
train_feature = train_feature[:99000000]

class ManDist(keras.layers.Layer):  # 封装成keras层的曼哈顿距离计算

    # 初始化ManDist层，此时不需要任何参数输入
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    # 自动建立ManDist层
    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    # 计算曼哈顿距离
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    # 返回结果
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


sc = StandardScaler()
col_len = len(train_feature.columns)
sc.fit(pd.concat([train_feature, val_feature, test_feature]))
train_feature = sc.transform(train_feature)
val_feature = sc.transform(val_feature)
test_feature = sc.transform(test_feature)

def get_model(embedding_matrix):

    K.clear_session()
    #The embedding layer containing the word vectors
    emb_layer = Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        trainable=False
    )
    sdrop=SpatialDropout1D(rate=0.2)
    lstm_layer = Bidirectional(CuDNNLSTM(64, return_sequences=True, 
kernel_initializer=glorot_uniform(seed = 123)))
    gru_layer = Bidirectional(CuDNNGRU(64, return_sequences=True, 
kernel_initializer=glorot_uniform(seed = 123)))
    
    cnn1d_layer=keras.layers.Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")

    # Define inputs
    seq1 = Input(shape=(maxlen1,))
    x1 = emb_layer(seq1)
    x1 = sdrop(x1)
    lstm1 = lstm_layer(x1)
    gru1 = gru_layer(lstm1)
    att_1 = Attention(maxlen1)(lstm1)
    att_3 = Attention(maxlen1)(gru1)
    cnn1 = cnn1d_layer(lstm1)
    
    avg_pool = GlobalAveragePooling1D()
    max_pool = GlobalMaxPooling1D()
    
    seq2 = Input(shape=(maxlen2,))
    x2 = emb_layer(seq2)
    x2 = sdrop(x2)
    lstm2 = lstm_layer(x2)
    gru2 = gru_layer(lstm2)
    att_2 = Attention(maxlen2)(lstm2)
    att_4 = Attention(maxlen2)(gru2)
    cnn2 = cnn1d_layer(lstm2)
    
    x1=concatenate([att_1,att_3,avg_pool(cnn1),max_pool(cnn1),avg_pool(gru1),max_pool(gru1)])
    x2=concatenate([att_2,att_4,avg_pool(cnn2),max_pool(cnn2),avg_pool(gru2),max_pool(gru2)])
    
    merge = Multiply()([x1, x2])
    merge = Dropout(0.2)(merge)
    
    hin = Input(shape=(col_len,))
    # htime = Dense(col_len,activation='relu')(hin)
    x = Concatenate()([merge,hin])
    # The MLP that determines the outcome
    x = Dense(64,kernel_initializer=he_uniform(seed=123), activation='relu',)(x)
    # x = Dropout(0.2)(x)
    # x = BatchNormalization()(x)

    pred = Dense(1,kernel_initializer=he_uniform(seed=123), activation='sigmoid')(x)

    
    model = Model(inputs=[seq1,seq2,hin], outputs=pred)

    model.compile(loss='binary_crossentropy',
                  optimizer=AdamW(lr=0.001,weight_decay=0.02,),
                  metrics=["accuracy",auc])
    # model.summary()
    return model


####模型训练

print("train...")
print("###"*30)
gc.collect()
K.clear_session()
model = get_model(embedding_matrix)
# model = esim()
model.summary()
early_stopping = EarlyStopping(
    monitor='val_loss', min_delta=0.0001, patience=2, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=1, min_lr=0.0001, verbose=2)
bst_model_path = out+'chizhurnn_chizhu_weight.h5'
checkpoint = ModelCheckpoint(bst_model_path, monitor='val_loss', mode='min',
                             save_best_only=True, verbose=1, save_weights_only=True)
callbacks = [checkpoint, reduce_lr, early_stopping]
print("load weight....")
# model.load_weights(bst_model_path)

hist = model.fit([train_X1,train_X2,train_feature],train['label'],
                    validation_data=([val_X1,val_X2,val_feature], val['label']),
                     epochs=30, batch_size=2048,
#                      class_weight="auto",
                     callbacks=callbacks,verbose=1

                     )

model.load_weights(bst_model_path)

res = np.squeeze(model.predict(
    [val_X1, val_X2, val_feature], batch_size=2048, verbose=1))

print("val auc:{}".format(roc_auc_score(val['label'], res)))
val['prob'] = res


def perauc(df):
    temp = pd.DataFrame(index=range(1))
    temp['query_id'] = df['query_id'].values[0]
    try:
        temp['auc'] = roc_auc_score(df['label'].values.astype(int), df['prob'])
    except:
        temp['auc'] = 0.5
    return temp


eval_df = val.groupby("query_id", as_index=False).apply(lambda x: perauc(x))
eval_df.index = range(len(eval_df))
print("qauc:", eval_df['auc'].mean())

test_prob = np.squeeze(model.predict(
    [test_X1, test_X2, test_feature], batch_size=2048, verbose=1))


sub = test[['query_id', 'query_title_id']]
sub['prediction'] = test_prob
sub.to_csv(out+"/submit_rnn.csv", index=False, header=False)


