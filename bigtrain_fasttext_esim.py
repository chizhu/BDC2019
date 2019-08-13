from keras.activations import softmax
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
import fasttext
tqdm.pandas()
np.random.seed(1017)
rn.seed(1017)
tf.set_random_seed(1017)
path = "/home/kesci/input/bytedance/"
out = '/home/kesci/work/zhifeng/'
print(os.listdir(path))

w2v = fasttext.load_model(out+'corpus.fasttext.model')
word2index = {word: index+1 for index, word in enumerate(w2v.words)}
index2word = {index+1: word for index, word in enumerate(w2v.words)}
def gen_feature_help(line, label_tag=True, token=word2index, maxlen_answer=20,
                     maxlen_query=8):
    if label_tag:
        _, _q, _, _a, _label = line.strip().split(',')
    else:
        _, _q, _, _a = line.strip().split(',')
    q_seq = [token.get(item, 0) for item in _q.strip().split()]
    a_seq = [token.get(item, 0) for item in _a.strip().split()]
    q_pad = [0]*(maxlen_query - len(q_seq)) + q_seq[-maxlen_query:]
    a_pad = [0]*(maxlen_answer - len(a_seq)) + a_seq[-maxlen_answer:]
    if label_tag:
        return q_pad, a_pad, int(_label)
    return q_pad, a_pad


def gen_train(path, batch_size=256, label_tag=True, chunk_size=1000, shuffle=True, maxlen_answer=20, maxlen_query=8):
    while True:
        fin = open(path, 'r')
        batch_q, batch_a, batch_label = [], [], []
        for line in fin:
            if len(batch_q) == chunk_size*batch_size:
                batch_q = np.array(batch_q)
                batch_a = np.array(batch_a)
                if label_tag:
                    batch_label = np.array(batch_label)
                idx = list(range(chunk_size*batch_size))
                if shuffle:
                    np.random.shuffle(idx)
                for i in range(chunk_size):
                    if label_tag:
                        yield [np.array(batch_q[idx[i*batch_size:i*batch_size+batch_size]]), np.array(batch_a[idx[i*batch_size:i*batch_size+batch_size]])], np.array(batch_label[idx[i*batch_size:i*batch_size+batch_size]])
                    else:
                        yield [np.array(batch_q[idx[i*batch_size:i*batch_size+batch_size]]), np.array(batch_a[idx[i*batch_size:i*batch_size+batch_size]])]
                batch_q, batch_a, batch_label = [], [], []
            if label_tag:
                q, a, l = gen_feature_help(line, label_tag=label_tag)
            else:
                q, a = gen_feature_help(line, label_tag=label_tag)
                l = 0
            batch_q.append(q)
            batch_a.append(a)
            if label_tag:
                batch_label.append(l)

        batch_q = np.array(batch_q)
        batch_a = np.array(batch_a)

        if label_tag:
            batch_label = np.array(batch_label)
        idx = list(range(len(batch_q)))
        if shuffle:
            np.random.shuffle(idx)
        for i in range(int(np.ceil(len(batch_q)/batch_size))):
            if label_tag:
                yield [np.array(batch_q[idx[i*batch_size:i*batch_size+batch_size]]), np.array(batch_a[idx[i*batch_size:i*batch_size+batch_size]])], np.array(batch_label[idx[i*batch_size:i*batch_size+batch_size]])
            else:
                yield [np.array(batch_q[idx[i*batch_size:i*batch_size+batch_size]]), np.array(batch_a[idx[i*batch_size:i*batch_size+batch_size]])]
        fin.close()


def get_embedding_matrix():
    m = np.zeros(shape=(len(index2word)+1, 100))
    for i, w in index2word.items():
        m[i, :] = w2v[w]
    return m


embed_matrix = get_embedding_matrix()
maxlen_query = 8
maxlen_answer = 20


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
    ptas = tf.stack([binary_PTA(y_true, y_pred, k)
                     for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.stack([binary_PFA(y_true, y_pred, k)
                     for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
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


def create_pretrained_embedding(pretrained_weights, trainable=False, **kwargs):
    "Create embedding layer from a pretrained weights array"
    in_dim, out_dim = pretrained_weights.shape
    embedding = Embedding(in_dim, out_dim, weights=[
                          pretrained_weights], trainable=False, **kwargs)
    return embedding


def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape


def substract(input_1, input_2):
    "Substract element-wise"
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_ = Concatenate()([sub, mult])
    return out_


def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_


def time_distributed(input_, layers):
    "Apply a list of layers in TimeDistributed mode"
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_)
    out_ = node_
    return out_


def soft_attention_alignment(input_1, input_2):
    "Align text representation with neural soft attention"
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                                     output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


def decomposable_attention(pretrained_weights,
                           num_shape,
                           projection_dim=300, projection_hidden=0, projection_dropout=0.2,
                           compare_dim=500, compare_dropout=0.2,
                           dense_dim=300, dense_dropout=0.2,
                           lr=1e-3, activation='elu', maxlen=20):
    # Based on: https://arxiv.org/abs/1606.01933

    q1 = Input(name='q1', shape=(maxlen,))
    q2 = Input(name='q2', shape=(maxlen,))

    # Embedding
    embedding = create_pretrained_embedding(pretrained_weights,
                                            mask_zero=False)
    q1_embed = embedding(q1)
    q2_embed = embedding(q2)

    # Projection
    projection_layers = []
    if projection_hidden > 0:
        projection_layers.extend([
            Dense(projection_hidden, activation=activation),
            Dropout(rate=projection_dropout),
        ])
    projection_layers.extend([
        Dense(projection_dim, activation=None),
        Dropout(rate=projection_dropout),
    ])
    q1_encoded = time_distributed(q1_embed, projection_layers)
    q2_encoded = time_distributed(q2_embed, projection_layers)

    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

    # Compare
    q1_combined = Concatenate()(
        [q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()(
        [q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])
    compare_layers = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
    ]
    q1_compare = time_distributed(q1_combined, compare_layers)
    q2_compare = time_distributed(q2_combined, compare_layers)

    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

    # Classifier
    merged = Concatenate()([q1_rep, q2_rep])
    dense = BatchNormalization()(merged)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = BatchNormalization()(dense)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[q1, q2], outputs=out_)
    model.compile(loss='binary_crossentropy',
                  optimizer=AdamW(lr=0.001, weight_decay=0.02,),
                  metrics=["accuracy", auc])
    return model


def esim(embedding_matrix,
         maxlen=20,
         lstm_dim=30,
         dense_dim=30,
         dense_dropout=0.5):
    # Based on arXiv:1609.06038
    q1 = Input(name='q1', shape=(8,))
    q2 = Input(name='q2', shape=(20,))

    # Embedding
    embedding = create_pretrained_embedding(
        embedding_matrix, mask_zero=False)
    bn = BatchNormalization(axis=2)
    q1_embed = bn(embedding(q1))
    q2_embed = bn(embedding(q2))

    # Encode
    encode = Bidirectional(CuDNNLSTM(lstm_dim, return_sequences=True))
    q1_encoded = encode(q1_embed)
    q2_encoded = encode(q2_embed)

    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

    # Compose
    q1_combined = Concatenate()(
        [q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()(
        [q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])

    compose = Bidirectional(CuDNNLSTM(lstm_dim, return_sequences=True))
    q1_compare = compose(q1_combined)
    q2_compare = compose(q2_combined)

    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

    # leaks_input = Input(shape=(num_shape,))
    # leaks_dense = Dense(dense_dim//2, activation='relu')(leaks_input)

    # Classifier
    merged = Concatenate()([q1_rep, q2_rep])

    dense = BatchNormalization()(merged)
    dense = Dense(dense_dim, activation='elu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = Dense(dense_dim, activation='elu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[q1, q2], outputs=out_)
    model.compile(loss='binary_crossentropy',
                  optimizer=AdamW(lr=0.0003, weight_decay=0.02,),
                  metrics=["accuracy"])
    return model


####模型训练
train_gen = gen_train(path='/home/kesci/zhifeng/train.csv',
                      batch_size=4096, label_tag=True, chunk_size=1000)
val_gen = gen_train(path='/home/kesci/zhifeng/val.csv',
                    batch_size=4096, label_tag=True, chunk_size=1000)
print("train...")
print("###"*30)
gc.collect()
K.clear_session()
model = esim(embed_matrix)
model.summary()
early_stopping = EarlyStopping(
    monitor='val_loss', min_delta=0.0001, patience=2, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=1, min_lr=0.0001, verbose=2)
bst_model_path = '/home/kesci/chizhu/chizhu_w2v_esim_weight_{epoch}_{val_loss}.h5'
checkpoint = ModelCheckpoint(bst_model_path, monitor='val_loss', mode='min',
                             save_best_only=False,
                             verbose=1, save_weights_only=True, period=1)
callbacks = [checkpoint, reduce_lr, early_stopping]
# print("load weight....")


hist = model.fit_generator(train_gen, steps_per_epoch=int(np.ceil(999000000/2048)),
                           epochs=10, verbose=1, callbacks=callbacks,
                           validation_data=val_gen, validation_steps=int(
                               np.ceil(1000000/2048)),
                           max_queue_size=10, workers=1, use_multiprocessing=False)

val_gen = gen_train(path='/home/kesci/zhifeng/val.csv',
                    batch_size=4096, label_tag=True, chunk_size=1000, shuffle=False)
val_prob = model.predict_generator(
    val_gen, steps=int(np.ceil(1000000/4096)), verbose=1)

f = open('/home/kesci/zhifeng/val.csv', 'r')
q, a, l = [], [], []
for line in f:
    qid, _, aid, _, label = line.strip().split(',')
    q.append(qid)
    a.append(aid)
    l.append(int(label))

val_df = pd.DataFrame({'qid': q, 'aid': a, 'label': l})
val_df['prob'] = val_prob.flatten()

roc_auc_score(val_df['label'], val_df['prob'])


def perauc(df):
    temp = pd.Series()
    try:
        temp['auc'] = roc_auc_score(df['label'], df['prob'])
    except:
        temp['auc'] = 0.5
    return temp


eval_df = val_df.groupby("qid").apply(perauc)
eval_df.index = range(len(eval_df))
print("qauc:", eval_df['auc'].mean())

test_gen = gen_train(path='/home/kesci/input/bytedance/test_final_part1.csv',
                     batch_size=4096, label_tag=False, chunk_size=1, shuffle=False)
prob = model.predict_generator(
    test_gen, steps=int(np.ceil(20000000/4096)), verbose=1)
sub = pd.read_csv('/home/kesci/work/chizhu/submit_rnn.csv',
                  names=['qid', 'aid', 'prob'])
sub['prob'] = prob.flatten()
sub.to_csv('/home/kesci/work/chizhu/raw_w2v_esim_testa.csv',
           index=False, header=False)

test_gen = gen_train(path='/home/kesci/input/bytedance/bytedance_contest.final_2.csv',
                     batch_size=4096, label_tag=False, chunk_size=1, shuffle=False)
prob = model.predict_generator(
    test_gen, steps=int(np.ceil(100000000/4096)), verbose=1)
final = pd.read_csv(path+"bytedance_contest.final_2.csv", names=[
                    'query_id', 'query', 'query_title_id', 'title'])[['query_id', 'query_title_id']]
final['prob'] = prob.flatten()
final.to_csv('/home/kesci/work/chizhu/raw_w2v_esim_testb.csv',
             index=False, header=False)
