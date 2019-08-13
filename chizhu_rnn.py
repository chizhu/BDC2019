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

f1 = pd.read_csv(out + 'f1.csv')
f2 = pd.read_csv(out + 'f2.csv')
f3 = pd.read_csv(out + 'f3.csv')
feature = pd.concat([f1, f2, f3], sort=False, axis=1)
del f1, f2, f3
gc.collect()

train_w2v = pd.read_pickle("/home/kesci/work/zhifeng/train.cosine.w2v.pkl")
val_w2v = pd.read_pickle("/home/kesci/work/zhifeng/val.cosine.w2v.pkl")
testa_w2v = pd.read_pickle("/home/kesci/work/zhifeng/test.cosine.w2v.pkl")
testb_w2v = pd.read_pickle(
    "/home/kesci/work/zhifeng/test_final.cosine.w2v.pkl")
feature['w2v_cos'] = list(train_w2v)+list(testa_w2v)+list(testb_w2v)

train_w2v = pd.read_pickle(
    "/home/kesci/work/zhifeng/train.cosine.fasttext.pkl")
val_w2v = pd.read_pickle("/home/kesci/work/zhifeng/val.cosine.fasttext.pkl")
testa_w2v = pd.read_pickle("/home/kesci/work/zhifeng/test.cosine.fasttext.pkl")
testb_w2v = pd.read_pickle(
    "/home/kesci/work/zhifeng/test_final.cosine.fasttext.pkl")
feature['fast_cos'] = list(train_w2v)+list(val_w2v) + \
    list(testa_w2v)+list(testb_w2v)
del train_w2v, val_w2v, testa_w2v, testb_w2v
gc.collect()
feature.shape

len_train = 99000000
len_val = 1000000
len_testa = 20000000
len_testb = 100000000
sc = StandardScaler()
feature = sc.fit_transform(feature)
train_feature = feature[:len_train]
val_feature = feature[len_train:len_train+len_val]
testa_feature = feature[len_train+len_val:len_train+len_val+len_testa]
testb_feature = feature[-len_testb:]
print(train_feature.shape, val_feature.shape,testa_feature.shape,testb_feature.shape)

del feature
gc.collect()

w2v = Word2Vec.load('/home/kesci/work/chizhu/new_skip_w2v_all_300.model')
word2index = {word: index+1 for index, word in enumerate(w2v.wv.index2entity)}
index2word = {index+1: word for index, word in enumerate(w2v.wv.index2entity)}


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


def gen_train(path, feature, batch_size=256, label_tag=True, chunk_size=1000, shuffle=True, maxlen_answer=20, maxlen_query=8):
    while True:
        fin = open(path, 'r')
        batch_q, batch_a, batch_f, batch_label = [], [], [], []
        for i, line in enumerate(fin):
            if len(batch_q) == chunk_size*batch_size:
                batch_q = np.array(batch_q)
                batch_a = np.array(batch_a)
                batch_f = np.array(batch_f)
                if label_tag:
                    batch_label = np.array(batch_label)
                idx = list(range(chunk_size*batch_size))
                if shuffle:
                    np.random.shuffle(idx)
                for i in range(chunk_size):
                    if label_tag:
                        yield ([np.array(batch_q[idx[i*batch_size:i*batch_size+batch_size]]),
                                np.array(
                                    batch_a[idx[i*batch_size:i*batch_size+batch_size]]),
                                np.array(batch_f[idx[i*batch_size:i*batch_size+batch_size]])],
                               np.array(batch_label[idx[i*batch_size:i*batch_size+batch_size]]))
                    else:
                        yield [np.array(batch_q[idx[i*batch_size:i*batch_size+batch_size]]),
                               np.array(
                                   batch_a[idx[i*batch_size:i*batch_size+batch_size]]),
                               np.array(batch_f[idx[i*batch_size:i*batch_size+batch_size]])]
                batch_q, batch_a, batch_f, batch_label = [], [], [], []
            if label_tag:
                q, a, l = gen_feature_help(line, label_tag=label_tag)
            else:
                q, a = gen_feature_help(line, label_tag=label_tag)
                l = 0
            batch_q.append(q)
            batch_a.append(a)
            batch_f.append(feature[i])
            if label_tag:
                batch_label.append(l)

        batch_q = np.array(batch_q)
        batch_a = np.array(batch_a)
        batch_f = np.array(batch_f)

        if label_tag:
            batch_label = np.array(batch_label)
        idx = list(range(len(batch_q)))
        if shuffle:
            np.random.shuffle(idx)
        for i in range(int(np.ceil(len(batch_q)/batch_size))):
            if label_tag:
                yield ([np.array(batch_q[idx[i*batch_size:i*batch_size+batch_size]]),
                        np.array(
                            batch_a[idx[i*batch_size:i*batch_size+batch_size]]),
                        np.array(batch_f[idx[i*batch_size:i*batch_size+batch_size]])],
                       np.array(batch_label[idx[i*batch_size:i*batch_size+batch_size]]))
            else:
                yield [np.array(batch_q[idx[i*batch_size:i*batch_size+batch_size]]),
                       np.array(
                           batch_a[idx[i*batch_size:i*batch_size+batch_size]]),
                       np.array(batch_f[idx[i*batch_size:i*batch_size+batch_size]])]
        fin.close()


def get_embedding_matrix():
    m = np.zeros(shape=(len(index2word)+1, 300))
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


class Lookahead(object):
    """Add the [Lookahead Optimizer](https://arxiv.org/abs/1907.08610) functionality for [keras](https://keras.io/).
    """

    def __init__(self, k=5, alpha=0.5):
        self.k = k
        self.alpha = alpha
        self.count = 0

    def inject(self, model):
        """Inject the Lookahead algorithm for the given model.
        The following code is modified from keras's _make_train_function method.
        See: https://github.com/keras-team/keras/blob/master/keras/engine/training.py#L497
        """
        if not hasattr(model, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')

        model._check_trainable_weights_consistency()

        if model.train_function is None:
            inputs = (model._feed_inputs +
                      model._feed_targets +
                      model._feed_sample_weights)
            if model._uses_dynamic_learning_phase():
                inputs += [K.learning_phase()]
            fast_params = model._collected_trainable_weights

            with K.name_scope('training'):
                with K.name_scope(model.optimizer.__class__.__name__):
                    training_updates = model.optimizer.get_updates(
                        params=fast_params,
                        loss=model.total_loss)
                    slow_params = [K.variable(p) for p in fast_params]
                fast_updates = (model.updates +
                                training_updates +
                                model.metrics_updates)

                slow_updates, copy_updates = [], []
                for p, q in zip(fast_params, slow_params):
                    slow_updates.append(K.update(q, q + self.alpha * (p - q)))
                    copy_updates.append(K.update(p, q))

                # Gets loss and metrics. Updates weights at each call.
                fast_train_function = K.function(
                    inputs,
                    [model.total_loss] + model.metrics_tensors,
                    updates=fast_updates,
                    name='fast_train_function',
                    **model._function_kwargs)

                def F(inputs):
                    self.count += 1
                    R = fast_train_function(inputs)
                    if self.count % self.k == 0:
                        K.batch_get_value(slow_updates)
                        K.batch_get_value(copy_updates)
                    return R

                model.train_function = F
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
    seq1 = Input(shape=(maxlen_query,))
    x1 = emb_layer(seq1)
    x1 = sdrop(x1)
    lstm1 = lstm_layer(x1)
    gru1 = gru_layer(lstm1)
    att_1 = Attention(maxlen_query)(lstm1)
    att_3 = Attention(maxlen_query)(gru1)
    cnn1 = cnn1d_layer(lstm1)
    
    avg_pool = GlobalAveragePooling1D()
    max_pool = GlobalMaxPooling1D()
    
    seq2 = Input(shape=(maxlen_answer,))
    x2 = emb_layer(seq2)
    x2 = sdrop(x2)
    lstm2 = lstm_layer(x2)
    gru2 = gru_layer(lstm2)
    att_2 = Attention(maxlen_answer)(lstm2)
    att_4 = Attention(maxlen_answer)(gru2)
    cnn2 = cnn1d_layer(lstm2)
    
    x1=concatenate([att_1,att_3,avg_pool(cnn1),max_pool(cnn1),avg_pool(gru1),max_pool(gru1)])
    x2=concatenate([att_2,att_4,avg_pool(cnn2),max_pool(cnn2),avg_pool(gru2),max_pool(gru2)])
    
    merge = Multiply()([x1, x2])
    merge = Dropout(0.2)(merge)
    
    hin = Input(shape=(19,))
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
train_gen = gen_train(path='/home/kesci/zhifeng/train.smaller.csv',feature=train_feature,batch_size=2048,
label_tag=True,chunk_size=5000)
val_gen = gen_train(path='/home/kesci/zhifeng/val.csv',feature=val_feature,batch_size=2048,
label_tag=True,chunk_size=5000)
print("train...")
print("###"*30)
gc.collect()
K.clear_session()
model = get_model(embed_matrix)
lookahead = Lookahead(k=5, alpha=0.5) # Initialize Lookahead
lookahead.inject(model) # add into model
model.summary()
early_stopping = EarlyStopping(monitor='val_loss',min_delta=0.0001, patience=2, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=1, min_lr=0.0001, verbose=2)
bst_model_path = out+'chizhurnn_chizhu_weight.h5'
checkpoint = ModelCheckpoint(bst_model_path , monitor='val_loss', mode='min',
                                      save_best_only=True, verbose=1,save_weights_only=True )
callbacks = [checkpoint,reduce_lr,early_stopping]
hist = model.fit_generator(train_gen, steps_per_epoch=int(np.ceil(99000000/2048)),
epochs=10, verbose=1, callbacks=callbacks, 
validation_data=val_gen, validation_steps = int(np.ceil(1000000/2048)),
max_queue_size=10, workers=1, use_multiprocessing=False)

val_gen = gen_train(path='/home/kesci/zhifeng/val.csv', feature=val_feature,
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
                     feature=testa_feature, batch_size=4096, label_tag=False, chunk_size=1, shuffle=False)
prob = model.predict_generator(
    test_gen, steps=int(np.ceil(20000000/4096)), verbose=1)
sub = pd.read_csv('/home/kesci/work/chizhu/submit_rnn.csv',
                  names=['qid', 'aid', 'prob'])
sub['prob'] = prob.flatten()
sub.to_csv('/home/kesci/work/chizhu/chizhu_rnn_testa.csv',
           index=False, header=False)
test_gen = gen_train(path='/home/kesci/input/bytedance/bytedance_contest.final_2.csv',
                     feature=testb_feature, batch_size=4096, label_tag=False, chunk_size=1, shuffle=False)
prob = model.predict_generator(
    test_gen, steps=int(np.ceil(100000000/4096)), verbose=1)
final = pd.read_csv(path+"bytedance_contest.final_2.csv", names=[
                    'query_id', 'query', 'query_title_id', 'title'])[['query_id', 'query_title_id']]
final['prob'] = prob.flatten()
final.to_csv('/home/kesci/work/chizhu/chizhu_rnn_testb.csv',
             index=False, header=False)
