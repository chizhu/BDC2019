### 高校赛解决方案
#### 赛题介绍
* **数据**  

  提供10亿量级的数据，根据query和title预测query下doc点击率。数据已经脱敏并且分好词。

| 列名 | 类型 | 示例 |
| ------ | ------ | ------ |
| query_id | int | 3 |
| query | hash string，term空格分割 | 1 9 117 |
| query_title_id | title在query下的唯一标识 | 2 |
| title | hash string，term空格分割 | 3 9 120 |
| label | int，取值{0, 1} | 0 |
* **任务分析**
  二分类问题。文本相似度+ctr点击预测
* **难点**

  * 数据量大
  * 数据脱敏

#### 解决方案
##### 特征工程(FE)
* 问题长度
* 标题长度
* 标题长度-问题长度
* 问题是否全部在标题里面
* 标题和问题的共词个数
* 标题问题词语的共词个数/问题长度
* 标题问题词语的共词个数/标题长度
* 编辑距离
* 前一个词语是否相同
* 前二个词语是否相同
* 前三个词语是否相同
* 第一个词语在标题里面出现位置
* 第二个词语在标题里面出现位置
* 第三个词语在标题里面出现位置
* 标题求组合后词语
* 词语求组合后标题
* w2v_n_similarity
* fasttext的余弦相似度
* word2vec的余弦相似度

(共19个特征,放入LGB模型lb是0.597)
##### NN模型
* 孪生RNN
   * query+title双输入+FE特征
   * 使用最后一亿的数据（前9.9千万条数据训练+后1百万数据验证）
   * 网络结构
   ```python 
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
        lstm_layer = Bidirectional(CuDNNLSTM(64, return_sequences=True, kernel_initializer=glorot_uniform(seed = 123)))
        gru_layer = Bidirectional(CuDNNGRU(64, return_sequences=True, kernel_initializer=glorot_uniform(seed = 123)))
    
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
   ``` 
   
   * 使用AdamW优化器加快训练过程
   * 使用最新刚出的lookahead 优化器（reference:Lookahead Optimizer: k steps forward, 1 step back(https://arxiv.org/abs/1907.08610))
   Lookahead 算法的性能显著优于 SGD 和 Adam,它迭代地更新两组权重。直观来说，Lookahead 算法通过提前观察另一个优化器生成的「fast weights」序列，来选择搜索方向。该研究发现，Lookahead 算法能够提升学习稳定性，不仅降低了调参需要的功夫，同时还能提升收敛速度与效果。
   * 线上效果
     **lb 0.6214**
* **fine-tuning（亮点）**
  * 思考：官方提供10亿的数据量？先验知识告诉我们，数据越多效果越好，那么如何充分利用数据？
  * 解决方法
     *  先用10亿数据训练一个不加任何特征的裸NN，保存权重(如何能训练10亿？)
        > 文件流处理数据+分批次训练（训练10亿数据最大占用内存才10G） 
    *   加载裸NN模型，获得倒二层的feature map作为输出，加入新的FE特征输入，然后把基模型的feature map和FE特征拼接最后送入全连接层。用最后一亿的数据fine-tuning 整个网络。
        （再次展示预训练在NLP领域的举足轻重不可动摇的地位）
    
  * fine-tuning用到的模型（整体参数都是改小了的，因为只有单卡机器，如果可以多卡训练，放开参数估计单模可以0.64+）
     * word2vec300维+孪生RNN(小参数)  **lb 0.6248**
     * word2vec300维+ESIM（极小参数，最后时刻怕跑不完）    **lb 0.626**
     * fasttext100维+ESIM(小参数)  **lb 0.6336 单模都可以在A榜排到第三**
  * fine-tuning 网络结构
  ```python 
  def aux_esim_model(embed_matrix,model_weight_path):
        base_model = esim(embed_matrix)
        base_model.load_weights(model_weight_path)
        input_q, input_a = base_model.inputs
        input_f = Input((19,))
        hidden_esim = base_model.get_layer(index=28).output
        merged = Concatenate()([hidden_esim, input_f])
        #dense = BatchNormalization()(merged)
        dense = Dense(512, activation='relu')(merged)
        #dense = BatchNormalization()(dense)
        dense = Dropout(0.5)(dense)
        dense = Dense(256, activation='relu')(dense)
        #dense = BatchNormalization()(dense)
        dense = Dropout(0.5)(dense)
        out_ = Dense(1, activation='sigmoid')(dense)

        model = Model(inputs=[input_q,input_a,input_f], outputs=out_)
        model.compile(loss='binary_crossentropy',
                  optimizer=AdamW(lr=0.0003,weight_decay=0.02),
                  metrics=["accuracy"])
        return model    
  ```
  * ESIM 网络结构
  ```python
    def esim(embedding_matrix,
         maxlen=20,
         lstm_dim=64,
         dense_dim=128,
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

   
        merged = Concatenate()([q1_rep, q2_rep])

        dense = BatchNormalization()(merged)
        dense = Dense(dense_dim, activation='elu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(dense_dropout)  (dense)
        dense = Dense(dense_dim, activation='elu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(dense_dropout)(dense)
        out_ = Dense(1, activation='sigmoid')(dense)

        model = Model(inputs=[q1, q2], outputs=out_)
        model.compile(loss='binary_crossentropy',
                  optimizer=AdamW(lr=0.0003,weight_decay=0.02,),
                  metrics=["accuracy",auc])
        return model
  ```
  

#### 线上提交
* finetuning_fasttext_esim(**0.6336**)*0.6+\
  finetuning_w2v_esim(**0.626**)*0.2+\
  finetuning_w2v_esim(**0.6248**)*0.2=**lb 0.6366**
<hr>

* finetuning_fasttext_esim(**0.6336**)*0.5+\
  finetuning_w2v_esim(**0.626**)*0.2+\
  finetuning_w2v_esim(**0.6248**)*0.2+\
  孪生RNN(**0.6214**)*0.1=ensemble_NN 

  lgb(**0.597**)*0.1+ensemble_NN*0.9= **lb 0.6371**


  
  
#### 我们的优势
* 工业可部署
> 真实的线上业务也是庞大的数据量，如何充分利用数据是个难题。我们的方案适用于大数据量（流式训练全量数据内存小+finetuing迁移学习效果佳）

* 简单而实用
> 我们总共才19个特征，不需要提取大量的手工特征，所以可以说不依赖于LGB模型，LGB模型是全量模型，要么只能选用小数据集提特征要么大数据量提取不了特征，不易迭代。我们的方案流式处理，易于迭代更新。






