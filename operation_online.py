# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 15:13:24 2021

@author: LJC
"""
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')
os.chdir("E:/Competition/xf/UserPersona") 

# 读取数据，简单处理list数据
train = pd.read_csv('./data/train.txt', header=None)
test = pd.read_csv('./apply_new.txt', header=None)

train.columns = ['pid', 'label', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'model', 'make']
test.columns = ['pid', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'model', 'make']

train['label'] = train['label'].astype(int)

data = pd.concat([train,test])
data['label'] = data['label'].fillna(-1)

data['tagid'] = data['tagid'].apply(lambda x:eval(x))
data['tagid'] = data['tagid'].apply(lambda x:[str(i) for i in x])

# 超参数
# embed_size  embedding size
# MAX_NB_WORDS  tagid中的单词出现次数
# MAX_SEQUENCE_LENGTH  输入tagid list的长度
embed_size = 64
MAX_NB_WORDS = 230637
MAX_SEQUENCE_LENGTH = 128
# 训练word2vec，这里可以考虑elmo，bert等预训练
w2v_model = Word2Vec(sentences=data['tagid'].tolist(), vector_size=embed_size, window=5, min_count=1,epochs=10)

# 这里是划分训练集和测试数据
X_train = data[:train.shape[0]]['tagid']
X_test = data[train.shape[0]:]['tagid']

# 创建词典，利用了tf.keras的API，其实就是编码一下，具体可以看看API的使用方法
tokenizer = text.Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
word_index = tokenizer.word_index
# 计算一共出现了多少个单词，其实MAX_NB_WORDS我直接就用了这个数据

nb_words = len(word_index) + 1
print('Total %s word vectors.' % nb_words)
# 构建一个embedding的矩阵，之后输入到模型使用
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    try:
        embedding_vector = w2v_model.wv.get_vector(word)
    except KeyError:
        continue
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

y_categorical = train['label'].values

def my_model():
    embedding_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    # 词嵌入（使用预训练的词向量）
    embedder = Embedding(nb_words,
                         embed_size,
                         input_length=MAX_SEQUENCE_LENGTH,
                         weights=[embedding_matrix],
                         trainable=False
                         )
    embed = embedder(embedding_input)
    l = LSTM(128)(embed)
    flat = BatchNormalization()(l)
    drop = Dropout(0.2)(flat)
    main_output = Dense(1, activation='sigmoid')(drop)
    model = Model(inputs=embedding_input, outputs=main_output)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy'])
    return model

# 五折交叉验证
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
oof = np.zeros([len(train), 1])
predictions = np.zeros([len(test), 1])

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
    print("fold n{}".format(fold_ + 1))
    model = my_model()
    if fold_ == 0:
        model.summary()

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)
    bst_model_path = "./{}.h5".format(fold_)
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    X_tra, X_val = X_train[trn_idx], X_train[val_idx]
    y_tra, y_val = y_categorical[trn_idx], y_categorical[val_idx]

    model.fit(X_tra, y_tra,
              validation_data=(X_val, y_val),
              epochs=128, batch_size=256, shuffle=True,
              callbacks=[early_stopping, model_checkpoint])

    model.load_weights(bst_model_path)

    oof[val_idx] = model.predict(X_val)

    predictions += model.predict(X_test) / folds.n_splits
    print(predictions)
    del model

train['predict'] = oof
train['rank'] = train['predict'].rank()
train['p'] = 1
train.loc[train['rank'] <= train.shape[0] * 0.5, 'p'] = 0
bst_f1_tmp = f1_score(train['label'].values, train['p'].values)
print(bst_f1_tmp)

submit = test[['pid']]
submit['tmp'] = predictions
submit.columns = ['user_id', 'tmp']

submit['rank'] = submit['tmp'].rank()
submit['category_id'] = 1
submit.loc[submit['rank'] <= int(submit.shape[0] * 0.5), 'category_id'] = 0

print(submit['category_id'].mean())

submit[['user_id', 'category_id']].to_csv('open_{}.csv'.format(str(bst_f1_tmp).split('.')[1]), index=False)