# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 14:04:22 2021

@author: LJC
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from gensim.models import Word2Vec


os.chdir("E:/Competition/xf/UserPersona") 

train = pd.read_csv('./data/train.txt', header=None)
test = pd.read_csv('./data/apply_new.txt', header=None)

train.columns = ['pid', 'label', 'gender', 'age', 'tagid', 
                 'time', 'province', 'city', 'model', 'make']
test.columns = ['pid', 'gender', 'age', 'tagid', 'time', 
                'province', 'city', 'model', 'make']

data = pd.concat([train, test])
data['label'] = data['label'].fillna(-1)

# 构造sentences，每个元素转化为str
data['tagid'] = data['tagid'].apply(lambda x: eval(x))
data['tagid'] = data['tagid'].apply(lambda x:[str(i) for i in x])
sentences = data['tagid'].values.tolist()


## Embedding method 1 
emb_size = 32
model = Word2Vec(sentences,
                 vector_size=emb_size,
                 window=6,
                 min_count=5,
                 sg=0,
                 hs=0,
                 seed=1,
                 epochs=5)


emb_matrix = list()
for seq in sentences:
    vec = list()
    for w in seq:
        if w in model.wv.index_to_key:
            vec.append(model.wv[w])
    if len(vec) > 0:
        emb_matrix.append(np.mean(vec, axis=0))
    else:
        emb_matrix.append([0] * emb_size)

emb_matrix = np.array(emb_matrix)
for i in range(emb_size):
    data['tag_emb_{}'.format(i)] = emb_matrix[:, i]

np.savetxt('E:/Competition/xf/UserPersona/output/emb_mat.csv', emb_matrix, delimiter = ',')



## construct training data and test data
cat_cols = ['gender', 'age', 'province', 'city']
data[cat_cols] = data[cat_cols].astype('category')
X_train = data[~data['label'].isna()]
X_test = data[data['label'].isna()]

y = X_train['label']
KF = StratifiedKFold(n_splits=5, random_state=2021, shuffle=True)
params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'learning_rate': 0.05,
    'subsample': 0.8,
    'subsample_freq': 3,
    'colsample_btree': 0.8,
    'num_iterations': 10000,
    'verbose': -1
}
oof_lgb = np.zeros(len(X_train))
predictions_lgb = np.zeros((len(X_test)))


## 特征重要性
features = [
    i for i in data.columns
    if i not in ['pid', 'label', 'tagid', 'time', 'model', 'make']
]
feat_imp_df = pd.DataFrame({'feat': features, 'imp': 0})


## 五折交叉验证
for fold_, (trn_idx, val_idx) in enumerate(KF.split(X_train.values, y.values)):
    print("fold {}".format(fold_))
    print('trn_idx:', trn_idx)
    print('val_idx:', val_idx)
    trn_data = lgb.Dataset(X_train.iloc[trn_idx][features],
                           label=y.iloc[trn_idx])
    val_data = lgb.Dataset(X_train.iloc[val_idx][features],
                           label=y.iloc[val_idx])
    num_round = 10000
    clf = lgb.train(
        params,
        trn_data,
        num_round,
        valid_sets=[trn_data, val_data],
        verbose_eval=100,
        early_stopping_rounds=50,
        categorical_feature=cat_cols,
    )
    feat_imp_df['imp'] += clf.feature_importance() / 5
    oof_lgb[val_idx] = clf.predict(X_train.iloc[val_idx][features],
                                   num_iteration=clf.best_iteration)
    predictions_lgb[:] += clf.predict(X_test[features],
                                      num_iteration=clf.best_iteration)
    

# evaluation
auc = roc_auc_score(y, oof_lgb)
print("AUC score: {}".format(auc))
f1 = f1_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])
print("F1 score: {}".format(f1_score))
precision = precision_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])
print("Precision score: {}".format(precision))
recall = recall_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])
print("Recall score: {}".format(recall))


# save the results
X_test['category_id'] = [1 if i >= 2.5 else 0 for i in predictions_lgb]
X_test['user_id'] = X_test['pid']
output_path = "{:.6f}.csv".format(f1).lstrip('0.')
X_test[['user_id', 'category_id']].to_csv(output_path, index=False)