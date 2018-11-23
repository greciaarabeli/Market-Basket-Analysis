import numpy as np
import pandas as pd
import methods
from tqdm import tqdm
import TimeSeries_Clustering
train = pd.read_csv("1_train.csv", index_col=0)
test = pd.read_csv("1_test.csv", index_col=0)

methods_list = [methods.top10, methods.last_order, methods.last_reorder, methods.apriori, methods.lightfm, methods.do_xgboost, methods.xgboost_kshape, methods.xgboost_dtw, methods.xgboost_softdtw]

methods_name=['top_10', 'apriori', 'last_order', 'last_reorder', 'lightfm', 'xgboost', 'xgboost_kshape', 'xgboost_dtw', 'xgboost_softdtw']
batches=test.batch.unique()

scores_batch=pd.DataFrame(columns=['batch', 'method', 'recall', 'precision', 'fscore'])
num_cluster=10
return_pred=0
for batch in batches:
    i = 0
    train_batch=train[train["batch"]==batch]
    test_batch=test[test["batch"]==batch]
    for method in methods_list:
        recall, precision, fscore=method(train_batch, test_batch, return_pred, num_cluster) #1 if yes 0 if not
        scores_batch.append([batch, methods_name[i], recall, precision, fscore])
        i = i + 1



scores_batch.to_csv('scores_batch.csv')
