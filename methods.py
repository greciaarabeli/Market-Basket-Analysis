import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import sigma_gak, cdist_gak
from sklearn.metrics.cluster import adjusted_rand_score
import random
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
import xgboost
import gc
import Evrecsys
import sys
from itertools import combinations, groupby
from collections import Counter
from scipy import sparse
from lightfm import LightFM
from sklearn.metrics.pairwise import cosine_similarity
import Apriori
import lightfm_form
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import recall_at_k
from lightfm import LightFM
import TimeSeries_Clustering


### TOP 10 ###
def top10(train, test, return_pred, num_cluster):
    top_reorder_train=train.groupby("product_id")["reordered"].aggregate({'Total_reorders': 'sum'})['Total_reorders'].sort_values(ascending=False).head(10)
    top_reorder_train=np.array(top_reorder_train.values)
    test_history=test[test['reordered']==1].groupby('order_id').aggregate({'product_id':lambda x: list(x)})
    set1 = set(np.unique(test.order_id))
    set2 = set(np.unique(test_history.index))
    missing = pd.DataFrame.from_dict(list(sorted(set1 - set2)))
    missing['product_id']='NaN'
    missing = missing.rename(index=str, columns={0: "order_id"})
    y_true = test_history.reset_index()
    y_true = y_true.append(missing).reset_index()
    n=len(y_true)
    print(n)
    sum_precision=0
    sum_recall=0
    sum_fscore=0
    for i in y_true.index:
        p,r,f1=Evrecsys.calculate_prf(y_true.product_id[i],top_reorder_train)
        sum_precision=sum_precision+p
        sum_recall=sum_recall+r
        sum_fscore=sum_fscore+f1

    print('FINISH TOP 10')

    if return_pred==0:
        return sum_recall/n,sum_precision/n ,sum_fscore/n
    else:
        return sum_recall/n,sum_precision/n ,sum_fscore/n, top_reorder_train, test_history['product_id']

### LAST ORDER ###
def last_order(train, test, return_pred, num_cluster):
    last_orders = train.groupby("user_id")["order_number"].aggregate(np.max)
    t = pd.merge(left=last_orders.reset_index(), right=train, how='inner', on=['user_id', 'order_number'])
    t_last_order = t.groupby('order_id').aggregate({'product_id': lambda x: list(x)})
    t_last_order = pd.merge(t_last_order, train[['order_id', 'user_id']], on='order_id')
    t_last_order = t_last_order.drop_duplicates(subset=['order_id', 'user_id'], keep='first')

    test_history = test[test['reordered']==1].groupby('order_id').aggregate({'product_id': lambda x: list(x)})
    set1 = set(test.order_id.unique())
    set2 = set(test_history.index)
    missing = pd.DataFrame.from_dict(list(sorted(set1 - set2)))
    missing['product_id'] = 'NaN'
    missing = missing.rename(index=str, columns={0: "order_id"})
    test_history = test_history.reset_index()
    test_history = test_history.append(missing)
    test_history = pd.merge(test_history, test[['order_id', 'user_id']], on='order_id')
    test_history = test_history.drop_duplicates(subset=['order_id', 'user_id'], keep='first')

    t_last_order = pd.merge(t_last_order, test_history, on='user_id')
    t_last_order = t_last_order.sort_values('user_id').fillna('NaN')
    y_pred=t_last_order['product_id_x']
    y_true=t_last_order['product_id_y']

    sum_recall = 0
    sum_precision = 0
    sum_fscore = 0
    n = len(y_true)
    print(n)
    for i in y_true.index:
        p, r, f1 = Evrecsys.calculate_prf(y_true[i], y_pred[i])
        sum_precision = sum_precision + p
        sum_recall = sum_recall + r
        sum_fscore = sum_fscore + f1

    print('FINISH LAST ORDER')

    if return_pred==0:
        return sum_recall/n,sum_precision/n ,sum_fscore/n
    else:
        return sum_recall/n,sum_precision/n ,sum_fscore/n, y_pred, y_true



### LAST REORDER ###
def last_reorder(train, test, return_pred, num_cluster):
    last_orders = train.groupby("user_id")["order_number"].aggregate(np.max)
    t = pd.merge(left=last_orders.reset_index(), right=train[train.reordered == 1], how='inner',
                 on=['user_id', 'order_number'])
    t_last_order = t.groupby('order_id').aggregate({'product_id': lambda x: list(x)})
    t_last_order = pd.merge(t_last_order, train[['order_id', 'user_id']], on='order_id')
    t_last_order = t_last_order.drop_duplicates(subset=['order_id', 'user_id'], keep='first')

    test_history = test[test['reordered']==1].groupby('order_id').aggregate({'product_id': lambda x: list(x)})
    set1 = set(test.order_id.unique())
    set2 = set(test_history.index)
    missing = pd.DataFrame.from_dict(list(sorted(set1 - set2)))
    missing['product_id'] = 'NaN'
    missing = missing.rename(index=str, columns={0: "order_id"})
    test_history = test_history.reset_index()
    test_history = test_history.append(missing)
    test_history = pd.merge(test_history, test[['order_id', 'user_id']], on='order_id')
    test_history = test_history.drop_duplicates(subset=['order_id', 'user_id'], keep='first')

    t_last_order = pd.merge(t_last_order, test_history, on='user_id', how='right')
    t_last_order = t_last_order.sort_values('user_id').fillna('NaN')
    y_pred=t_last_order['product_id_x']
    y_true=t_last_order['product_id_y']

    sum_recall = 0
    sum_precision = 0
    sum_fscore = 0
    n = len(y_true)
    print(n)
    for i in y_true.index:
        p, r, f1 = Evrecsys.calculate_prf(y_true[i], y_pred[i])
        sum_precision = sum_precision + p
        sum_recall = sum_recall + r
        sum_fscore = sum_fscore + f1

    print('FINISH LAST REORDER')

    if return_pred==0:
        return sum_recall/n,sum_precision/n ,sum_fscore/n
    else:
        return sum_recall/n,sum_precision/n ,sum_fscore/n, y_pred, y_true

### APRIORI ###
def apriori(train, test, return_pred, num_cluster):
    train_orders_i = train.set_index('order_id')['product_id'].rename('item_id')
    test_orders_i = test.set_index('order_id')['product_id'].rename('item_id')

    #item_name = train['product_id', 'product_name', 'aisle_id', 'department_id'].rename(columns={'product_id': 'item_id', 'product_name': 'item_name'})
    rules_i = Apriori.association_rules(train_orders_i, 0.01)
    #rules_final_i = Apriori.merge_item_name(rules_i, item_name).sort_values('lift', ascending=False)
    #display(rules_final_i)

    # Train set pairs
    train_pairs_gen_i = Apriori.get_item_pairs(train_orders_i)
    train_pairs_i = Apriori.freq(train_pairs_gen_i).to_frame("freqAB")
    train_pairs_i = train_pairs_i.reset_index().rename(columns={'level_0': 'item_A', 'level_1': 'item_B'})
    train_pairs_i['pair'] = train_pairs_i.item_A.astype(str).str.cat(train_pairs_i.item_B.astype(str), sep='-')

    # Test set pairs
    test_pairs_gen_i = Apriori.get_item_pairs(test_orders_i)
    test_pairs_i = Apriori.freq(test_pairs_gen_i).to_frame("freqAB")
    test_pairs_i = test_pairs_i.reset_index().rename(columns={'level_0': 'item_A', 'level_1': 'item_B'})
    test_pairs_i['pair'] = test_pairs_i.item_A.astype(str).str.cat(test_pairs_i.item_B.astype(str), sep='-')

    # Rules set pairs
    rules_i['pair'] = rules_i.item_A.astype(str).str.cat(rules_i.item_B.astype(str), sep='-')

    test_pair_set_i = set(np.unique(test_pairs_i.pair))
    train_pair_set_i = set(np.unique(train_pairs_i.pair))
    rules_pair_set_i = set(np.unique(rules_i.pair))

    # TP= Pairs that exist in a priori pred and test
    tp = len(list(test_pair_set_i & rules_pair_set_i))

    # TN= pairs that exists train set but not in test
    tn = len(list(test_pair_set_i - train_pair_set_i))

    # FN= Pairs that exists in test but not in a priori
    fn = len(list(rules_pair_set_i - test_pair_set_i))

    # FP= Pairs that exists in a priori but not in test
    fp = len(list(test_pair_set_i - rules_pair_set_i))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (recall * precision) / (recall + precision)
    print('APRIORI')
    return recall, precision, f1


### LIGHTFM ###
def lightfm(train, test, return_pred, num_cluster):
    set1 = set(np.unique(train.product_id))
    set2 = set(np.unique(test.product_id))
    missing = pd.DataFrame.from_dict(list(sorted(set1 - set2)))
    added = pd.DataFrame.from_dict(list(sorted(set2 - set1)))

    for i in range(len(missing)):
        a = missing[0][i]
        test = test.append({'product_id': a}, ignore_index=True)
    for i in range(len(added)):
        a = added[0][i]
        train = train.append({'product_id': a}, ignore_index=True)

    train = train.fillna(0)
    test = test.fillna(0)

    grouped_train_i = train.groupby(["user_id", "product_id"])["reordered"].aggregate("sum").reset_index()
    grouped_test_i = test.groupby(["user_id", "product_id"])["reordered"].aggregate("sum").reset_index()

    interactions_i = lightfm_form.create_interaction_matrix(df=grouped_train_i,
                                                            user_col='user_id',
                                                            item_col='product_id',
                                                            rating_col='reordered')

    interactions_test_i = lightfm_form.create_interaction_matrix(df=grouped_test_i,
                                                                 user_col='user_id',
                                                                 item_col='product_id',
                                                                 rating_col='reordered')

    mf_model = lightfm_form.runMF(interactions=interactions_i,
                                  n_components=30, loss='warp', epoch=40, n_jobs=4)

    #test_history = test[test['reordered'] == 1].groupby('order_id').aggregate({'product_id': lambda x: list(x)})
    #test_history = pd.merge(test_history, test[['order_id', 'user_id']], on='order_id')
    #test_history = test_history.drop_duplicates(subset=['order_id', 'user_id'], keep='first')
    #n_users, n_items = interactions_i.shape

    #results = []
    #test_history['pred'] = 0
    #for user_id in test_history['user_id']:
    #    print(user_id)
    #    recom = mf_model.predict(user_id, np.arange(n_items), num_threads=4)
    #    recom = pd.Series(recom)
    #    recom.sort_values(ascending=False, inplace=True)
    #    if (len(results) == 0):
    #        results = np.array(recom.iloc[0:10].index.values)
    #    else:
    #        results = np.vstack((results, recom.iloc[0:10].index.values))

    #results_df = pd.DataFrame(data=results)
    #columns = results_df.columns.values
    #test_history['pred'] = results_df[columns].values.tolist()

    #y_pred = test_history['pred']
    #y_true = test_history['product_id']

    test_precision = precision_at_k(mf_model, sparse.csr_matrix(interactions_test_i.values), k=10).mean()
    test_recall = recall_at_k(mf_model, sparse.csr_matrix(interactions_test_i.values), k=10).mean()
    f_test = 2 * test_precision * test_recall / (test_precision + test_recall)

    print('FINISH LIGHTFM')
    if return_pred==0:
        return test_recall, test_precision, f_test
    else:
        return test_recall, test_precision, f_test, y_pred, y_true


### XGBOOST ###
def do_xgboost(train, test, return_pred, num_cluster):
    param = {'max_depth': 10,
             'eta': 0.02,
             'colsample_bytree': 0.4,
             'subsample': 0.75,
             'silent': 1,
             'nthread': 27,
             'eval_metric': 'logloss',
             'objective': 'binary:logistic',
             'tree_method': 'hist'
             }
    orders_set_test=test.order_id.unique()
    y_train = train['reordered']
    X_train = train.drop(['reordered', 'eval_set', 'batch', 'total','product_name', 'add_to_cart_order'], axis=1)

    X_test = test.drop_duplicates(subset=['order_id', 'user_id'], keep='first')
    X_test=X_test.drop(['product_id','add_to_cart_order', 'reordered', 'eval_set', 'product_name', 'aisle_id', 'department_id', 'batch'], axis=1)
    X_train_sub=train.drop_duplicates(subset=['product_id', 'user_id'], keep='first')
    X_train_sub=X_train_sub[['product_id', 'user_id', 'aisle_id','department_id']]
    X_test=pd.merge(left=X_test, right=X_train_sub, how='right',on=['user_id'])
    X_test = X_test[['order_id', 'product_id', 'user_id', 'order_number', 'order_dow','order_hour_of_day', 'days_since_prior_order', 'aisle_id','department_id']]
    X_train = X_train[['order_id', 'product_id', 'user_id', 'order_number', 'order_dow','order_hour_of_day', 'days_since_prior_order', 'aisle_id','department_id']]
    dtrain = xgboost.DMatrix(X_train, label=y_train)
    dtest = xgboost.DMatrix(X_test)
    model = xgboost.train(param, dtrain)
    predict_labels = model.predict(dtest)
    
    X_test['pred']=predict_labels
    X_test['pred'] = (X_test['pred'] > X_test['pred'].mean()) * 1
    pred = X_test[X_test['pred'] == 1].groupby('order_id').aggregate({'product_id': lambda x: list(x)})
    true =test[test['reordered'] == 1].groupby('order_id').aggregate({'product_id': lambda x: list(x)})
    sum_recall = 0
    sum_precision = 0
    sum_fscore = 0
    n = len(orders_set_test)
    for i in orders_set_test:
        if i in true.index:
            y = true['product_id'][i]
        else:
            y = 'nan'
        if i in pred.index:
            y_hat = pred['product_id'][i][:10]
        else:
            y_hat = 'nan'

        p, r, f1 = Evrecsys.calculate_prf(y, y_hat)
        sum_precision = sum_precision + p
        sum_recall = sum_recall + r
        sum_fscore = sum_fscore + f1

    print('FINISH XGBOOST')

    if return_pred==0:
        return sum_recall/n,sum_precision/n ,sum_fscore/n
    else:
        return sum_recall/n,sum_precision/n ,sum_fscore/n, y_pred, y_true


### XGBOOST KSHAPE###
def xgb_kshape(train, test, return_pred, num_cluster):
    ts=TimeSeries_Clustering.make_timeseries(train)

    formatted_dataset = to_time_series_dataset(ts)
    X_train, sz = TimeSeries_Clustering.normalize_data(formatted_dataset)
    ks, y_pred = TimeSeries_Clustering.k_shape(X_train, n_clusters=num_cluster)
    scores = TimeSeries_Clustering.compute_scores(ks, X_train, y_pred)
    plt.boxplot(scores)

    TimeSeries_Clustering.plot_data(ks, X_train, y_pred, sz, ks.n_clusters)
    y_pred_df = pd.DataFrame(y_pred)
    userindex = train.user_id.unique()
    userindex = np.sort(userindex)
    y_pred_df['user_id'] = userindex
    test = test.merge(y_pred_df, on='user_id').rename({0: 'cluster'}, axis='columns')
    train = train.merge(y_pred_df, on='user_id').rename({0: 'cluster'}, axis='columns')

    sum_recall = 0
    sum_precision = 0
    sum_fscore = 0

    clusters = test.cluster.unique()
    for cluster in clusters:
        train_i = train[train['cluster'] == cluster]
        test_i = test[test['cluster'] == cluster]
        recall, precision, fscore = do_xgboost(train_i, test_i, return_pred, num_cluster)
        sum_recall = sum_recall + recall
        sum_precision = sum_precision + precision
        sum_fscore = sum_fscore + fscore
    n = len(clusters)
    print('FINISH XGBOOST KSHAPE')
    return sum_recall / n, sum_precision / n, sum_fscore / n

### XGBOOST DTW ###
def xgb_dtw(train, test, return_pred, num_cluster):
    ts=TimeSeries_Clustering.make_timeseries(train)

    formatted_dataset = to_time_series_dataset(ts)
    X_train, sz = TimeSeries_Clustering.normalize_data(formatted_dataset)
    dtw_km = TimeSeriesKMeans(n_clusters=num_cluster, metric="dtw", max_iter_barycenter=10, verbose=True, random_state=0)
    y_pred = dtw_km.fit_predict(X_train)
    scores = TimeSeries_Clustering.compute_scores(dtw_km, X_train, y_pred)
    plt.boxplot(scores)
    TimeSeries_Clustering.plot_data(dtw_km, X_train, y_pred, sz, dtw_km.n_clusters, centroid=True)
    y_pred_df = pd.DataFrame(y_pred)
    userindex = train.user_id.unique()
    userindex = np.sort(userindex)
    y_pred_df['user_id'] = userindex
    test = test.merge(y_pred_df, on='user_id').rename({0: 'cluster'}, axis='columns')
    train = train.merge(y_pred_df, on='user_id').rename({0: 'cluster'}, axis='columns')

    sum_recall = 0
    sum_precision = 0
    sum_fscore = 0

    clusters = test.cluster.unique()
    for cluster in clusters:
        train_i = train[train['cluster'] == cluster]
        test_i = test[test['cluster'] == cluster]
        recall, precision, fscore = do_xgboost(train_i, test_i, return_pred, num_cluster)
        sum_recall = sum_recall + recall
        sum_precision = sum_precision + precision
        sum_fscore = sum_fscore + fscore
    n = len(clusters)
    print('FINISH XGBOOST DTW')
    return sum_recall / n, sum_precision / n, sum_fscore / n

### XGBOOST SOFTDTW ###
def xgb_softdtw(train, test, return_pred, num_cluster):
    ts=TimeSeries_Clustering.make_timeseries(train)

    formatted_dataset = TimeSeries_Clustering.to_time_series_dataset(ts)
    X_train, sz = normalize_data(formatted_dataset)
    sdtw_km = TimeSeriesKMeans(n_clusters=num_cluster, metric="softdtw", metric_params={"gamma_sdtw": .01}, verbose=True,random_state=0)
    y_pred = sdtw_km.fit_predict(X_train)
    scores = TimeSeries_Clustering.compute_scores(sdtw_km, X_train, y_pred)
    plt.boxplot(scores)
    TimeSeries_Clustering.plot_data(sdtw_km, X_train, y_pred, sz, sdtw_km.n_clusters, centroid=True)
    y_pred_df = pd.DataFrame(y_pred)
    userindex = train.user_id.unique()
    userindex = np.sort(userindex)
    y_pred_df['user_id'] = userindex
    test = test.merge(y_pred_df, on='user_id').rename({0: 'cluster'}, axis='columns')
    train = train.merge(y_pred_df, on='user_id').rename({0: 'cluster'}, axis='columns')

    sum_recall = 0
    sum_precision = 0
    sum_fscore = 0

    clusters = test.cluster.unique()
    for cluster in clusters:
        train_i = train[train['cluster'] == cluster]
        test_i = test[test['cluster'] == cluster]
        recall, precision, fscore = do_xgboost(train_i, test_i, return_pred, num_cluster)
        sum_recall = sum_recall + recall
        sum_precision = sum_precision + precision
        sum_fscore = sum_fscore + fscore
    n = len(clusters)
    print('FINISH SOFTDTW')
    return sum_recall / n, sum_precision / n, sum_fscore / n


