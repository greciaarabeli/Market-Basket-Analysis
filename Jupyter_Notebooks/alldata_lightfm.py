import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import gc
import Evrecsys
import sys
from itertools import combinations, groupby
from collections import Counter
from scipy import sparse
from lightfm import LightFM
import Evrecsys
import lightfm_form
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import recall_at_k

order_products_prior = pd.read_csv("order_products__prior.csv")

orders = pd.read_csv("orders.csv")


def get_scores(method, true, pred):
    sum_recall = 0
    sum_precision = 0
    sum_fscore = 0
    n = len(true)
    for i in true.index:
        p, r, f1 = Evrecsys.calculate_prf(true[i], pred[i])
        sum_precision = sum_precision + p
        sum_recall = sum_recall + r
        sum_fscore = sum_fscore + f1

    return [method, sum_recall / n, sum_precision / n, sum_fscore / n]


# lightfm
def use_lightfm(train, test):
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


    print('FIN ADDING')

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
    print('FIN ITERACTIONS MATRICES')

    mf_model = lightfm_form.runMF(interactions=interactions_i,
                                  n_components=30, loss='warp', epoch=40, n_jobs=4)

    print('FIN MODEL')


    test_history = test.groupby('order_id').aggregate({'product_id': lambda x: list(x)})
    test_history = pd.merge(test_history, test[['order_id', 'user_id']], on='order_id')
    test_history = test_history.drop_duplicates(subset=['order_id', 'user_id'], keep='first')
    n_users, n_items = interactions_i.shape
    print(interactions_i.shape)
    print(interactions_test_i.shape)
    print(np.arange(n_items))
    results = []
    #test_history['pred'] = 0
    #for user_id in test_history['user_id']:
      #  print(user_id)
      #  recom = mf_model.predict(user_id, np.arange(n_items),num_threads=4)
      #  recom = pd.Series(recom)
      #  recom.sort_values(ascending=False, inplace=True)
      #  if (len(results) == 0):
      #      results = np.array(recom.iloc[0:10].index.values)
      #  else:
     #       results = np.vstack((results, recom.iloc[0:10].index.values))

    #results_df = pd.DataFrame(data=results)
    #columns = results_df.columns.values
    #test_history['pred'] = results_df[columns].values.tolist()

    #y_pred = test_history['pred']
    #y_true = test_history['product_id']

    test_precision = precision_at_k(mf_model, sparse.csr_matrix(interactions_test_i.values), k=10).mean()
    test_recall = recall_at_k(mf_model, sparse.csr_matrix(interactions_test_i.values), k=10).mean()
    f_test = 2 * test_precision * test_recall / (test_precision + test_recall)
    print(f_test)
    return test_precision, test_recall, f_test


orders=orders.loc[orders['eval_set']=='prior']
users_list=np.unique(orders.user_id)

random.Random(4).shuffle(users_list)

users_group={}
x=0
y=len(users_list)
for i in range(x,y,2000):
    x=i
    users_group['users_%s' % i]= (users_list[x:x+2000])

users_group=users_group.values()



results = []
for group in users_group:
    print(group)
    orders_set = orders[orders["user_id"].isin(group)]
    print(np.shape(orders))
    idx_test = orders_set.groupby(['user_id'])['order_number'].transform(max) == orders_set['order_number']
    orders_set_test = orders_set[idx_test]
    print(np.shape(orders_set_test))

    orders_set_testlist = np.unique(orders_set_test.order_id)
    orders_set_train = orders_set[-orders_set["order_id"].isin(orders_set_testlist)]
    orders_set_trainlist = np.unique(orders_set_train.order_id)
    train = order_products_prior[order_products_prior["order_id"].isin(orders_set_trainlist)]
    test = order_products_prior[order_products_prior["order_id"].isin(orders_set_testlist)]
    print(np.shape(train))
    train = pd.merge(train, orders_set_train, on='order_id', how='left')
    test = pd.merge(test, orders_set_test, on='order_id', how='left')
    print(np.shape(test))
    test_precision_light, test_recall_light, f_test_light = use_lightfm(train, test)
    print('finish lightfm')

    results.append([group, group, group, group])

    # lightfm
    #scores_li = get_scores('lightfm', y_true_lightfm, y_pred_lightfm)
    results.append(['lightfm', test_recall_light,  test_precision_light, f_test_light])

result = pd.DataFrame(results,columns=['method', 'recall', 'precision', 'fscore'])
result.to_csv('alldata_lightfm.csv')