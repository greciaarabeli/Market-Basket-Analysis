import pandas as pd
import numpy as np
import sys
from itertools import combinations, groupby
from collections import Counter
from scipy import sparse
from lightfm import LightFM
from sklearn.metrics.pairwise import cosine_similarity
import Apriori
import Evaluation_metrics


#Import data
order_products_prior = pd.read_csv("order_products__prior.csv")
orders = pd.read_csv("orders.csv")
products = pd.read_csv("products.csv")

orders=orders.loc[orders['eval_set']=='prior']

#We will work with 5000 users, last order will be in test set
number_users=5000
users=range(1,number_users+1)
orders_set=orders[orders["user_id"].isin(users)]
idx_test = orders_set.groupby(['user_id'])['order_number'].transform(max) == orders_set['order_number']
orders_set_test=orders_set[idx_test]

orders_set_testlist=np.unique(orders_set_test.order_id)
orders_set_train = orders_set[-orders_set["order_id"].isin(orders_set_testlist)]
orders_set_trainlist=np.unique(orders_set_train.order_id)
train= order_products_prior[order_products_prior["order_id"].isin(orders_set_trainlist)]
test=order_products_prior[order_products_prior["order_id"].isin(orders_set_testlist)]
train = pd.merge(train, orders_set_train, on='order_id', how='left')
test=pd.merge(test, orders_set_test, on='order_id', how='left')

print('Train set has %f orders',len(np.unique(train.order_id)))
print('Test set has %f orders',len(np.unique(test.order_id)))
print('Train set has %f users',len(np.unique(train.user_id)))
print('Test set has %f users',len(np.unique(test.user_id)))

def last_order(train, test):
    last_orders = train.groupby("user_id")["order_number"].aggregate(np.max)
    t = pd.merge(left=last_orders.reset_index(), right=train, how='inner', on=['user_id', 'order_number'])
    t_last_order = t.groupby('order_id').aggregate({'product_id': lambda x: list(x)})
    t_last_order = pd.merge(t_last_order, train[['order_id', 'user_id']], on='order_id')
    t_last_order = t_last_order.drop_duplicates(subset=['order_id', 'user_id'], keep='first')

    test_history = test.groupby('order_id').aggregate({'product_id': lambda x: list(x)})
    test_history = pd.merge(test_history, test[['order_id', 'user_id']], on='order_id')
    test_history = test_history.drop_duplicates(subset=['order_id', 'user_id'], keep='first')

    t_last_order = pd.merge(t_last_order, test_history, on='user_id')
    t_last_order = t_last_order.sort_values('user_id')
    y_pred=t_last_order['product_id_x']
    y_true=t_last_order['product_id_y']

    return y_pred , y_true

def last_order_reorder(train, test):
    last_orders = train.groupby("user_id")["order_number"].aggregate(np.max)
    t = pd.merge(left=last_orders.reset_index(), right=train[train.reordered == 1], how='inner',
                 on=['user_id', 'order_number'])
    t_last_order = t.groupby('order_id').aggregate({'product_id': lambda x: list(x)})
    t_last_order = pd.merge(t_last_order, train[['order_id', 'user_id']], on='order_id')
    t_last_order = t_last_order.drop_duplicates(subset=['order_id', 'user_id'], keep='first')

    test_history = test.groupby('order_id').aggregate({'product_id': lambda x: list(x)})
    test_history = pd.merge(test_history, test[['order_id', 'user_id']], on='order_id')
    test_history = test_history.drop_duplicates(subset=['order_id', 'user_id'], keep='first')

    t_last_order = pd.merge(t_last_order, test_history, on='user_id')
    t_last_order = t_last_order.sort_values('user_id')
    y_pred=t_last_order['product_id_x']
    y_true=t_last_order['product_id_y']

    return y_pred , y_true

def apriori(train_i, test_i):
    train_orders_i = train_i.set_index('order_id')['product_id'].rename('item_id')
    test_orders_i = test_i.set_index('order_id')['product_id'].rename('item_id')
    rules_i = association_rules(train_orders_i, 0.01)
    rules_final_i = merge_item_name(rules_i, item_name).sort_values('lift', ascending=False)
    display(rules_final_i)

    # Train set pairs
    train_pairs_gen_i = get_item_pairs(train_orders_i)
    train_pairs_i = freq(train_pairs_gen_i).to_frame("freqAB")
    train_pairs_i = train_pairs_i.reset_index().rename(columns={'level_0': 'item_A', 'level_1': 'item_B'})
    train_pairs_i['pair'] = train_pairs_i.item_A.astype(str).str.cat(train_pairs_i.item_B.astype(str), sep='-')

    # Test set pairs
    test_pairs_gen_i = get_item_pairs(test_orders_i)
    test_pairs_i = freq(test_pairs_gen_i).to_frame("freqAB")
    test_pairs_i = test_pairs_i.reset_index().rename(columns={'level_0': 'item_A', 'level_1': 'item_B'})
    test_pairs_i['pair'] = test_pairs_i.item_A.astype(str).str.cat(test_pairs_i.item_B.astype(str), sep='-')

    # Rules set pairs
    rules_i['pair'] = rules_i.item_A.astype(str).str.cat(rules_i.item_B.astype(str), sep='-')

    test_pair_set_i = set(np.unique(test_pairs_i.pair))
    train_pair_set_i = set(np.unique(train_pairs_i.pair))
    rules_pair_set_i = set(np.unique(rules_i.pair))
    return rules_pair_set_i, test_pair_set_i

def lightfm(train, test):





    precision = precision_at_k(mf_model_i, sparse.csr_matrix(interactions_i.values)).mean()

    auc = auc_score(mf_model_i, sparse.csr_matrix(interactions_i.values)).mean()

    recall = recall_at_k(mf_model_i, sparse.csr_matrix(interactions_i.values)).mean()

    f_train = 2 * train_precision * train_recall / (train_precision + train_recall)

    map =
    results.append(
        [i, train_precision, test_precision, train_auc, test_auc, train_recall, test_recall, f_train, f_test])
