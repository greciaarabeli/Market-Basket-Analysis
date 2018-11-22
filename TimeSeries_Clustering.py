import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import KShape
from sklearn.metrics.pairwise import cosine_similarity
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import sigma_gak, cdist_gak



def plot_data(ks, X_train, y_pred, sz, n_clusters=3, centroid=False):
    plt.figure(figsize=(12, 25))
    for yi in range(n_clusters):
        plt.subplot(n_clusters, 1, 1 + yi)
        for xx in X_train[y_pred == yi]:
            # , alpha=.2
            plt.plot(xx.ravel(), "k-")
            # ,
        if centroid:
            plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
        plt.xlim(0, sz)
        # plt.ylim(-4, 4)
        plt.title("Cluster %d" % (yi + 1))

    plt.tight_layout()
    plt.show()


def normalize_data(data):
    X_train = TimeSeriesScalerMeanVariance().fit_transform(data)
    sz = X_train.shape[1]

    return X_train, sz


def k_shape(X_train, n_clusters, verbose=True, seed=0):
    # Euclidean k-means
    ks = KShape(n_clusters=n_clusters, verbose=verbose, random_state=seed)

    return ks, ks.fit_predict(X_train)


def compute_scores(ks, X_train, y_pred, centroid=False):
    scores = []
    # range(ks.n_clusters)
    for yi in np.unique(y_pred):
        tp_list = []
        for xx in X_train[y_pred == yi]:

            if centroid:
                predicted = ks.cluster_centers_[yi].ravel()
                actual = xx.ravel()
                score = adjusted_rand_score(actual, predicted)
                scores.append(score)

            else:
                predicted = xx.ravel()
                tp_list.append(predicted)

        if not centroid:
            half = len(tp_list) // 2
            first_half = tp_list[:half]
            second_half = tp_list[half:]

            for i in np.arange(half):
                score = adjusted_rand_score(first_half[i], second_half[i])
                scores.append(score)

    return scores


def get_scores(method, true, pred):
    sum_recall = 0
    sum_precision = 0
    sum_fscore = 0
    n = len(true)
    for i in orders_set_test.order_id:
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

    return [method, sum_recall / n, sum_precision / n, sum_fscore / n]

def make_timeseries(train):
    orders = train.drop_duplicates(subset=['order_id', 'user_id'], keep='first').sort_values('order_number')
    orders['week'] = (orders.groupby(['user_id'])['days_since_prior_order'].cumsum() / 7).fillna(0).astype(int)
    orders['week_day'] = orders[['week', 'order_dow']].values.tolist()
    cross_ts = pd.crosstab(orders.user_id, orders.week, values=orders.total, aggfunc='sum')
    cross_ts = cross_ts.fillna(0).values.tolist()
    return cross_ts
