import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
from sklearn.preprocessing import label_binarize


####### PREDICTION ACCURACY #######

#MAE
def calculate_mae(y_actual, y_predicted):
    return mean_absolute_error(y_true, y_pred)

#MSE
def calculate_mse(y_actual, y_predicted):
    """
    Determines the Root Mean Square Error of the predictions.
    Args:
        y_actual: actual ratings in the format of an array of [ (userId, itemId, actualRating) ]
        y_predicted: predicted ratings in the format of an array of [ (userId, itemId, predictedRating) ]
    Assumptions:
        y_actual and y_predicted are in the same order.
    """
    return mean_squared_error(y_actual, y_predicted)

#RMSE
def calculate_rmse(y_actual, y_predicted):
    """
    Determines the Root Mean Square Error of the predictions.
    Args:
        y_actual: actual ratings in the format of an array of [ (userId, itemId, actualRating) ]
        y_predicted: predicted ratings in the format of an array of [ (userId, itemId, predictedRating) ]
    Assumptions:
        y_actual and y_predicted are in the same order.
    """
    return sqrt(mean_squared_error(y_actual, y_predicted))


###### PRECISION OF RECOMMENDATION ######

#Precision, Recall, fscore
def calculate_prf(y_true, y_pred):
    y_true = set(y_true)
    y_pred = set(y_pred)
    cross_size = len(y_true & y_pred)
    if cross_size == 0: return 0,0,0.
    p = 1. * cross_size / len(y_pred)
    r = 1. * cross_size / len(y_true)
    f1= 2 * p * r / (p + r)
    return p,r,f1

#AUC
def auc_score(predictions, test):
    '''
    This simple function will output the area under the curve using sklearn's metrics.

    parameters:

    - predictions: your prediction output

    - test: the actual target result you are comparing to

    returns:

    - AUC (area under the Receiver Operating Characterisic curve)
    '''
    test=label_binarize(test, classes=range(49688))
    predictions=label_binarize(predictions, classes=range(49688))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(49688):
        fpr[i], tpr[i], _ = metrics.roc_curve(test[:, i], predictions[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(test.ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc["micro"]

###### RANKING PRECISION #####

#Mean Average Precision (MAP)
def apk(y_true, y_pred, k=10):
    if len(y_pred) > k:
        y_pred = y_pred[:k]
    score: int = 0
    num_hits = 0

    for i, p in enumerate(y_pred):
        if p in y_true and p not in y_pred[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / (min(len(y_true), k))

#DCG
def dcg_score(y_true, y_score, k=10, gain='exponential'):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gain == "exponential":
        gain = 2 ** y_true - 1
    elif gain == "linear":
        gain = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)

#### USER EXPERIENECES ###

#Coverage
def calculate_catalog_coverage(y_test, y_train, y_predicted):
    """
    Calculates the percentage of user-item pairs that were predicted by the algorithm.
    The full data is passed in as y_test and y_train to determine the total number of potential user-item pairs
    Then the predicted data is passed in to determine how many user-item pairs were predicted.
    It is very important to NOT pass in the sorted and cut prediction RDD and that the algorithm trys to predict all pairs
    The use the function 'cartesian' as shown in line 25 of content_based.py is helpful in that regard
    Args:
        y_test: the data used to test the RecSys algorithm in the format of an RDD of [ (userId, itemId, actualRating) ]
        y_train: the data used to train the RecSys algorithm in the format of an RDD of [ (userId, itemId, actualRating) ]
        y_predicted: predicted ratings in the format of a RDD of [ (userId, itemId, predictedRating) ].  It is important that this is not the sorted and cut prediction RDD
    Returns:
        catalog_coverage: value representing the percentage of user-item pairs that were able to be predicted
    """

    y_full_data = y_test.union(y_train)

    prediction_count = y_predicted.count()
    #obtain the number of potential users and items from the actual array as the algorithms cannot predict something that was not trained
    num_users = y_full_data.map(lambda row: row[0]).distinct().count()
    num_items = y_full_data.map(lambda row: row[1]).distinct().count()
    potential_predict = num_users*num_items
    catalog_coverage = prediction_count/float(potential_predict)*100

    return catalog_coverage

#Diversity


#Serendipity


    """
    def calculate_serendipity(y_train, y_test, y_predicted, sqlCtx, rel_filter=1):
    Calculates the serendipity of the recommendations.
    This measure of serendipity in particular is how surprising relevant recommendations are to a user
    serendipity = 1/N sum( max(Pr(s)- Pr(S), 0) * isrel(s)) over all items
    The central portion of this equation is the difference of probability that an item is rated for a user
    and the probability that item would be recommended for any user.
    The first ranked item has a probability 1, and last ranked item is zero.  prob_by_rank(rank, n) calculates this
    Relevance is defined by the items in the hold out set (y_test).
    If an item was rated it is relevant, which WILL miss relevant non-rated items.
    Higher values are better
    Method derived from the Coursera course: Recommender Systems taught by Prof Joseph Konstan (Universitu of Minesota)
    and Prof Michael Ekstrand (Texas State University)
    Args:
        y_train: actual training ratings in the format of an array of [ (userId, itemId, actualRating) ].
        y_test: actual testing ratings to test in the format of an array of [ (userId, itemId, actualRating) ].
        y_predicted: predicted ratings in the format of a RDD of [ (userId, itemId, predictedRating) ].
            It is important that this is not the sorted and cut prediction RDD
        rel_filter: the threshold of item relevance. So for MovieLens this may be 3.5, LastFM 0.
            Ratings/interactions have to be at or above this mark to be considered relevant
    Returns:
        average_overall_serendipity: the average amount of surprise over all users
        average_serendipity: the average user's amount of surprise over their recommended items
    

    full_corpus = y_train.union(y_test).map(lambda (u,i,r): (u,i,float(r)))

    fields = [StructField("user", LongType(),True),StructField("item", LongType(), True),\
          StructField("rating", FloatType(), True) ]
    schema = StructType(fields)
    schema_rate = sqlCtx.createDataFrame(full_corpus, schema)
    schema_rate.registerTempTable("ratings")

    item_ranking = sqlCtx.sql("select item, avg(rating) as avg_rate, row_number() over(ORDER BY avg(rating) desc) as rank \
        from ratings group by item order by avg_rate desc")

    n = item_ranking.count()
    #determine the probability for each item in the corpus
    item_ranking_with_prob = item_ranking.map(lambda (item_id, avg_rate, rank): (item_id, avg_rate, rank, prob_by_rank(rank, n)))

    #format the 'relevant' predictions as a queriable table
    #these are those predictions for which we have ratings above the threshold
    y_test = y_test.filter(lambda (u,i,r): r>=rel_filter).map(lambda (u,i,r): (u,i,float(r)))

    predictionsAndRatings = y_predicted.map(lambda x: ((x[0], x[1]), x[2])) \
      .join(y_test.map(lambda x: ((x[0], x[1]), x[2])))
    temp = predictionsAndRatings.map(lambda (a,b): (a[0], a[1], b[1], b[1]))
    fields = [StructField("user", LongType(),True),StructField("item", LongType(), True),\
          StructField("prediction", FloatType(), True), StructField("actual", FloatType(), True) ]
    schema = StructType(fields)
    schema_preds = sqlCtx.createDataFrame(temp, schema)
    schema_preds.registerTempTable("preds")

    #determine the ranking of predictions by each user
    user_ranking = sqlCtx.sql("select user, item, prediction, row_number() \
        over(Partition by user ORDER BY prediction desc) as rank \
        from preds order by user, prediction desc")
    user_ranking.registerTempTable("user_rankings")

    #find the number of predicted items by user
    user_counts = sqlCtx.sql("select user, count(item) as num_found from preds group by user")
    user_counts.registerTempTable("user_counts")

    #use the number of predicted items and item rank to determine the probability an item is predicted
    user_info = sqlCtx.sql("select r.user, item, prediction, rank, num_found from user_rankings as r, user_counts as c\
        where r.user=c.user")
    user_ranking_with_prob = user_info.map(lambda (user, item, pred, rank, num): \
                                     (user, item, rank, num, prob_by_rank(rank, num)))

    #now combine the two to determine (user, item_prob_diff) by item
    data = user_ranking_with_prob.keyBy(lambda p: p[1])\
        .join(item_ranking_with_prob.keyBy(lambda p:p[0]))\
        .map(lambda (item, (a,b)): (a[0], max(a[4]-b[3],0)))\

    #combine the item_prob_diff by user and average to get the average serendiptiy by user
    sumCount = data.combineByKey(lambda value: (value, 1),
                             lambda x, value: (x[0] + value, x[1] + 1),
                             lambda x, y: (x[0] + y[0], x[1] + y[1]))
    serendipityByUser = sumCount.map(lambda (label, (value_sum, count)): (label, value_sum / float(count)))

    num = float(serendipityByUser.count())
    average_serendipity = serendipityByUser.map(lambda (user, serendipity):serendipity).reduce(add)/num

    #alternatively we could average not by user first, so heavier users will be more influential
    #for now we shall return both
    average_overall_serendipity = data.map (lambda (user, serendipity): serendipity).reduce(add)/float(data.count())
    return (average_overall_serendipity, average_serendipity)"""


#Novelty
"""calculate_novelty(y_train, y_test, y_predicted, sqlCtx)
    
    Novelty measures how new or unknown recommendations are to a user
    An individual item's novelty can be calculated as the log of the popularity of the item
    A user's overal novelty is then the sum of the novelty of all items
    Method derived from 'Auraslist: Introducing Serendipity into Music Recommendation' by Y Zhang, D Seaghdha, D Quercia, and T Jambor
    Args:
        y_train: actual training ratings in the format of an array of [ (userId, itemId, actualRating) ].
        y_test: actual testing ratings to test in the format of an array of [ (userId, itemId, actualRating) ].
            y_train and y_test are necessary to determine the overall item ranking
        y_predicted: predicted ratings in the format of a RDD of [ (userId, itemId, predictedRating) ].
            It is important that this IS the sorted and cut prediction RDD
    Returns:
        avg_overall_novelty: the average amount of novelty over all users
        avg_novelty: the average user's amount of novelty over their recommended items
   

    full_corpus = y_train.union(y_test).map(lambda (u,i,r): (u,i,float(r)))

    fields = [StructField("user", LongType(),True),StructField("item", LongType(), True),\
          StructField("rating", FloatType(), True) ]
    schema = StructType(fields)
    schema_rate = sqlCtx.createDataFrame(full_corpus, schema)
    schema_rate.registerTempTable("ratings")

    item_ranking = sqlCtx.sql("select item, avg(rating) as avg_rate, row_number() over(ORDER BY avg(rating) desc) as rank \
        from ratings group by item order by avg_rate desc")

    n = item_ranking.count()
    item_ranking_with_nov = item_ranking.map(lambda (item_id, avg_rate, rank): (item_id, (avg_rate, rank, log(max(prob_by_rank(rank, n), 1e-100), 2))))

    user_novelty = y_predicted.keyBy(lambda (u, i, p): i).join(item_ranking_with_nov).map(lambda (i,((u_p),(pop))): (u_p[0], pop[2]))\
        .groupBy(lambda (user, pop): user).map(lambda (user, user_item_probs):(np.mean(list(user_item_probs), axis=0)[1])).collect()

    all_novelty = y_predicted.keyBy(lambda (u, i, p): i).join(item_ranking_with_nov).map(lambda (i,((u_p),(pop))): (pop[2])).collect()
    avg_overall_novelty = float(np.mean(all_novelty))

    avg_novelty = float(np.mean(user_novelty))

    return (avg_overall_novelty, avg_novelty) """





