import csv
from collections import namedtuple, defaultdict
import os
import glob
from pathlib import Path

# general precision@M and recall@M function
def precision_recall_at_k(user_est_true, k=10, threshold=3.5):
    precisions = dict()
    recalls = dict()
    
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls

## for predictions in pandas DataFrame
def precision_recall_at_k_4df(df, k=10, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, true_r, est in df[['reviewerID', 'overall', 'value']].values:
        user_est_true[uid].append((est, true_r))

    return precision_recall_at_k(user_est_true, k, threshold)

## for predictions in surprise DataFrame
def precision_recall_at_k_4ds(predictions, k=10, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    return precision_recall_at_k(user_est_true, k, threshold)


def make_dirs(XP_NAME):
    ## make dirs if not exist
    if not os.path.isdir('./xps/'):
        os.mkdir('./xps/')

    if not os.path.isdir('./xps/' + XP_NAME):
        os.mkdir('./xps/' + XP_NAME)

    XP_DIR = './xps/' + XP_NAME + '/'
    return XP_DIR


## define aux functions
def write_row(file, row):
    with open(file, "a", newline='') as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(row)
    
## this should be used to save xp row
def write_to_csv(xp_row, xp_name):
    XP_DIR = make_dirs(xp_name)
    predictor_file = Path('%s%s.csv' % (XP_DIR, xp_row.xpdata.label))
    # write headers if file not yet created
    if not predictor_file.exists():
        write_row(str(predictor_file), ['category', 'predictor', 'nfactors', 'rmse', 'mae'])
    
    # write header if file not yet created
    recall_precision_file = Path( '%skpr_%s_%s.csv' % (XP_DIR, xp_row.xpdata.label, xp_row.dataset))    
    if not recall_precision_file.exists():
        write_row(str(recall_precision_file), ['category', 'predictor', 'nfactors', 'k', 'recall', 'precision'])
    
    # write data
    write_row(str(predictor_file), [xp_row.dataset, xp_row.xpdata.label, xp_row.xpdata.nfactors,
               xp_row.rmse, xp_row.mae])
    
    for k in xp_row.recall:
        write_row(str(recall_precision_file), [xp_row.dataset, xp_row.xpdata.label, xp_row.xpdata.nfactors, k,
               xp_row.recall[k], xp_row.precision[k]])
    
   
## define named tuples
XPData = namedtuple('XPData', ['predictor', 'label', 'nfactors'])
XPRow = namedtuple('XPRow', ['dataset', 'xpdata', 'rmse', 'mae', 'precision', 'recall'])