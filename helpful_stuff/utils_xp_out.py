import csv
from collections import namedtuple
import os
from pathlib import Path

# named tuple describing experiment details
XPDescription = namedtuple('XPDescription', ['predictor', 'label', 'nfactors'])
# named tuple describing experiment evaluation results
XPResults = namedtuple('XPResults', ['dataset', 'xpdata', 'rmse', 'mae', 'precision', 'recall'])

ROOT = 'D:/Evaluations/master/'


def make_dirs(xp_name, dataset_name):
    """
    Makes output directories for experiment evaluation.

    :param xp_name: str, name of experiment
    :param dataset_name: str, name of dataset
    :return: str, path to experiment directory
    """
    # make dirs if not exist
    path = ROOT
    if not os.path.isdir(path):
        os.mkdir(path)

    path += dataset_name
    if not os.path.isdir(path):
        os.mkdir(path)

    path += '/' + xp_name
    if not os.path.isdir(path):
        os.mkdir(path)

    return path + '/'


def write_to_csv(xp_results, dataset_name, xp_name):
    """
    Writes RMSE, MAE and recall\precision results into corresponding files.
    :param xp_results: named tuple with evaluation details.
    :param dataset_name: str, name of dataset
    :param xp_name: name of experiment
    :return: none
    """
    xp_dir = make_dirs(xp_name, dataset_name)
    predictor_file = Path('%s%s.csv' % (xp_dir, xp_results.xpdata.label))

    # write headers if file not yet created
    if not predictor_file.exists():
        _write_row(str(predictor_file), ['category', 'predictor', 'nfactors', 'rmse', 'mae'])

    # write header if file not yet created
    recall_precision_file = Path('%skpr_%s_%s.csv' % (xp_dir, xp_results.xpdata.label, xp_results.dataset))
    if not recall_precision_file.exists():
        _write_row(str(recall_precision_file), ['category', 'predictor', 'nfactors', 'k', 'recall', 'precision'])

    # write data
    _write_row(str(predictor_file), [xp_results.dataset, xp_results.xpdata.label, xp_results.xpdata.nfactors,
                                     xp_results.rmse, xp_results.mae])

    for k in xp_results.recall:
        _write_row(str(recall_precision_file), [xp_results.dataset, xp_results.xpdata.label, xp_results.xpdata.nfactors,
                                                k, xp_results.recall[k], xp_results.precision[k]])


def _write_row(file, row):
    with open(file, "a", newline='') as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(row)
