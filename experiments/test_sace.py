import os
import pickle
import datetime
import numpy as np
import pandas as pd

from keras.models import load_model
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform

from ece.blackbox import BlackBox
from ece.random_ece import RandomECE
from ece.feature_ece import FeatureECE
from ece.brute_force_sace import BruteForceECE
from ece.neighbor_ece import NeighborECE
from ece.cluster_ece import KMeansECE
from ece.tree_ece import TreeECE
from ece.ensemble_ece import EnsembleECE
from ece.distr_ece import DistrECE
from ece.casebased_ece import CaseBasedECE

from cf_eval.metrics import *

from experiments.config import *
from experiments.util import get_tabular_dataset


def main():

    nbr_test = 100
    dataset = 'compas'
    black_box = 'RF'
    normalize = 'standard'
    # normalize_str = '' if normalize is None else '_%s' % normalize
    # dataset = sys.argv[1]
    # black_box = sys.argv[2]
    # cfe = sys.argv[3]
    # nbr_test = 100 if len(sys.argv) < 5 else int(sys.argv[4])
    # known_train = True if len(sys.argv) < 6 else sys.argv[5]
    # search_diversity = True

    np.random.seed(random_state)

    if dataset not in dataset_list:
        print('unknown dataset %s' % dataset)
        return -1

    if black_box not in blackbox_list:
        print('unknown black box %s' % black_box)
        return -1

    # if cfe not in cfe_list:
    #     print('unknown counterfactual explainer %s' % cfe)
    #     return -1

    print(datetime.datetime.now(), dataset, black_box)

    data = get_tabular_dataset(dataset, path_dataset, normalize=normalize, test_size=test_size,
                               random_state=random_state, encode=None if black_box == 'LGBM' else 'onehot',
                               remove_missing=False, return_original=True)
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    class_values = data['class_values']
    if dataset == 'titanic':
        class_values = ['Not Survived', 'Survived']
    features_names = data['feature_names']
    variable_features = data['variable_features']
    variable_features_names = data['variable_features_names']
    continuous_features = data['continuous_features']
    continuous_features_all = data['continuous_features_all']
    categorical_features_lists = data['categorical_features_lists']
    categorical_features_lists_all = data['categorical_features_lists_all']
    categorical_features_all = data['categorical_features_all']
    continuous_features_names = data['continuous_features_names']
    categorical_features_names = data['categorical_features_names']
    scaler = data['scaler']
    nbr_features = data['n_cols']
    ratio_cont = data['n_cont_cols'] / nbr_features
    df = data['df']
    # print(df.isnull().sum())
    # print(X_train.shape)
    #print(np.isnan(X_train).any(axis=1))
    #print(np.where(np.isnan(X_test).any(axis=1))[0][:10])
    # for i, x in enumerate(X_test):
    #     if np.isnan(x[0]):
    #         print(i)
    #         break

    variable_cont_features_names = [c for c in variable_features_names if c in continuous_features_names]
    variable_cate_features_names = list(
        set([c.split('=')[0] for c in variable_features_names if c.split('=')[0] in categorical_features_names]))

    if black_box in ['DT', 'RF', 'SVM', 'NN', 'LGBM']:
        bb = pickle.load(open(path_models + '%s_%s.pickle' % (dataset, black_box), 'rb'))
    elif black_box in ['DNN']:
        bb = load_model(path_models + '%s_%s.h5' % (dataset, black_box))
    else:
        print('unknown black box %s' % black_box)
        raise Exception

    bb = BlackBox(bb)

    metric = ('euclidean', 'jaccard')

    # exp = DistrECE(variable_features, weights=None, metric=metric,
    #                 feature_names=None, continuous_features=continuous_features,
    #                 categorical_features_lists=categorical_features_lists, normalize=False,
    #                 n_attempts=10, n_batch=1000, stopping_eps=0.01, kind='gaussian_matched')
    exp = FeatureECE(variable_features, weights=None, metric=metric, feature_names=None,
                     continuous_features=continuous_features,
                     categorical_features_lists=categorical_features_lists, normalize=False,
                     pooler=None, tol=0.01,
                     nbr_intervals=5, nbr_features_to_test=2)
    exp.fit(bb, X_train)

    idx = 1   # 1565 cont, 5 cate
    k = 5

    x = X_test[idx]
    y_val = bb.predict(x.reshape(1, -1))[0]

    print(x[variable_features])
    print(y_val)

    print('-----')
    cf_list = exp.get_counterfactuals(x, k=k)

    for c in cf_list:
        print(c[variable_features])

    print('-----')
    print('-----')
    print('-----')

    exp = BruteForceECE(variable_features, weights=None, metric=metric, feature_names=None,
                        continuous_features=continuous_features,
                        categorical_features_lists=categorical_features_lists, normalize=False,
                        pooler=None, tol=0.01,
                        nbr_intervals=5, nbr_features_to_test=2)
    exp.fit(bb, X_train)

    x = X_test[idx]
    y_val = bb.predict(x.reshape(1, -1))[0]

    print(x[variable_features])
    print(y_val)

    print('-----')
    cf_list = exp.get_counterfactuals(x, k=k)

    for c in cf_list:
        print(c[variable_features])



if __name__ == "__main__":
    main()
