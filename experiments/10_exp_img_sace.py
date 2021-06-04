import sys

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
from ece.neighbor_ece import NeighborECE
from ece.cluster_ece import KMeansECE
from ece.tree_ece import TreeECE
from ece.ensemble_ece import EnsembleECE
from ece.distr_ece import DistrECE
from ece.casebased_ece import CaseBasedECE

from cf_eval.metrics import *

from experiments.config import *
from experiments.util import get_image_dataset

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from ece.blackbox import BlackBox

from ece.pooler import ImageIdentity
from ece.pooler import ImagePooler
from ece.pooler import ImageAutoencoder

from sklearn.preprocessing import MinMaxScaler


def experiment(bb, X_train, X_test, y_train, y_test, pooler_type, window_size,
                   nbr_test, dataset, black_box, class_values, X_train_flat, X_test_flat, filename_results, path_ae):

    covertype = 'naive'
    n_estimators = 10

    if pooler_type == 'identity':
        dims = X_train[0].shape[0:2]
        pooler = ImageIdentity(dims)
        nbr_features = dims[0] * dims[1]
    elif pooler_type == 'kernel':
        dims = X_train[0].shape[0:2]
        window = window_size
        size = tuple((np.array(dims) // np.array(window)))
        pooler = ImagePooler(dims, window)
        nbr_features = size[0] * size[1]
    elif pooler_type == 'vae':
        input_shape = X_train.shape[1:]
        n_classes = len(class_values)
        latent_dim = 4
        ae_type = 'vae'
        path_ae_local = path_ae + '/%s_img_1000/' % ae_type
        name = '%s_%s' % (dataset, ae_type)
        pooler = ImageAutoencoder(input_shape, latent_dim, n_classes, ae_type, path_ae_local, name)
        nbr_features = latent_dim
    elif pooler_type == 'cvae':
        input_shape = X_train.shape[1:]
        n_classes = len(class_values)
        latent_dim = 4
        ae_type = 'cvae'
        path_ae_local = path_ae + '/%s_img_1000/' % ae_type
        name = '%s_%s' % (dataset, ae_type)
        pooler = ImageAutoencoder(input_shape, latent_dim, n_classes, ae_type, path_ae_local, name)
        nbr_features = latent_dim
    else:
        raise ValueError('Unkown pooler type %s' % pooler_type)

    max_samples_count = len(X_train) * 0.2
    if max_samples_count > 1000:
        max_samples = 1000 / len(X_train)
    else:
        max_samples = 0.2

    time_start = datetime.datetime.now()
    exp = EnsembleECE(variable_features=None, weights=None, metric='euclidean',
                      feature_names=None, continuous_features=np.arange(nbr_features),
                      categorical_features_lists=None, normalize=False, pooler=pooler,
                      n_estimators=n_estimators, max_samples=max_samples, max_features='auto',
                      base_estimator='pippo',
                      estimators_params={
                           'dist': {'n_attempts': 5,
                                    'n_batch': 100,
                                    'stopping_eps': 0.1,
                                    'kind': 'gaussian_matched',
                                    'tol': 0.01},
                           'tree': {'use_instance_weights': False,
                                    'kernel_width': None,
                                    'min_samples_leaf': 0.01,
                                    'max_depth': None,
                                    'closest_in_leaf': True},
                           'feat': {'nbr_intervals': 10,
                                    'nbr_features_to_test': 1,
                                    'tol': 0.01},
                       },
                      n_jobs=-1, verbose=0)

    exp.fit(bb, X_train)

    time_train = (datetime.datetime.now() - time_start).total_seconds()

    index_test_instances = np.random.choice(range(len(X_test)), nbr_test)

    print(datetime.datetime.now(), dataset, black_box, 'ens-h', 'img', pooler_type)

    cf_list_all = list()
    x_eval_list = list()

    for test_id, i in enumerate(index_test_instances):
        print(datetime.datetime.now(), dataset, black_box, 'ens-h', covertype, pooler_type,
              test_id, len(index_test_instances), '%.2f' % (test_id / len(index_test_instances)))

        x = X_test[i]
        y_val = bb.predict(x.reshape((1,) + x.shape))[0]

        for k in [
            1,  # 2, 3, 4,
            5,  # 8,
            10,  # 12, 14,
            15,  # 16, 18, 20
        ]:

            time_start_i = datetime.datetime.now()

            cf_list = exp.get_counterfactuals(x, k=k, covertype=covertype,
                                              lambda_par=1.0, cf_rate=0.5, cf_rate_incr=0.1)

            time_test = (datetime.datetime.now() - time_start_i).total_seconds()

            cf_list_flat = np.array([cf.flatten() for cf in cf_list])

            dims = X_train[0].shape[0:2]
            variable_features = np.arange(dims[0] * dims[1])
            x_eval = evaluate_cf_list_img_ts(cf_list, x, bb, y_val, k, variable_features,
                                             X_train[:1000], X_test[:1000], len(variable_features),
                                             X_train_flat[:1000], X_test_flat[:1000])

            x_eval['dataset'] = dataset
            x_eval['black_box'] = black_box
            x_eval['method'] = 'ens-h'
            x_eval['idx'] = i
            x_eval['k'] = k
            x_eval['time_train'] = time_train
            x_eval['time_test'] = time_test
            x_eval['runtime'] = time_train + time_test
            x_eval['metric'] = 'euclidean'
            x_eval['covertype'] = covertype
            x_eval['pooler'] = pooler_type

            x_eval_list.append(x_eval)
            if len(cf_list):
                cf_list_all.append(cf_list_flat[0])

        if len(cf_list_all) > 1:
            instability_si = np.mean(squareform(pdist(np.array(cf_list_all), metric='euclidean')))
        else:
            instability_si = 0.0

        for x_eval in x_eval_list:
            x_eval['instability_si'] = instability_si

        df = pd.DataFrame(data=x_eval_list)
        df = df[columns_img_ts]

        if not os.path.isfile(filename_results):
            df.to_csv(filename_results, index=False)
        else:
            df.to_csv(filename_results, mode='a', index=False, header=False)


def main():

    nbr_test = 1
    dataset = 'mnist'
    black_box = 'DNN'

    print(datetime.datetime.now(), dataset, black_box)

    data = get_image_dataset(dataset, path_dataset, categories=None, filter=None,
                             use_rgb=True, flatten=False, expand_dims=True)
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    window_size = data['window_sizes'][0]
    class_values = data['class_values']

    X_train_flat = np.array([x.flatten() for x in X_train])
    X_test_flat = np.array([x.flatten() for x in X_test])

    bb = load_model(path_models + '%s_%s.h5' % (dataset, black_box))
    bb = BlackBox(bb)

    for pooler_type in [
        'identity',
        'kernel',
        'vae',
        'cvae'
    ]:
        filename_results = path_results + 'img_ens_%s.csv' % pooler_type
        experiment(bb, X_train, X_test, y_train, y_test, pooler_type, window_size,
                   nbr_test, dataset, black_box, class_values, X_train_flat, X_test_flat, filename_results, path_ae)


if __name__ == "__main__":
    main()

