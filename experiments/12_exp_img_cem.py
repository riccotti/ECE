import sys

import os
import pickle
import datetime
import numpy as np
import pandas as pd

from keras.models import load_model
from scipy.spatial.distance import squareform

from ece.blackbox import BlackBox
from alibi.explainers import CounterFactual

from cf_eval.metrics import *

from experiments.config import *
from experiments.util import get_image_dataset

from ece.blackbox import BlackBox

from alibi.explainers import CEM


def experiment(bb, bb_keras, X_train, X_test, y_train, y_test, window_size,
                   nbr_test, dataset, black_box, class_values, X_train_flat, X_test_flat, filename_results, path_ae):

    time_start = datetime.datetime.now()
    mode = 'PN'  # 'PN' (pertinent negative) or 'PP' (pertinent positive)
    shape = (1,) + X_train.shape[1:]  # instance shape
    kappa = 0.  # minimum difference needed between the prediction probability for the perturbed instance on the
    # class predicted by the original instance and the max probability on the other classes
    # in order for the first loss term to be minimized
    beta = .1  # weight of the L1 loss term
    gamma = 100  # weight of the optional auto-encoder loss term
    c_init = 1.  # initial weight c of the loss term encouraging to predict a different class (PN) or
    # the same class (PP) for the perturbed instance compared to the original instance to be explained
    c_steps = 10  # nb of updates for c
    max_iterations = 1000  # nb of iterations per value of c
    feature_range = (X_train.min(), X_train.max())  # feature range for the perturbed instance
    clip = (-1000., 1000.)  # gradient clipping
    lr = 1e-2  # initial learning rate
    no_info_val = -1.  # a value, float or feature-wise, which can be seen as containing no info to make a prediction
    # perturbations towards this value means removing features, and away means adding features
    # for our MNIST images, the background (-0.5) is the least informative,
    # so positive/negative perturbations imply adding/removing features

    predict_fn = lambda x: bb.predict_proba(x)

    ae = load_model(path_ae + 'cem_ae_%s.h5' % dataset)

    # initialize CEM explainer and explain instance
    exp = CEM(predict_fn, mode, shape, kappa=kappa, beta=beta, feature_range=feature_range,
              gamma=gamma, ae_model=ae, max_iterations=max_iterations,
              c_init=c_init, c_steps=c_steps, learning_rate_init=lr, clip=clip, no_info_val=no_info_val)

    exp.fit(X_train, no_info_type='median')  # we need to define what feature values contain the least

    time_train = (datetime.datetime.now() - time_start).total_seconds()

    index_test_instances = np.random.choice(range(len(X_test)), nbr_test)

    print(datetime.datetime.now(), dataset, black_box, 'cem', 'img')

    cf_list_all = list()
    x_eval_list = list()

    for test_id, i in enumerate(index_test_instances):
        time_start_i = datetime.datetime.now()
        print(datetime.datetime.now(), dataset, black_box, 'cem',
              test_id, len(index_test_instances), '%.2f' % (test_id / len(index_test_instances)))

        x = X_test[i]
        y_val = bb.predict(x.reshape((1,) + X_test[i].shape))[0]

        explanation = exp.explain(x.reshape((1,) + X_test[i].shape))
        cf_list = explanation.PN
        if cf_list is None:
            cf_list = np.array([])
        time_test = (datetime.datetime.now() - time_start_i).total_seconds()

        cf_list_flat = np.array([cf.flatten() for cf in cf_list])

        for k in [
            1,  # 2, 3, 4,
            5,  # 8,
            10,  # 12, 14,
            15,  # 16, 18, 20
        ]:

            dims = X_train[0].shape[0:2]
            variable_features = np.arange(dims[0] * dims[1])
            x_eval = evaluate_cf_list_img_ts(cf_list, x, bb, y_val, k, variable_features,
                                             X_train[:1000], X_test[:1000], len(variable_features),
                                             X_train_flat[:1000], X_test_flat[:1000])

            x_eval['dataset'] = dataset
            x_eval['black_box'] = black_box
            x_eval['method'] = 'cem'
            x_eval['idx'] = i
            x_eval['k'] = k
            x_eval['time_train'] = time_train
            x_eval['time_test'] = time_test
            x_eval['runtime'] = time_train + time_test
            x_eval['metric'] = 'euclidean'
            x_eval['covertype'] = np.nan
            x_eval['pooler'] = np.nan

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

    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    print(datetime.datetime.now(), dataset, black_box)

    data = get_image_dataset(dataset, path_dataset, categories=None, filter=None,
                             use_rgb=True, flatten=False, expand_dims=True)
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    window_size = data['window_sizes'][0]
    class_values = data['class_values']

    X_train_flat = np.array([x.flatten() for x in X_train])
    X_test_flat = np.array([x.flatten() for x in X_test])

    bb_keras = load_model(path_models + '%s_%s.h5' % (dataset, black_box))
    bb = BlackBox(bb_keras)

    path_ae_local = path_ae + 'cem_ae_img_4/'

    filename_results = path_results + 'img_cem.csv'
    experiment(bb, bb_keras, X_train, X_test, y_train, y_test, window_size,
               nbr_test, dataset, black_box, class_values, X_train_flat, X_test_flat, filename_results,
               path_ae_local)


if __name__ == "__main__":
    main()

