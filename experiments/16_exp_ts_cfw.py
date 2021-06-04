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
from experiments.util import get_ts_dataset

from ece.blackbox import BlackBox



def experiment(bb, bb_keras, X_train, X_test, y_train, y_test, window_size,
                   nbr_test, dataset, black_box, class_values, X_train_flat, X_test_flat, filename_results, path_ae):

    time_start = datetime.datetime.now()
    shape = (1,) + X_train.shape[1:]
    target_proba = 1.0
    tol = 0.01  # want counterfactuals with p(class)>0.99
    target_class = 'other'  # any class other than 7 will do
    max_iter = 1000
    lam_init = 1e-1
    max_lam_steps = 10
    learning_rate_init = 0.1
    feature_range = (X_train.min(), X_train.max())

    predict_fn = lambda x: bb.predict_proba(x)

    # initialize explainer
    exp = CounterFactual(predict_fn, shape=shape, target_proba=target_proba, tol=tol,
                         target_class=target_class, max_iter=max_iter, lam_init=lam_init,
                         max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                         feature_range=feature_range)

    # exp.fit(bb, X_train)  # no fit for watcher

    time_train = (datetime.datetime.now() - time_start).total_seconds()

    index_test_instances = np.random.choice(range(len(X_test)), nbr_test)

    print(datetime.datetime.now(), dataset, black_box, 'cfw', 'ts')

    cf_list_all = list()
    x_eval_list = list()

    for test_id, i in enumerate(index_test_instances):
        time_start_i = datetime.datetime.now()
        print(datetime.datetime.now(), dataset, black_box, 'cfw',
              test_id, len(index_test_instances), '%.2f' % (test_id / len(index_test_instances)))

        x = X_test[i]
        y_val = bb.predict(x.reshape((1,) + X_test[i].shape))[0]

        explanation = exp.explain(x.reshape((1,) + X_test[i].shape))
        if explanation.cf is not None:
            cf_list = explanation.cf['X']
        else:
            cf_list = np.array([])
        time_test = (datetime.datetime.now() - time_start_i).total_seconds()

        cf_list_flat = np.array([cf.flatten() for cf in cf_list])

        for k in [
            1,  # 2, 3, 4,
            5,  # 8,
            10,  # 12, 14,
            15,  # 16, 18, 20
        ]:

            dims = X_train[0].shape[0:1]
            variable_features = np.arange(dims[0])
            x_eval = evaluate_cf_list_img_ts(cf_list, x, bb, y_val, k, variable_features,
                                             X_train[:1000], X_test[:1000], len(variable_features),
                                             X_train_flat[:1000], X_test_flat[:1000])

            x_eval['dataset'] = dataset
            x_eval['black_box'] = black_box
            x_eval['method'] = 'cfw'
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
    dataset = 'gunpoint'
    black_box = 'CNN'
    normalize = 'standard'

    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    print(datetime.datetime.now(), dataset, black_box)

    data = get_ts_dataset(dataset, path_dataset, normalize)
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    window_size = (data['window_sizes'][0],)
    class_values = data['class_values']

    X_train_flat = np.array([x.flatten() for x in X_train])
    X_test_flat = np.array([x.flatten() for x in X_test])

    bb_keras = load_model(path_models + '%s_%s.h5' % (dataset, black_box))
    bb = BlackBox(bb_keras)

    filename_results = path_results + 'ts_cfw.csv'
    experiment(bb, bb_keras, X_train, X_test, y_train, y_test, window_size,
               nbr_test, dataset, black_box, class_values, X_train_flat, X_test_flat, filename_results, path_ae)


if __name__ == "__main__":
    main()

