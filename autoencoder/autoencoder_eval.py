import os
import pickle
import numpy as np
import pandas as pd

from keras.models import load_model
from sklearn.metrics import mean_squared_error

from ece.blackbox import BlackBox

from experiments.config import *
from experiments.util import get_ts_dataset, get_image_dataset

from autoencoder.vae import VariationalAutoencoderImage, VariationalAutoencoderTimeSeries
from autoencoder.cvae import ConditionalVariationalAutoencoderImage, ConditionalVariationalAutoencoderTimeSeries

datasets = {
    'img': ['mnist', 'fashion_mnist'],
    'ts': [
        'gunpoint', 'italypower', 'ecg200', 'ecg5days',
        # 'electricdevices', 'arrowhead', 'ecg5000'
    ],

}


def avg_sum_abs_diff(A, B):
    return np.mean(np.sum(np.abs(A - B), axis=1))


def avg_sum_squared_diff(A, B):
    return np.mean(np.sum((A - B) ** 2, axis=1))


def ae_error_eval(A, B):
    if len(A.shape) >= 3:
        A = A.reshape(A.shape[0], A.shape[1] * A.shape[2])
        B = B.reshape(B.shape[0], B.shape[1] * B.shape[2])

    mse = mean_squared_error(A, B)
    rmse = mean_squared_error(A, B, squared=False)
    nrmse = rmse / np.sqrt(np.mean(A ** 2))
    asad = avg_sum_abs_diff(A, B)
    assd = avg_sum_squared_diff(A, B)

    res = {
        'mse': mse,
        'rmse': rmse,
        'nrmse': nrmse,
        'asad': asad,
        'assd': assd
    }
    return res


def main():

    epochs = 1000
    filters = 16
    latent_dim = 4
    num_conv_layers = 2
    hidden_dim = 16
    filename_results = path_results + 'autoencoders_performance2.csv'

    for data_type in [
        'img',
        'ts'
    ]:

        for dataset in datasets[data_type]:

            if data_type == 'img':
                kernel_size = (3, 3)
                strides = (1, 1)
                data = get_image_dataset(dataset, path_dataset, categories=None, filter=None,
                                         use_rgb=True, flatten=False, expand_dims=True)
            else:
                kernel_size = 4
                strides = 1
                data = get_ts_dataset(dataset, path_dataset, normalize='standard')

            for use_mse in [False, True]:

                if not use_mse:
                    optimizer = 'adam'
                else:
                    optimizer = 'rmsprop'

                for ae_type in ['vae', 'cvae']:

                    path_ae = '%s_%s_%s/' % (ae_type, data_type, epochs)
                    name = '%s_%s' % (dataset, ae_type)

                    if use_mse:
                        name = '%s_%s_mse' % (dataset, ae_type)

                    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

                    input_shape = X_train.shape[1:]
                    n_classes = len(data['class_values'])

                    print(data_type, dataset, ae_type, optimizer, use_mse)

                    if data_type == 'img':
                        if ae_type == 'vae':
                            ae = VariationalAutoencoderImage(input_shape, latent_dim=latent_dim,
                                                             num_conv_layers=num_conv_layers,
                                                             filters=filters, kernel_size=kernel_size, strides=strides,
                                                             hidden_dim=hidden_dim,
                                                             use_mse=use_mse, optimizer=optimizer,
                                                             store_intermediate=False, save_graph=False,
                                                             path=path_ae, name=name, verbose=0)
                        else:
                            ae = ConditionalVariationalAutoencoderImage(input_shape, n_classes, latent_dim=latent_dim,
                                                                        num_conv_layers=num_conv_layers,
                                                                        filters=filters, kernel_size=kernel_size,
                                                                        strides=strides,
                                                                        hidden_dim=hidden_dim,
                                                                        use_mse=use_mse, optimizer=optimizer,
                                                                        store_intermediate=False,
                                                                        save_graph=False,
                                                                        path=path_ae, name=name, verbose=0)

                    else:
                        if ae_type == 'vae':
                            ae = VariationalAutoencoderTimeSeries(input_shape, latent_dim=latent_dim,
                                                                  num_conv_layers=num_conv_layers,
                                                                  filters=filters, kernel_size=kernel_size,
                                                                  strides=strides,
                                                                  hidden_dim=hidden_dim, use_mse=use_mse,
                                                                  optimizer=optimizer,
                                                                  store_intermediate=False, save_graph=False,
                                                                  path=path_ae, name=name, verbose=0)
                        else:
                            ae = ConditionalVariationalAutoencoderTimeSeries(input_shape, n_classes,
                                                                             latent_dim=latent_dim,
                                                                             num_conv_layers=num_conv_layers,
                                                                             filters=filters, kernel_size=kernel_size,
                                                                             strides=strides,
                                                                             hidden_dim=hidden_dim, use_mse=use_mse,
                                                                             optimizer=optimizer,
                                                                             store_intermediate=False, save_graph=False,
                                                                             path=path_ae, name=name, verbose=0)
                    ae.load_model()

                    if ae_type == 'vae':
                        X_test_rec = ae.decode(ae.encode(X_test))
                    else:
                        y_test_oh = ae.oh_encoder.transform(y_test.reshape(-1, 1)).toarray()
                        X_lat = ae.encode([X_test, y_test_oh])[0]
                        X_test_rec = ae.decode([X_lat, y_test_oh])

                    eval_dict = {
                        'ae_type': ae_type,
                        'data_type': data_type,
                        'dataset': dataset,
                        'optimizer': optimizer,
                        'use_mse': use_mse,
                    }

                    error_eval = ae_error_eval(X_test, X_test_rec)

                    eval_dict.update(error_eval)

                    if data_type == 'img':
                        black_box = 'DNN'
                        bb = load_model(path_models + '%s_%s.h5' % (dataset, black_box))
                    else:
                        black_box = 'CNN'
                        bb = load_model(path_models + '%s_%s.h5' % (dataset, black_box))

                    bb = BlackBox(bb)

                    y_pred_o = bb.predict(X_test)
                    y_pred_r = bb.predict(X_test_rec)
                    # print(y_pred_o.shape, y_pred_r.shape)

                    # delta_pred = np.sum(np.abs(y_pred_o - y_pred_r))/len(y_pred_o)
                    delta_pred = np.sum([1 if yo == yr else 0 for yo, yr in zip(y_pred_o, y_pred_r)]) / len(y_pred_o)

                    y_pred_proba_o = bb.predict(X_test)
                    y_pred_proba_r = bb.predict(X_test_rec)
                    # print(y_pred_proba_o.shape, y_pred_proba_r.shape)

                    delta_pred_proba = np.sum(np.abs(y_pred_proba_o - y_pred_proba_r))/len(y_pred_o)

                    eval_dict['%s_delta_pred' % black_box] = delta_pred
                    eval_dict['%s_delta_pred_proba' % black_box] = delta_pred_proba

                    if ae_type == 'cvae':
                        delta_cond_list = list()
                        for class_val in range(n_classes):
                            X_gen = ae.generate(1000, class_val)
                            y_pred_o = np.array([class_val] * 1000)
                            y_pred_r = bb.predict(X_gen)

                            # print(y_pred_o.shape, y_pred_r.shape)

                            delta_cond = np.sum([1 if yo == yr else 0 for yo, yr in zip(y_pred_o, y_pred_r)]) / len(y_pred_o)
                            delta_cond_list.append(delta_cond)

                        eval_dict['%s_delta_cond' % black_box] = np.mean(delta_cond_list)
                    else:
                        eval_dict['%s_delta_cond' % black_box] = 0.0

                    df_cf = pd.DataFrame(data=[eval_dict], columns=list(eval_dict.keys()))

                    if not os.path.isfile(filename_results):
                        df_cf.to_csv(filename_results, index=False)
                    else:
                        df_cf.to_csv(filename_results, mode='a', index=False, header=False)


if __name__ == "__main__":
    main()
