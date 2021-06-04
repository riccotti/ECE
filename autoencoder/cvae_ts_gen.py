import datetime
import numpy as np
import matplotlib.pyplot as plt

from experiments.config import *
from experiments.util import get_ts_dataset

from autoencoder.cvae import ConditionalVariationalAutoencoderTimeSeries


def main():
    dataset = 'ecg5days'

    kernel_size = 4
    strides = 1
    filters = 16
    latent_dim = 4
    num_conv_layers = 2
    hidden_dim = 16
    use_mse = False
    if not use_mse:
        optimizer = 'adam'
    else:
        optimizer = 'rmsprop'

    epochs = 1000

    path_cvae = 'cvae_ts_%s/' % epochs
    name = '%s_cvae' % dataset

    if use_mse:
        name = '%s_cvae_mse' % dataset

    print(datetime.datetime.now(), dataset, 'cvae')

    data = get_ts_dataset(dataset, path_dataset, normalize='standard')
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

    input_shape = X_train.shape[1:]
    n_classes = len(data['class_values'])

    ae = ConditionalVariationalAutoencoderTimeSeries(input_shape, n_classes, latent_dim=latent_dim,
                                                     num_conv_layers=num_conv_layers,
                                                     filters=filters, kernel_size=kernel_size, strides=strides,
                                                     hidden_dim=hidden_dim, use_mse=use_mse, optimizer=optimizer,
                                                     store_intermediate=False, save_graph=False,
                                                     path=path_cvae, name=name, verbose=1)
    ae.load_model()

    r, c = 5, 5
    for class_val in range(n_classes):

        X = ae.generate(r * c, class_val)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        xaxis = np.arange(input_shape[0])
        for i in range(r):
            for j in range(c):
                axs[i, j].plot(xaxis, X[cnt].reshape((input_shape[0], input_shape[1])))
                #axs[i, j].axis('off')
                cnt += 1
        fig.savefig('%s%s_%s.png' % (path_cvae, name, class_val))
        plt.close()


if __name__ == "__main__":
    main()

