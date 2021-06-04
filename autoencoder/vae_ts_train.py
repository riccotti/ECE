import os
import datetime
from sklearn.model_selection import train_test_split

from experiments.config import *
from experiments.util import get_ts_dataset

from autoencoder.vae import VariationalAutoencoderTimeSeries


def main():

    dataset = 'electricdevices'

    kernel_size = 4
    strides = 1
    filters = 16
    latent_dim = 4
    num_conv_layers = 2
    hidden_dim = 16
    use_mse = True
    if not use_mse:
        optimizer = 'adam'
    else:
        optimizer = 'rmsprop'

    epochs = 1000
    batch_size = 16

    path_vae = 'vae_ts_%s/' % epochs
    name = '%s_vae' % dataset

    if use_mse:
        name = '%s_vae_mse' % dataset

    if not os.path.exists(path_vae):
        os.makedirs(path_vae)

    print(datetime.datetime.now(), dataset, 'vae')

    data = get_ts_dataset(dataset, path_dataset, normalize='standard')
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

    input_shape = X_train.shape[1:]

    X_train_vae, X_val_vae, _, _ = train_test_split(X_train, y_train, test_size=test_size,
                                                    random_state=random_state, stratify=y_train)

    ae = VariationalAutoencoderTimeSeries(input_shape, latent_dim=latent_dim, num_conv_layers=num_conv_layers,
                                          filters=filters, kernel_size=kernel_size, strides=strides,
                                          hidden_dim=hidden_dim, use_mse=use_mse, optimizer=optimizer,
                                          store_intermediate=False, save_graph=False,
                                          path=path_vae, name=name, verbose=1)

    ae.fit(X_train_vae, X_val=X_val_vae, epochs=epochs, batch_size=batch_size)


if __name__ == "__main__":
    main()

