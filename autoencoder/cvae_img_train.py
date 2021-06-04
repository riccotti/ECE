import os
import datetime

from sklearn.model_selection import train_test_split

from experiments.config import *
from experiments.util import get_image_dataset

from autoencoder.cvae import ConditionalVariationalAutoencoderImage


def main():

    dataset = 'fashion_mnist'

    kernel_size = (3, 3)
    strides = (1, 1)
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
    batch_size = 128

    path_cvae = 'cvae_img_%s/' % epochs
    name = '%s_cvae' % dataset

    if use_mse:
        name = '%s_cvae_mse' % dataset

    print(datetime.datetime.now(), dataset, 'cvae')

    import tensorflow as tf
    print(tf.config.experimental.list_physical_devices('GPU'))

    if not os.path.exists(path_cvae):
        os.makedirs(path_cvae)

    data = get_image_dataset(dataset, path_dataset, categories=None, filter=None,
                             use_rgb=True, flatten=False, expand_dims=True)
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

    input_shape = X_train.shape[1:]
    n_classes = len(data['class_values'])

    X_train_ae, X_val_ae, y_train_ae, y_val_ae = train_test_split(X_train, y_train, test_size=test_size,
                                                                  random_state=random_state, stratify=y_train)

    ae = ConditionalVariationalAutoencoderImage(input_shape, n_classes, latent_dim=latent_dim,
                                                num_conv_layers=num_conv_layers, filters=filters,
                                                kernel_size=kernel_size, strides=strides, hidden_dim=hidden_dim,
                                                use_mse=use_mse, optimizer=optimizer, store_intermediate=False,
                                                save_graph=False, path=path_cvae, name=name, verbose=1)

    ae.fit([X_train_ae, y_train_ae], Xy_val=[X_val_ae, y_val_ae], epochs=epochs, batch_size=batch_size)


if __name__ == "__main__":
    main()

