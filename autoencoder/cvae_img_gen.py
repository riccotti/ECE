import datetime
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

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

    path_vae = 'cvae_img_%s/' % epochs
    name = '%s_cvae' % dataset

    if use_mse:
        name = '%s_cvae_mse' % dataset

    print(datetime.datetime.now(), dataset, 'vae')

    data = get_image_dataset(dataset, path_dataset, categories=None, filter=None,
                             use_rgb=True, flatten=False, expand_dims=True)
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

    encoder = OneHotEncoder()
    y_train1h = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()

    input_shape = X_train.shape[1:]
    n_classes = y_train1h.shape[1]

    ae = ConditionalVariationalAutoencoderImage(input_shape, n_classes, latent_dim=latent_dim,
                                                num_conv_layers=num_conv_layers,
                                                filters=filters, kernel_size=kernel_size, strides=strides,
                                                hidden_dim=hidden_dim,
                                                use_mse=use_mse, optimizer=optimizer, store_intermediate=False,
                                                save_graph=False,
                                                path=path_vae, name=name, verbose=True)

    ae.load_model()

    r, c = 5, 5
    for class_val in range(n_classes):

        X = ae.generate(r * c, class_val)
        X = X * 255.

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(X[cnt].reshape((input_shape[0], input_shape[1])), cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig('%s%s_%s.png' % (path_vae, name, class_val))
        plt.close()


if __name__ == "__main__":
    main()

