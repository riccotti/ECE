import datetime
import matplotlib.pyplot as plt

from experiments.config import *
from experiments.util import get_image_dataset

from autoencoder.vae import VariationalAutoencoderImage, VariationalAutoencoderImageSimple


def main():

    dataset = 'mnist'

    kernel_size = (3, 3)
    strides = (1, 1)
    filters = 16
    latent_dim = 4
    num_conv_layers = 2
    hidden_dim = 16
    use_mse = False
    if not use_mse:
        optimizer = 'adam'
    else:
        optimizer = 'rmsprop'

    epochs = 4

    path_vae = 'vae_img_%s/' % epochs
    name = '%s_vae' % dataset

    if use_mse:
        name = '%s_vae_mse' % dataset

    print(datetime.datetime.now(), dataset, 'vae')

    data = get_image_dataset(dataset, path_dataset, categories=None, filter=None,
                             use_rgb=True, flatten=False, expand_dims=True)
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

    input_shape = X_train.shape[1:]

    # ae = VariationalAutoencoderImage(input_shape, latent_dim=latent_dim, num_conv_layers=num_conv_layers,
    #                                  filters=filters, kernel_size=kernel_size, strides=strides, hidden_dim=hidden_dim,
    #                                  use_mse=use_mse, optimizer=optimizer, store_intermediate=False, save_graph=False,
    #                                  path=path_vae, name=name, verbose=0)

    use_mse = True
    optimizer = 'adam'
    ae = VariationalAutoencoderImageSimple(input_shape, latent_dim=latent_dim, num_conv_layers=num_conv_layers,
                                           filters=filters, kernel_size=kernel_size, strides=strides,
                                           hidden_dim=hidden_dim, use_mse=use_mse, optimizer=optimizer,
                                           store_intermediate=False, save_graph=False, path=path_vae, name=name,
                                           verbose=1)

    ae.load_model()

    r, c = 5, 5

    X = ae.generate(r * c)
    X = X * 255.

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(X[cnt].reshape((input_shape[0], input_shape[1])), cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig('%s%s.png' % (path_vae, name))
    plt.close()

    X_rec = ae.vae.predict(X_test[:5])
    print(X_rec.shape)
    r, c = 2, 5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            img = X_test[cnt] if i == 0 else X_rec[cnt]
            axs[i, j].imshow(img.reshape((input_shape[0], input_shape[1])), cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
        cnt = 0
    fig.savefig('%s%s_or_rec.png' % (path_vae, name))
    plt.close()


if __name__ == "__main__":
    main()

