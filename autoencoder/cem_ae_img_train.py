import os
import datetime
from sklearn.model_selection import train_test_split

from experiments.config import *
from experiments.util import get_image_dataset

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D
from tensorflow.keras.models import Model


def ae_model_cem(input_shape):
    x_in = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x_in)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(1, (3, 3), activation=None, padding='same')(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(1, (3, 3), activation=None, padding='same')(x)

    autoencoder = Model(x_in, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder


def main():

    dataset = 'fashion_mnist'

    epochs = 4
    batch_size = 128

    path_ae = 'cem_ae_img_%s/' % epochs
    name = 'cem_ae_%s' % dataset

    if not os.path.exists(path_ae):
        os.makedirs(path_ae)

    print(datetime.datetime.now(), dataset, 'ae', 'cem')

    data = get_image_dataset(dataset, path_dataset, categories=None, filter=None,
                             use_rgb=True, flatten=False, expand_dims=True)
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

    X_train_vae, X_val_vae, _, _ = train_test_split(X_train, y_train, test_size=test_size,
                                                    random_state=random_state, stratify=y_train)

    input_shape = X_train.shape[1:]

    ae = ae_model_cem(input_shape)
    ae.summary()
    ae.fit(X_train_vae, X_train_vae, epochs=epochs, batch_size=batch_size, validation_data=(X_val_vae, X_val_vae),
           verbose=1)
    ae.save(path_ae + name + '.h5', save_format='h5')


if __name__ == "__main__":
    main()

