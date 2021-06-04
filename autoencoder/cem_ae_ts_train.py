import os
import datetime
from sklearn.model_selection import train_test_split

from experiments.config import *
from experiments.util import get_ts_dataset

from tensorflow.keras.layers import Conv1D, Input
from tensorflow.keras.models import Model


def ae_model_cem(input_shape):
    x_in = Input(shape=input_shape)
    x = Conv1D(16, 4, activation='relu', padding='same')(x_in)
    x = Conv1D(16, 4, activation='relu', padding='same')(x)
    # x = MaxPooling1D(8, padding='same')(x)
    encoded = Conv1D(1, 4, activation=None, padding='same')(x)

    x = Conv1D(16, 4, activation='relu', padding='same')(encoded)
    # x = UpSampling1D(8)(x)
    x = Conv1D(16, 4, activation='relu', padding='same')(x)
    decoded = Conv1D(1, 4, activation=None, padding='same')(x)

    autoencoder = Model(x_in, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder


def main():

    dataset = 'gunpoint'

    epochs = 20
    batch_size = 128

    path_ae = 'cem_ae_ts_%s/' % epochs
    name = 'cem_ae_%s' % dataset

    if not os.path.exists(path_ae):
        os.makedirs(path_ae)

    print(datetime.datetime.now(), dataset, 'ae', 'cem')

    data = get_ts_dataset(dataset, path_dataset, normalize='standard')
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

    input_shape = X_train.shape[1:]

    X_train_vae, X_val_vae, _, _ = train_test_split(X_train, y_train, test_size=test_size,
                                                    random_state=random_state, stratify=y_train)

    ae = ae_model_cem(input_shape)
    ae.summary()
    ae.fit(X_train_vae, X_train_vae, epochs=epochs, batch_size=batch_size, validation_data=(X_val_vae, X_val_vae),
           verbose=1)
    ae.save(path_ae + name + '.h5', save_format='h5')


if __name__ == "__main__":
    main()

