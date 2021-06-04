import os
import datetime
from sklearn.model_selection import train_test_split

from experiments.config import *
from experiments.util import get_ts_dataset

from tensorflow.keras.layers import Conv1D, Input, Flatten, UpSampling1D
from tensorflow.keras.models import Model

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.layers import RepeatVector, TimeDistributed


def ae_model_cegp(input_shape):
    # encoder
    x_in = Input(shape=input_shape)
    x = Conv1D(16, 4, activation='relu', padding='same')(x_in)
    x = Conv1D(16, 4, activation='relu', padding='same')(x)
    # x = MaxPooling1D(8, padding='same')(x)
    # encoded = Conv1D(1, 4, activation=None, padding='same')(x)
    x = Flatten()(x)
    encoded = Dense(8)(x)
    encoder = Model(x_in, encoded)
    # encoder.summary()

    # decoder
    dec_in = Input(shape=(8, 1))
    x = Dense(16 * input_shape[0])(dec_in)
    x = Reshape((input_shape[0], 128))(x)
    x = Conv1D(16, 4, activation='relu', padding='same')(x)
    # x = UpSampling1D(16 * input_shape[0])(x)
    x = Conv1D(16, 4, activation='relu', padding='same')(x)
    decoded = Conv1D(1, 1, activation=None, padding='same')(x)
    decoder = Model(dec_in, decoded)
    # decoder.summary()

    # autoencoder = encoder + decoder
    x_out = decoder(encoder(x_in))
    autoencoder = Model(x_in, x_out)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder, decoder


def main():

    dataset = 'ecg200'

    epochs = 20
    batch_size = 128

    path_ae = 'cegp_ae_ts_%s/' % epochs
    name = 'cegp_ae_%s' % dataset

    if not os.path.exists(path_ae):
        os.makedirs(path_ae)

    print(datetime.datetime.now(), dataset, 'ae', 'cegp')

    data = get_ts_dataset(dataset, path_dataset, normalize='standard')
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

    X_train_vae, X_val_vae, _, _ = train_test_split(X_train, y_train, test_size=test_size,
                                                    random_state=random_state, stratify=y_train)

    input_shape = X_train.shape[1:]

    ae, enc, dec = ae_model_cegp(input_shape)
    ae.summary()
    ae.fit(X_train_vae, X_train_vae, epochs=epochs, batch_size=batch_size, validation_data=(X_val_vae, X_val_vae),
           verbose=1)
    ae.save(path_ae + name + '.h5', save_format='h5')
    enc.save(path_ae + name + '_enc' + '.h5', save_format='h5')


if __name__ == "__main__":
    main()

