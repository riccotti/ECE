import pickle
import numpy as np

import keras.backend as K
from keras.layers import Dense, Input
from keras.layers import Conv2D, Conv2DTranspose, Flatten, Lambda
from keras.layers import Conv1D
from keras.layers import Reshape, Concatenate
from keras.models import Model
from keras.models import model_from_json
from keras.losses import mse, binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import OneHotEncoder


# reparameterization trick: instead of sampling from Q(z|X), sample epsilon = N(0,1)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def save(model, model_name, path):
    model_filename = '%s%s.json' % (path, model_name)
    weights_filename = '%s%s_weights.hdf5' % (path, model_name)
    options = {'file_arch': model_filename, 'file_weight': weights_filename}
    json_string = model.to_json()
    open(options['file_arch'], 'w').write(json_string)
    model.save_weights(options['file_weight'])


def load(model_name, path):
    model_filename = "%s%s.json" % (path, model_name)
    weights_filename = "%s%s_weights.hdf5" % (path, model_name)
    model = model_from_json(open(model_filename, 'r').read())
    model.load_weights(weights_filename)
    return model


class ConditionalVariationalAutoencoderImage:

    def __init__(self, input_shape, n_classes, latent_dim=2, num_conv_layers=2, filters=16, kernel_size=4,
                 strides=(1, 1), hidden_dim=16, use_mse=False, optimizer='adam',
                 store_intermediate=False, save_graph=False, path='./', name='vae', verbose=False, patience=20):

        self.input_shape = input_shape
        self.image_size = input_shape[1]
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.num_conv_layers = num_conv_layers
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.hidden_dim = hidden_dim
        self.use_mse = use_mse
        self.optimizer = optimizer

        self.store_intermediate = store_intermediate
        self.save_graph = save_graph
        self.path = path
        self.name = name
        self.verbose = verbose
        self.patience = patience

        self.oh_encoder = OneHotEncoder()

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.cvae = self._build_cvae()

    def _build_encoder(self):
        self.inputs = Input(shape=self.input_shape, name='encoder_input1')
        self.input_cond = Input(shape=(self.n_classes,), name='encoder_input2')
        x = self.inputs
        for i in range(self.num_conv_layers):
            self.filters *= 2
            x = Conv2D(filters=self.filters,
                       kernel_size=self.kernel_size,
                       activation='relu',
                       strides=self.strides,
                       padding='same')(x)

        # shape info needed to build decoder model
        self.shape = K.int_shape(x)

        # generate latent vector Q(z|X)
        x = Flatten()(x)
        x = Dense(self.hidden_dim, activation='relu')(x)
        self.z_mean = Dense(self.latent_dim, name='z_mean')(x)
        self.z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])
        # z_cond = Concatenate(axis=-1)([z, self.input_cond])

        # instantiate encoder model
        encoder = Model([self.inputs, self.input_cond], [self.z_mean, self.z_log_var, z, self.input_cond],
                        name='encoder')
        if self.verbose:
            encoder.summary()
        return encoder

    def _build_decoder(self):
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        latent_cond = Input(shape=(self.n_classes,))

        x = Concatenate(axis=1)([latent_inputs, latent_cond])
        x = Dense(self.latent_dim, activation='relu')(x)
        x = Dense(self.hidden_dim, activation='relu')(x)
        x = Dense(self.shape[1] * self.shape[2] * self.shape[3], activation='relu')(x)
        x = Reshape((self.shape[1], self.shape[2], self.shape[3]))(x)

        # use Conv2DTranspose to reverse the conv layers from the encoder
        for i in range(self.num_conv_layers):
            x = Conv2DTranspose(filters=self.filters,
                                kernel_size=self.kernel_size,
                                activation='relu',
                                strides=self.strides,
                                padding='same')(x)
            self.filters //= 2

        outputs = Conv2DTranspose(filters=1,
                                  kernel_size=self.kernel_size,
                                  activation='sigmoid',
                                  padding='same',
                                  name='decoder_output')(x)

        # instantiate decoder model
        decoder = Model([latent_inputs, latent_cond], outputs, name='decoder')
        if self.verbose:
            decoder.summary()
        return decoder

    def _build_cvae(self):

        output = self.decoder(self.encoder([self.inputs, self.input_cond])[2:])
        cvae = Model([self.inputs, self.input_cond], output, name='cvae')

        if self.use_mse:
            reconstruction_loss = mse(K.flatten(self.inputs), K.flatten(output))
        else:
            reconstruction_loss = binary_crossentropy(K.flatten(self.inputs), K.flatten(output))

        reconstruction_loss *= self.image_size * self.image_size
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        cvae_loss = K.mean(reconstruction_loss + kl_loss)
        cvae.add_loss(cvae_loss)

        cvae.compile(optimizer=self.optimizer)
        if self.verbose:
            cvae.summary()
        return cvae

    def save_model(self):
        save(self.encoder, '%s_encoder' % self.name, self.path)
        save(self.decoder, '%s_decoder' % self.name, self.path)
        save(self.cvae, '%s_cvae' % self.name, self.path)
        pickle_file = open('%s%s_oh_encoder' % (self.path, self.name), 'wb')
        pickle.dump(self.oh_encoder, pickle_file)
        pickle_file.close()

    def load_model(self):
        self.encoder = load('%s_encoder' % self.name, self.path)
        self.decoder = load('%s_decoder' % self.name, self.path)
        self.cvae = load('%s_cvae' % self.name, self.path)
        self.oh_encoder = pickle.load(open('%s%s_oh_encoder' % (self.path, self.name), 'rb'))

    def fit(self, Xy, Xy_val=None, epochs=10000, batch_size=128):
        Xy[1] = self.oh_encoder.fit_transform(Xy[1].reshape(-1, 1)).toarray()
        Xy_val[1] = self.oh_encoder.transform(Xy_val[1].reshape(-1, 1)).toarray()
        # validation_data = (Xy_val, None)
        validation_data = (Xy_val, Xy_val)
        es = EarlyStopping(monitor='val_loss', verbose=self.verbose, patience=self.patience)
        cp = ModelCheckpoint('%s%s_cvae_cp.hdf5' % (self.path, self.name), verbose=self.verbose, save_best_only=True,
                             save_weights_only=True)
        # self.cvae.fit(Xy, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=self.verbose,
        #               callbacks=[es, cp])
        self.cvae.fit(Xy, Xy, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=self.verbose,
                      callbacks=[es, cp])
        self.cvae.load_weights('%s%s_cvae_cp.hdf5' % (self.path, self.name))
        self.save_model()

    def encode(self, X):
        return self.encoder.predict(X)

    def decode(self, X):
        return self.decoder.predict(X)

    def generate(self, samples=1, class_val=None):
        Z = np.random.normal(size=(samples, self.latent_dim))
        if class_val is None:
            y = np.random.choice(range(self.n_classes), size=samples, replace=True)
        else:
            y = np.array([class_val] * samples)
        y = self.oh_encoder.transform(y.reshape(-1, 1)).toarray()
        X = self.decode([Z, y])
        return X


class ConditionalVariationalAutoencoderTimeSeries:

    def __init__(self, input_shape, n_classes, latent_dim=2, num_conv_layers=2, filters=16, kernel_size=4,
                 strides=1, hidden_dim=16, use_mse=False, optimizer='adam',
                 store_intermediate=False, save_graph=False, path='./', name='vae', verbose=False, patience=20):
        self.input_shape = input_shape
        self.ts_size = input_shape[1]
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.num_conv_layers = num_conv_layers
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.hidden_dim = hidden_dim
        self.use_mse = use_mse
        self.optimizer = optimizer

        self.store_intermediate = store_intermediate
        self.save_graph = save_graph
        self.path = path
        self.name = name
        self.verbose = verbose
        self.patience = patience

        self.oh_encoder = OneHotEncoder()

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.cvae = self._build_cvae()

    def _build_encoder(self):
        self.inputs = Input(shape=self.input_shape, name='encoder_input1')
        self.input_cond = Input(shape=(self.n_classes,), name='encoder_input2')
        x = self.inputs
        for i in range(self.num_conv_layers):
            self.filters *= 2
            x = Conv1D(filters=self.filters,
                       kernel_size=self.kernel_size,
                       activation='relu',
                       strides=self.strides,
                       padding='same')(x)

        # shape info needed to build decoder model
        self.shape = K.int_shape(x)

        # generate latent vector Q(z|X)
        x = Flatten()(x)
        x = Dense(self.hidden_dim, activation='relu')(x)
        self.z_mean = Dense(self.latent_dim, name='z_mean')(x)
        self.z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])
        # z_cond = Concatenate(axis=-1)([z, self.input_cond])

        # instantiate encoder model
        encoder = Model([self.inputs, self.input_cond], [self.z_mean, self.z_log_var, z, self.input_cond],
                        name='encoder')
        if self.verbose:
            encoder.summary()
        return encoder

    def _build_decoder(self):
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        latent_cond = Input(shape=(self.n_classes,))

        x = Concatenate(axis=1)([latent_inputs, latent_cond])
        x = Dense(self.latent_dim, activation='relu')(x)
        x = Dense(self.hidden_dim, activation='relu')(x)
        x = Dense(self.shape[1] * self.shape[2], activation='relu')(x)
        x = Reshape((self.shape[1], self.shape[2]))(x)

        # use Conv2DTranspose to reverse the conv layers from the encoder
        for i in range(self.num_conv_layers):
            x = Conv1D(filters=self.filters,
                       kernel_size=self.kernel_size,
                       activation='relu',
                       strides=self.strides,
                       padding='same')(x)
            self.filters //= 2

        outputs = Conv1D(filters=1,
                         kernel_size=self.kernel_size,
                         activation='sigmoid',
                         padding='same',
                         name='decoder_output')(x)

        # instantiate decoder model
        decoder = Model([latent_inputs, latent_cond], outputs, name='decoder')
        if self.verbose:
            decoder.summary()
        return decoder

    def _build_cvae(self):
        output = self.decoder(self.encoder([self.inputs, self.input_cond])[2:])
        cvae = Model([self.inputs, self.input_cond], output, name='cvae')

        if self.use_mse:
            reconstruction_loss = mse(K.flatten(self.inputs), K.flatten(output))
        else:
            reconstruction_loss = binary_crossentropy(K.flatten(self.inputs), K.flatten(output))

        reconstruction_loss *= self.ts_size  # * self.image_size
        kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        cvae_loss = K.mean(reconstruction_loss + kl_loss)
        cvae.add_loss(cvae_loss)

        cvae.compile(optimizer=self.optimizer)
        if self.verbose:
            cvae.summary()
        return cvae

    def save_model(self):
        save(self.encoder, '%s_encoder' % self.name, self.path)
        save(self.decoder, '%s_decoder' % self.name, self.path)
        save(self.cvae, '%s_cvae' % self.name, self.path)
        pickle_file = open('%s%s_oh_encoder' % (self.path, self.name), 'wb')
        pickle.dump(self.oh_encoder, pickle_file)
        pickle_file.close()

    def load_model(self):
        self.encoder = load('%s_encoder' % self.name, self.path)
        self.decoder = load('%s_decoder' % self.name, self.path)
        self.cvae = load('%s_cvae' % self.name, self.path)
        self.oh_encoder = pickle.load(open('%s%s_oh_encoder' % (self.path, self.name), 'rb'))

    def fit(self, Xy, Xy_val=None, epochs=10000, batch_size=128):
        Xy[1] = self.oh_encoder.fit_transform(Xy[1].reshape(-1, 1)).toarray()
        Xy_val[1] = self.oh_encoder.transform(Xy_val[1].reshape(-1, 1)).toarray()
        # validation_data = (Xy_val, None)
        validation_data = (Xy_val, Xy_val)
        es = EarlyStopping(monitor='val_loss', verbose=self.verbose, patience=self.patience)
        cp = ModelCheckpoint('%s%s_cvae_cp.hdf5' % (self.path, self.name), verbose=self.verbose, save_best_only=True,
                             save_weights_only=True)
        # self.cvae.fit(Xy, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=self.verbose,
        #               callbacks=[es, cp])
        self.cvae.fit(Xy, Xy, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=self.verbose,
                      callbacks=[es, cp])
        self.cvae.load_weights('%s%s_cvae_cp.hdf5' % (self.path, self.name))
        self.save_model()

    def encode(self, X):
        return self.encoder.predict(X)

    def decode(self, X):
        return self.decoder.predict(X)

    def generate(self, samples=1, class_val=None):
        Z = np.random.normal(size=(samples, self.latent_dim))
        if class_val is None:
            y = np.random.choice(range(self.n_classes), size=samples, replace=True)
        else:
            y = np.array([class_val] * samples)
        y = self.oh_encoder.transform(y.reshape(-1, 1)).toarray()
        X = self.decode([Z, y])
        return X

