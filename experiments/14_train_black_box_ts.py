import os
import pickle
import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers import Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D
from keras.layers import Bidirectional, LSTM
from keras.layers import Permute, concatenate
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from experiments.config import *
from experiments.util import get_ts_dataset

from ece.blackbox import BlackBox

# from keras import backend as K
# print(K.image_data_format()) # print current format
# K.set_image_data_format('channels_first') # set format


def build_cnn(input_shape, output_shape, optimizer, loss):
    # input_shape = (n_timesteps, 1)
    # optimizer = 'adam'
    # loss = 'sparse_categorical_crossentropy'

    model = Sequential()

    model.add(Conv1D(filters=16, kernel_size=8, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(GlobalAveragePooling1D())

    # model.add(Dense(30, activation='relu'))
    model.add(Dense(output_shape, activation='sigmoid'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return model


def build_resnet(input_shape, output_shape, optimizer, loss, n_feature_maps=64):

    input_layer = Input(input_shape)

    # BLOCK 1

    conv_x = Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = Activation('relu')(output_block_1)

    # BLOCK 2

    conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = Activation('relu')(output_block_2)

    # BLOCK 3

    conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = BatchNormalization()(output_block_2)

    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = Activation('relu')(output_block_3)

    # FINAL

    gap_layer = GlobalAveragePooling1D()(output_block_3)

    output_layer = Dense(output_shape, activation='softmax')(gap_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return model


def build_bilstm(input_shape, output_shape, optimizer, loss):
    model = Sequential()

    model.add(Bidirectional(LSTM(units=128, input_shape=input_shape)))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(units=128, activation='relu'))
    model.add(keras.layers.Dense(output_shape, activation='softmax'))

    model.add(Dense(output_shape, activation='sigmoid'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return model


def build_lstmfcn(input_shape, output_shape, optimizer, loss, num_cells=8):

    input_layer = Input(input_shape)

    x = LSTM(num_cells)(input_layer)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(input_layer)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    output_layer = Dense(output_shape, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    # model.summary()

    # add load model code here to fine-tune

    return model


# def build_alstmfcn(input_shape, output_shape, optimizer, loss, num_cells=8):
#
#     # input_layer = Input(input_shape)
#     input_layer = Input((input_shape[1], input_shape[0]))
#     from experiments.layer_utils import AttentionLSTM
#
#     x = AttentionLSTM(num_cells)(input_layer)
#     x = Dropout(0.8)(x)
#
#     y = Permute((2, 1))(input_layer)
#     y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
#     y = BatchNormalization()(y)
#     y = Activation('relu')(y)
#
#     y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
#     y = BatchNormalization()(y)
#     y = Activation('relu')(y)
#
#     y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
#     y = BatchNormalization()(y)
#     y = Activation('relu')(y)
#
#     y = GlobalAveragePooling1D()(y)
#
#     x = concatenate([x, y])
#
#     output_layer = Dense(output_shape, activation='softmax')(x)
#
#     model = Model(inputs=input_layer, outputs=output_layer)
#     model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
#
#     # model.summary()
#
#     # add load model code here to fine-tune
#
#     return model


def main():
    # import tensorflow as tf
    # print(tf.config.list_physical_devices("GPU"))
    # print(tf.test.gpu_device_name())
    # return -1

    # dataset = sys.argv[1]
    # black_box = sys.argv[2]
    # dataset = 'gunpoint'
    # black_box = 'BiLSTM'
    normalize = 'standard'

    for black_box in [ #'BiLSTM',
        'LSTMFCN']:

        for dataset in [
            'arrowhead', 'ecg200', 'ecg5000', 'ecg5days', 'facefour', 'herring',
                        'gunpoint',
                        'italypower',
            'electricdevices', 'phalanges'
        ]:

            if dataset not in dataset_list:
                print('unknown dataset %s' % dataset)
                return -1

            if black_box not in blackbox_list:
                print('unknown black box %s' % black_box)
                return -1

            print(datetime.datetime.now(), dataset, black_box)

            data = get_ts_dataset(dataset, path_dataset, normalize)
            X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

            dim_in = X_train.shape[1:]
            dim_out = len(np.unique(y_train))
            optimizer = 'adam'
            loss = 'sparse_categorical_crossentropy'

            X_train_dnn, X_val_dnn, y_train_dnn, y_val_dnn = train_test_split(X_train, y_train, test_size=test_size,
                                                                              random_state=random_state, stratify=y_train)

            if black_box == 'CNN':
                bb = build_cnn(input_shape=dim_in, output_shape=dim_out, optimizer=optimizer, loss=loss)

                checkpoint = ModelCheckpoint(filepath=path_models + '%s_%s.h5' % (dataset, black_box),
                                             monitor='val_acc', verbose=1, save_best_only=True)
                lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=0.5e-6)

                callbacks = [checkpoint, lr_reducer]

                bb.fit(X_train_dnn, y_train_dnn, validation_data=(X_val_dnn, y_val_dnn), epochs=1000, batch_size=128,
                       shuffle=True, callbacks=callbacks)

            elif black_box == 'ResNet':
                bb = build_resnet(input_shape=dim_in, output_shape=dim_out, optimizer=optimizer, loss=loss,
                                  n_feature_maps=64)

                checkpoint = ModelCheckpoint(filepath=path_models + '%s_%s.h5' % (dataset, black_box),
                                             monitor='val_acc', verbose=1, save_best_only=True)
                lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=0.5e-6)

                callbacks = [checkpoint, lr_reducer]

                bb.fit(X_train_dnn, y_train_dnn, validation_data=(X_val_dnn, y_val_dnn), epochs=1000, batch_size=128,
                       shuffle=True, callbacks=callbacks)

            elif black_box == 'BiLSTM':
                bb = build_bilstm(input_shape=dim_in, output_shape=dim_out, optimizer=optimizer, loss=loss)

                checkpoint = ModelCheckpoint(filepath=path_models + '%s_%s.h5' % (dataset, black_box),
                                             monitor='val_acc', verbose=1, save_best_only=True)
                lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=0.5e-6)

                callbacks = [checkpoint, lr_reducer]

                bb.fit(X_train_dnn, y_train_dnn, validation_data=(X_val_dnn, y_val_dnn), epochs=1000, batch_size=128,
                       shuffle=True, callbacks=callbacks)

            elif black_box == 'LSTMFCN':
                # Number of cells CELLS = [8, 64, 128]
                bb = build_lstmfcn(input_shape=dim_in, output_shape=dim_out, optimizer=optimizer, loss=loss, num_cells=8)

                checkpoint = ModelCheckpoint(filepath=path_models + '%s_%s.h5' % (dataset, black_box),
                                             monitor='loss', verbose=1, save_best_only=True)
                lr_reducer = ReduceLROnPlateau(monitor='loss', factor=1. / np.cbrt(2), cooldown=0, patience=100,
                                               min_lr=1e-4)

                callbacks = [checkpoint, lr_reducer]

                bb.fit(X_train_dnn, y_train_dnn, validation_data=(X_val_dnn, y_val_dnn), epochs=1000, batch_size=128,
                       shuffle=True, callbacks=callbacks)

            # elif black_box == 'ALSTMFCN':
            #     # Number of cells CELLS = [8, 64, 128]
            #     bb = build_alstmfcn(input_shape=dim_in, output_shape=dim_out, optimizer=optimizer, loss=loss, num_cells=8)
            #
            #     checkpoint = ModelCheckpoint(filepath=path_models + '%s_%s.h5' % (dataset, black_box),
            #                                  monitor='loss', verbose=1, save_best_only=True)
            #     lr_reducer = ReduceLROnPlateau(monitor='loss', factor=1. / np.cbrt(2), cooldown=0, patience=100,
            #                                    min_lr=1e-4)
            #
            #     callbacks = [checkpoint, lr_reducer]
            #
            #     bb.fit(X_train_dnn, y_train_dnn, validation_data=(X_val_dnn, y_val_dnn), epochs=1000, batch_size=128,
            #            shuffle=True, callbacks=callbacks)

            else:
                print('not implemented trainign for black box %s' % black_box)
                return -1

            bb.save(path_models + '%s_%s.h5' % (dataset, black_box))

            bb = BlackBox(bb)

            y_pred_train = bb.predict(X_train)
            y_pred_test = bb.predict(X_test)

            res = {
                'dataset': dataset,
                'black_box': black_box,
                'accuracy_train': accuracy_score(y_train, y_pred_train),
                'accuracy_test': accuracy_score(y_test, y_pred_test),
                'f1_macro_train': f1_score(y_train, y_pred_train, average='macro'),
                'f1_macro_test': f1_score(y_test, y_pred_test, average='macro'),
                'f1_micro_train': f1_score(y_train, y_pred_train, average='micro'),
                'f1_micro_test': f1_score(y_test, y_pred_test, average='micro'),
            }

            print(dataset, black_box)
            print('accuracy_train', res['accuracy_train'])
            print('accuracy_test', res['accuracy_test'])
            # print(np.unique(bb.predict(X_test), return_counts=True))

            df = pd.DataFrame(data=[res])
            columns = ['dataset', 'black_box', 'accuracy_train', 'accuracy_test', 'f1_macro_train', 'f1_macro_test',
                       'f1_micro_train', 'f1_micro_test']
            df = df[columns]

            filename_results = path_results + 'classifiers_performance_ts.csv'
            if not os.path.isfile(filename_results):
                df.to_csv(filename_results, index=False)
            else:
                df.to_csv(filename_results, mode='a', index=False, header=False)


if __name__ == "__main__":
    main()


