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
from keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.regularizers import l2

from experiments.config import *
from experiments.util import get_image_dataset

from ece.blackbox import BlackBox


def build_dnn(input_shape, output_shape, optimizer, loss):
    model = Sequential()

    # add Convolutional layers
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    # Densely connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # output layer
    model.add(Dense(output_shape, activation='softmax'))
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.summary()
    return model


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, output_shape, depth):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1, strides=strides,
                                 activation=None, batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(output_shape,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, output_shape, depth):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(output_shape,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def build_resnet(input_shape, output_shape, loss, depth, filepath, model_type='v1'):
    if model_type == 'v1':
        model = resnet_v1(input_shape=input_shape, output_shape=output_shape, depth=depth)
    else:
        model = resnet_v2(input_shape=input_shape, output_shape=output_shape, depth=depth)

    model.compile(loss=loss, optimizer=Adam(lr=lr_schedule(0)), metrics=['accuracy'])
    model.summary()

    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    return model, callbacks


def main():
    # import tensorflow as tf
    # print(tf.config.list_physical_devices("GPU"))
    # print(tf.test.gpu_device_name())
    # return -1

    # dataset = sys.argv[1]
    # black_box = sys.argv[2]
    # dataset = 'cifar10'
    black_box = 'DNN2'
    normalize = '255'
    # normalize_str = '' if normalize is None else '_%s' % normalize
    for dataset in [ #'mnist', 'fashion_mnist',
                    'cifar10']:

        if dataset not in dataset_list:
            print('unknown dataset %s' % dataset)
            return -1

        if black_box not in blackbox_list:
            print('unknown black box %s' % black_box)
            return -1

        # filter = 'sobel' if dataset == 'cifar10' else None

        print(datetime.datetime.now(), dataset, black_box)

        data = get_image_dataset(dataset, path_dataset, categories=None, filter=None,
                                 use_rgb=True, flatten=False)
        X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

        # print(X_train.shape)
        # print(np.min(X_train[0]), np.max(X_train[0]), np.mean(X_train[0]), np.median(X_train[0]))

        if black_box in ['DNN', 'DNN2']:
            dim_0 = X_train.shape[1:]
            if len(np.unique(y_train)) > 2:
                encoder = OneHotEncoder()
                y_train1h = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
                loss = 'categorical_crossentropy'
                dim_out = len(np.unique(y_train))
            else:
                y_train1h = y_train
                loss = 'binary_crossentropy'
                dim_out = 1
        else:
            print('not implemented trainign for black box %s' % black_box)
            return -1

        X_train_dnn, X_val_dnn, y_train_dnn, y_val_dnn = train_test_split(X_train, y_train1h, test_size=test_size,
                                                                          random_state=random_state, stratify=y_train1h)

        if 'cifar' not in dataset:
            bb = build_dnn(input_shape=dim_0, output_shape=dim_out, optimizer='adam', loss=loss)
            bb.fit(X_train_dnn, y_train_dnn, validation_data=(X_val_dnn, y_val_dnn), epochs=100, batch_size=128)
        else:
            n = 3
            depth = n * 6 + 2
            model_type = 'v2' if '2' in black_box else 'v1'
            bb, callbacks = build_resnet(input_shape=dim_0, output_shape=dim_out, loss=loss, depth=depth,
                                         filepath=path_models + '%s_%s.h5' % (dataset, black_box),
                                         model_type=model_type)
            bb.fit(X_train_dnn, y_train_dnn, validation_data=(X_val_dnn, y_val_dnn), epochs=200, batch_size=128,
                   shuffle=True, callbacks=callbacks)

        if black_box in ['DNN', 'DNN2']:
            bb.save(path_models + '%s_%s.h5' % (dataset, black_box))
        else:
            print('not implemented training for black box %s' % black_box)
            return -1

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
        print(np.unique(bb.predict(X_test), return_counts=True))

        df = pd.DataFrame(data=[res])
        columns = ['dataset', 'black_box', 'accuracy_train', 'accuracy_test', 'f1_macro_train', 'f1_macro_test',
                   'f1_micro_train', 'f1_micro_test']
        df = df[columns]

        filename_results = path_results + 'classifiers_performance_img.csv'
        if not os.path.isfile(filename_results):
            df.to_csv(filename_results, index=False)
        else:
            df.to_csv(filename_results, mode='a', index=False, header=False)


if __name__ == "__main__":
    main()


