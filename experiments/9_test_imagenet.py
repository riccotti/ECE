# from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
# import numpy as np
#
# model = InceptionV3(weights='imagenet')
#
# img_path = '/Users/riccardo/Documents/Research/CounterfactualExplanations/dataset/imagenet/ILSVRC2012_val_00049530.JPEG'
# img = image.load_img(img_path, target_size=(299, 299))
# x = image.img_to_array(img)
# print(x)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
# print(x)
#
# preds = model.predict(x)
# # decode the results into a list of tuples (class, description, probability)
# # (one such list for each sample in the batch)
# print('Predicted:', decode_predictions(preds, top=3)[0])
# # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
#
# print(model.predict(x).shape)
# # print(model.predict_proba(x))

import os
import datetime

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

from experiments.config import *
from experiments.util import get_image_dataset

from ece.blackbox import BlackBox


def main():

    import tensorflow as tf
    print(tf.config.list_physical_devices("GPU"))
    print(tf.test.gpu_device_name())

    dataset = 'imagenet1000'
    black_box = 'VGG16'

    if dataset not in dataset_list:
        print('unknown dataset %s' % dataset)
        return -1

    if black_box not in blackbox_list:
        print('unknown black box %s' % black_box)
        return -1

    print(datetime.datetime.now(), dataset, black_box)

    data = get_image_dataset(dataset, path_dataset, categories=None, filter=None,
                             use_rgb=True, flatten=False, model=black_box)
    _, X_test, _, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    # decode_predictions = data['decode_predictions']
    X_test, y_test = X_test, y_test

    if black_box == 'InceptionV3':
        from tensorflow.keras.layers import Input
        input_tensor = Input(shape=(224, 224, 3))
        bb = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
    elif black_box == 'VGG16':
        bb = VGG16(weights='imagenet')
    elif black_box == 'VGG19':
        bb = VGG19(weights='imagenet')
    elif black_box == 'ResNet50':
        bb = ResNet50(weights='imagenet')
    elif black_box == 'InceptionResNetV2':
        from tensorflow.keras.layers import Input
        input_tensor = Input(shape=(224, 224, 3))
        bb = InceptionResNetV2(input_tensor=input_tensor, weights='imagenet', include_top=True)
    else:
        print('unknown black box %s' % black_box)
        return -1

    bb = BlackBox(bb)

    y_pred_test = bb.predict(X_test)

    res = {
        'dataset': dataset,
        'black_box': black_box,
        'accuracy_train': accuracy_score(y_test, y_pred_test),
        'accuracy_test': accuracy_score(y_test, y_pred_test),
        'f1_macro_train': f1_score(y_test, y_pred_test, average='macro'),
        'f1_macro_test': f1_score(y_test, y_pred_test, average='macro'),
        'f1_micro_train': f1_score(y_test, y_pred_test, average='micro'),
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

    filename_results = path_results + 'classifiers_performance_img.csv'
    if not os.path.isfile(filename_results):
        df.to_csv(filename_results, index=False)
    else:
        df.to_csv(filename_results, mode='a', index=False, header=False)


if __name__ == "__main__":
    main()