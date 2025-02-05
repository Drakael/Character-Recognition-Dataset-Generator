"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelitelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Flatten, TimeDistributed, LSTM, Reshape
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import *
from keras.optimizers import *
from sklearn import metrics
import seaborn as sns
import os
import numpy as np
import pandas as pd
import json
root_dir = 'dataset_bw'

class_dirs = os.listdir(root_dir + '/test')

nb_classes = len(class_dirs)

print('nb_classes', nb_classes)

width = height = 24

nb_channels = 1

if nb_channels == 1:
    color_mode = 'grayscale'
else:
    color_mode = 'rgb'

batch_size = 1
datagen = ImageDataGenerator(rescale=1/255)
train_generator = datagen.flow_from_directory(directory=root_dir + '/train',
                                              target_size=(width, height),
                                              batch_size=batch_size * nb_classes,
                                              class_mode="categorical",
                                              color_mode=color_mode,
                                              # shuffle=True,
                                              seed=0)
valid_generator = datagen.flow_from_directory(directory=root_dir + '/valid',
                                              target_size=(width, height),
                                              batch_size=batch_size * nb_classes,
                                              class_mode="categorical",
                                              color_mode=color_mode,
                                              # shuffle=True,
                                              seed=0)
test_generator = datagen.flow_from_directory(directory=root_dir + '/test',
                                             target_size=(width, height),
                                             batch_size=batch_size * nb_classes,
                                             class_mode="categorical",
                                             color_mode=color_mode,
                                             shuffle=False,
                                             seed=0)

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=64, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=8, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Models for training and evaluation (prediction)
    train_model = models.Model(x, out_caps)
    eval_model = models.Model(x, out_caps)

    return train_model, eval_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}e-{val_acc:03f}.h5', monitor='val_acc',
                                           save_best_only=True, save_weights_only=False, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=margin_loss,
                  metrics={'capsnet': 'accuracy'})
    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """
    model_name = 'OCR_French_CapsNet'
    model.load_weights(args.save_dir + '/' + model_name + '.h5')

    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        epochs=args.epochs,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/' + model_name + '.h5')
    print('Trained model saved to \'%s/' + model_name + '.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


# def load_mnist():
#     # the data, shuffled and split between train and test sets
#     from keras.datasets import mnist
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()

#     x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
#     x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
#     y_train = to_categorical(y_train.astype('float32'))
#     y_test = to_categorical(y_test.astype('float32'))
#     return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    # (x_train, y_train), (x_test, y_test) = load_mnist()

    # define model
    model, eval_model = CapsNet(input_shape=(height, width, nb_channels),
                                n_class=nb_classes,
                                routings=args.routings)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    train(model=model, args=args)
