import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras import layers, models, optimizers
import json
import sys
from capsulelitelayers import CapsuleLayer, PrimaryCap, Length, Mask

nb_channels = 1

kernel_size = 24

strides = 6

if nb_channels == 1:
    color_mode = 0
else:
    color_mode = -1

pic_path = 'ticket_ratp_octobre_2018.jpg'

img = cv2.imread(pic_path, color_mode)
img = img / 255

print('img.shape', img.shape)

plt.imshow(img, cmap='gray')
plt.show()

height = img.shape[0]

width = img.shape[1]

# if nb_channels == 1:
#     img = img.reshape((height, width, 1))

nb_patch = int((((height - kernel_size) / strides) + 1) *
               (((width - kernel_size) / strides) + 1))

print('nb_patch', nb_patch)

dump_folder = 'dump_analysis/'
batch = np.zeros((nb_patch, kernel_size, kernel_size))
cnt = 0
for i in range(0, height - kernel_size, strides):
    for j in range(0, width - kernel_size, strides):
        end_i = i + kernel_size
        end_j = j + kernel_size
        batch[cnt, ...] = img[i:end_i, j:end_j]
        # print('cnt', cnt, 'i', i, 'j', j, 'min', np.min(batch[cnt, ...]),
        #       'max', np.max(batch[cnt, ...]))
        cnt += 1

for i in range(2900, 2905):
    plt.figure()
    plt.imshow(batch[i, ...], cmap='gray')
    plt.show()

# sys.exit('die')


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
    return models.Model(x, out_caps)


with open('char_list.json') as json_file:
    char_list = json.load(json_file)

print('char_list', char_list)

with open('labels.json') as json_file:
    labels = json.load(json_file)

# model = CapsNet(input_shape=(height, width, nb_channels),
#                 n_class=len(char_list),
#                 routings=3)
# model.load_weights('weights-20e-0.985168.h5')


model_name = 'OCR_French_CNN_64_adadelta_992.h5'
# model_name = 'OCR_French_CapsNet.h5'

# model_dir = 'models_dump/'
# model_dir = 'result/'
model = load_model(model_name)

batch = batch.reshape((batch.shape[0], batch.shape[1], batch.shape[2], 1))

pred = model.predict(batch, verbose=1, batch_size=32)

predicted_class_indices = np.argmax(pred, axis=1)

for i, sample in enumerate(batch):
    preds = np.argsort(pred[i])[::-1]
    str_ = str(i) + '_' + char_list[labels[str(preds[0])]] + str(pred[i][preds[0]])
    str_ += '_'
    str_ += char_list[labels[str(preds[1])]] + str(pred[i][preds[1]])
    str_ += '_'
    str_ += char_list[labels[str(preds[2])]] + str(pred[i][preds[2]])
    str_ += '.jpg'
    cv2.imwrite(dump_folder + str_, batch[i, ...])


print('labels', labels)

predictions_str = [char_list[labels[str(k)]] for k in predicted_class_indices]
print('predictions_str', predictions_str)
