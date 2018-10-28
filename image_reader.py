import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import json
import sys

nb_channels = 1

kernel_size = 24

strides = 4

if nb_channels == 1:
    color_mode = 0
else:
    color_mode = -1

pic_path = 'Carte_identite_Louis_BEZ.jpg'

img = cv2.imread(pic_path, color_mode)

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

for i in range(800, 805):
    plt.figure()
    plt.imshow(batch[i, ...], cmap='gray')
    plt.show()

# sys.exit('die')

model_name = 'OCR_French_CNN_64_adadelta_992.h5'
# model_name = 'OCR_French_CapsNet.h5'

model_dir = 'models_dump/'
# model_dir = 'result/'

model = load_model(model_dir + model_name)

batch = batch.reshape((batch.shape[0], batch.shape[1], batch.shape[2], 1))

pred = model.predict(batch, verbose=1, batch_size=32)

predicted_class_indices = np.argmax(pred, axis=1)

with open('char_list.json') as json_file:
    char_list = json.load(json_file)

print('char_list', char_list)

with open('labels.json') as json_file:
    labels = json.load(json_file)

print('labels', labels)

predictions_str = [char_list[labels[str(k)]] for k in predicted_class_indices]
print('predictions_str', predictions_str)
