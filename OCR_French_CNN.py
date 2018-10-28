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

batch_size = 32
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

# x, y = train_generator.next()
# print(x.shape, y.shape)

nb_kernels = 58

input_img = Input(shape=(height, width, nb_channels))
# layer shape 24 x 24
x = Conv2D(nb_kernels, (9, 9), data_format='channels_last',
           activation='relu', padding='same')(input_img)
x = Dropout(0.2)(x)
# x = BatchNormalization()(x)
x = Conv2D(nb_kernels, (9, 9), data_format='channels_last',
           activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)
# x = BatchNormalization()(x)
# layer shape 12 x 12
x = Conv2D(nb_kernels, (9, 9), data_format='channels_last',
           activation='relu', padding='same')(x)
x = Dropout(0.2)(x)
# x = BatchNormalization()(x)
x = Conv2D(nb_kernels, (9, 9), data_format='channels_last',
           activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)
# layer shape 6 x 6
x = Flatten()(x)
x = Dense(nb_classes, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(nb_classes, activation='softmax')(x)

model = Model(input_img, output)
model.summary()
model.compile(optimizer='adadelta', loss=categorical_crossentropy,
              metrics=['accuracy'])

model_name = 'OCR_French_CNN_' + str(nb_kernels) + '_adadelta'

model_dir = 'models_dump/'
model_name = 'OCR_French_CNN_64_adadelta_992'
model = load_model(model_dir + model_name + '.h5')

checkpointer = ModelCheckpoint(filepath=model_dir + model_name +
                               '.{epoch:03d}e-acc{val_acc:.3f}.h5',
                               verbose=1, save_best_only=True)

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size

# model.fit_generator(
#     train_generator,
#     steps_per_epoch=STEP_SIZE_TRAIN,
#     epochs=50,
#     verbose=1,
#     validation_data=valid_generator,
#     validation_steps=STEP_SIZE_VALID,
#     callbacks=[checkpointer])

# model.save(model_dir + model_name + '.h5')

score = model.evaluate_generator(generator=valid_generator, verbose=1,
                                 steps=STEP_SIZE_VALID)
print("Valid Loss: ", score[0], "Accuracy: ", score[1])

test_generator.reset()

STEP_SIZE_TEST = test_generator.n//test_generator.batch_size + 1

score = model.evaluate_generator(generator=test_generator, verbose=1,
                                 steps=STEP_SIZE_TEST)
print("Valid Loss: ", score[0], "Accuracy: ", score[1])


test_generator.reset()

pred = model.predict_generator(test_generator, verbose=1, steps=STEP_SIZE_TEST)

predicted_class_indices = np.argmax(pred, axis=1)

# char_list = 'abcdefghijklmnopqrstuvwxyzàâäéèêëîïôùûüœçABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/\\,.;:!?"\'’-_()[]\{\}|&*+=%$µ§°#~@£¤` '
with open('char_list.json') as json_file:
    char_list = json.load(json_file)

print('char_list', char_list)

labels = (test_generator.class_indices)
# print('labels', labels)
labels = dict((v, int(k)) for k, v in labels.items())
with open('labels.json', 'w') as outfile:
    json.dump(labels, outfile)
# print('labels 2', labels)
predictions = [labels[k] for k in predicted_class_indices]
# print('predictions', predictions)
predictions_str = [char_list[labels[k]] for k in predicted_class_indices]
print('predictions_str', predictions_str)

y_test = test_generator.classes
# print('y_test', y_test)

confusion_matrix = metrics.confusion_matrix(predictions, y_test)

print(metrics.classification_report(predictions, y_test))

print(confusion_matrix)

sns.clustermap(confusion_matrix, annot=True, figsize=(20, 20))

# filenames = test_generator.filenames
# results = pd.DataFrame({"Filename": filenames,
#                        "Predictions": predictions})
# results.to_csv(model_dir + "results_" + model_name + ".csv", index=False)
