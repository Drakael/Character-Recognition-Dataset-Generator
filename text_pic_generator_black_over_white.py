from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import json

char_list = 'abcdefghijklmnopqrstuvwxyzàâäéèêëîïôùûüæœçÀÂÄÉÈÊËÎÏÔÙÛÜÆŒÇABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/\\,.;:!?"\'’-_<>()[]{}|&*+=%$µ§°#~@£¤²`^ '
with open('char_list_2.json', 'w') as outfile:
    json.dump(char_list, outfile)
nb_classes = len(char_list)

font_folder = 'fonts'
fonts = os.listdir(font_folder)

dataset_size = 500 * nb_classes
valid_ratio = 0.25
test_ratio = 0.5

root_dir = 'dataset_bw'


def generate_pic(folder, class_, str_, font, font_size, background_color,
                 text_color):
    if type(class_) is not str:
        class_ = str(class_)
    img = Image.new('RGB', (24, 24), color=background_color)

    fnt = ImageFont.truetype(font, font_size)
    d = ImageDraw.Draw(img)
    d.text((1, 1), str_, font=fnt, fill=text_color, align='center')

    class_dir = os.path.join(folder, class_)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    pics = os.listdir(class_dir)

    pic_name = str(len(pics))

    img.save(os.path.join(class_dir, pic_name + '.png'))


def make_dataset_folder(folder, dataset_size):
    for i in range(dataset_size):
        # rand_char_id = np.random.randint(len(char_list))
        # str_ = char_list[rand_char_id]
        rand_char_id = i % len(char_list)
        str_ = char_list[rand_char_id]

        rand_font_id = np.random.randint(len(fonts))
        font = fonts[rand_font_id]
        font_path = os.path.join(font_folder, font)

        font_size = np.random.randint(12, 18)
        rand_background_color = np.random.randint(130, 256, 1)
        background_color = (rand_background_color, rand_background_color,
                            rand_background_color)
        rand_color = np.random.randint(0, 121, 1)
        text_color = (rand_color, rand_color, rand_color)
        generate_pic(folder, rand_char_id, str_, font_path, font_size,
                     background_color, text_color)
        print('generated ' + str(i + 1) + '/' + str(dataset_size) + ' samples')


def make_dataset(root_dir, dataset_size, valid_ratio, test_ratio):
    train_dir = os.path.join(root_dir, 'train')
    valid_dir = os.path.join(root_dir, 'valid')
    test_dir = os.path.join(root_dir, 'test')
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    valid_size = int(dataset_size * valid_ratio)
    test_size = int(dataset_size * test_ratio)
    make_dataset_folder(train_dir, dataset_size)
    make_dataset_folder(valid_dir, valid_size)
    make_dataset_folder(test_dir, test_size)


if __name__ == '__main__':
    make_dataset(root_dir, dataset_size, valid_ratio, test_ratio)
