from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

char_list = 'abcdefghijklmnopqrstuvwxyzàâäéèêëîïôùûüœçABCDEFGHIJKLMNOPQRSTUV\
             WXYZ0123456789/\\,.;:!?"\'’-_()[]\{\}|&*-+=%$µ§°#~@£¤`'

font_folder = 'fonts'
dataset_size = 5000

root_dir = 'dataset'
if not os.path.exists(root_dir):
    os.makedirs(root_dir)


def generate_pic(class_, str_, font, font_size, background_color, text_color):
    if type(class_) is not str:
        class_ = str(class_)
    img = Image.new('RGB', (24, 24), color=background_color)

    fnt = ImageFont.truetype(font, font_size)
    d = ImageDraw.Draw(img)
    d.text((1, 1), str_, font=fnt, fill=text_color, align='center')

    class_dir = root_dir + '/' + class_
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    pics = os.listdir(class_dir)

    pic_name = str(len(pics))

    img.save(class_dir + '/' + pic_name + '.png')

fonts = os.listdir(font_folder)
for i in range(dataset_size):
    # rand_char_id = np.random.randint(len(char_list))
    # str_ = char_list[rand_char_id]
    rand_char_id = i % len(char_list)
    str_ = char_list[rand_char_id]

    rand_font_id = np.random.randint(len(fonts))
    font = fonts[rand_font_id]
    font_path = os.path.join(font_folder, font)

    font_size = np.random.randint(12, 18)
    background_color = tuple(np.random.randint(0, 256, 3))
    text_color = tuple(np.random.randint(0, 256, 3))
    generate_pic(rand_char_id, str_, font_path, font_size, background_color,
                 text_color)
    print('generated ' + str(i + 1) + '/' + str(dataset_size) + ' samples')
