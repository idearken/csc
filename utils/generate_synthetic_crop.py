"""
產生合成的訓練資料影像。使用的材料如下：
1. 從 public training data 裡分割出來的字元影像
2. 從 emnist dataset 裡取出的字元影像
3. 從 public training data 裡取出若干背景圖片 (用影像處理軟體將字元抹除)
"""
import os
import os.path as pth
import glob
import cv2
import tqdm
import random
import numpy as np
from collections import defaultdict


# 總共需要產生的資料數量
num_data = 333_333

# 每個產生的影像裡會有幾個字元，共有 8, 9, 10 三種可能，而產生機率依序如 datalen_ratio 所示。
datalen_list = [8, 9, 10]
datalen_ratio = [0.15, 0.05, 0.8]

# 支援的字元。注意沒有小寫字母及大寫「I」及「O」
char_list = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'

# 每個產生的影像裡會有幾個空白，共有 0, 1, 2, 3 種可能，而機率依序如 blank_ratio 所示。
blank_list = [0, 1, 2, 3]
blank_ratio = [0.1, 0.2, 0.5, 0.2]

background_path = '../../csc-datasets/synthetic_crop/backgrounds'  # 背景圖片所在目錄
handwrite_path = '../../csc-datasets/synthetic_crop/handwrite_chars'  # emnist 字元影像所在目錄
painting_path = '../../csc-datasets/synthetic_crop/painting_chars'  # 從 public training data 裡所分割出來的字元影像所在目錄
output_image_path = '../../csc-datasets/synthetic_crop/synthetic_1000k'  # 合成資料放置目錄
output_data_file = '../data/synthetic_training_1000k.csv'  # 合成資料列表檔 (含標籤)


painting_map = defaultdict(lambda: [])
for f in glob.glob(pth.join(painting_path, '*/*')):
    painting_map[pth.basename(pth.dirname(f))].append(f)

handwrite_map = defaultdict(lambda: [])
for f in glob.glob(pth.join(handwrite_path, '*/*')):
    handwrite_map[pth.basename(pth.dirname(f))].append(f)

mixing_map = defaultdict(lambda: [])
for f in glob.glob(pth.join(painting_path, '*/*')) + glob.glob(pth.join(handwrite_path, '*/*')):
    mixing_map[pth.basename(pth.dirname(f))].append(f)

background_list = glob.glob(pth.join(background_path, '*'))


def get_random_index(probs):
    assert probs[-1] == 1
    r = random.random()
    for i, lvl in enumerate(probs):
        if r < lvl:
            return i


def rand_scale(value, range):
    start, stop = range
    return int((random.random() * (stop - start) + start) * value)


def generate_crop_image(chars, write_file, mode='painting'):
    bg_img = cv2.imread(random.choice(background_list))
    bg_h, bg_w = bg_img.shape[: 2]

    if mode == 'painting':
        img_map = painting_map
        h_scale = (1, 1)
        x_shift = (-0.3, -0.1)
        y_shift = (0, 0)
    elif mode == 'handwrite':
        img_map = handwrite_map
        h_scale = (1, 1.5)
        x_shift = (-0.3, 0.3)
        y_shift = (0.9, 1.1)
    else:
        img_map = mixing_map
        h_scale = (1, 1.5)
        x_shift = (-0.3, 0.3)
        y_shift = (0.9, 1.1)

    x = 2
    ch_img_list = []
    loc_list = []
    for ch in chars:
        if ch == ' ':
            x += rand_scale(30, (0.6, 1.5))
            continue

        ch_img = cv2.imread(random.choice(img_map[ch]))
        if ch_img.shape == (28, 28, 3):
            ch_img = cv2.resize(ch_img, (42, 56))
            ch_img = ch_img // 2
            kernel = np.ones((3, 3), np.uint8)
            ch_img = cv2.erode(ch_img, kernel, iterations=1)

        ch_img = cv2.resize(ch_img, (ch_img.shape[1], rand_scale(ch_img.shape[0], h_scale)))
        ch_img_list.append(ch_img)

        y_adj = rand_scale(ch_img.shape[0], y_shift)
        x_adj = rand_scale(ch_img.shape[1], x_shift)
        x = max(0, x + x_adj)

        loc_list.append((x + ch_img.shape[1] // 2, bg_h // 2 + y_adj))
        x += ch_img.shape[1]

    max_width = loc_list[-1][0] + ch_img_list[-1].shape[1] // 2
    min_top = min(zip(ch_img_list, loc_list), key=lambda x: x[1][1] - x[0].shape[0] // 2)
    max_bottom = max(zip(ch_img_list, loc_list), key=lambda x: x[1][1] + x[0].shape[0] // 2)
    top = (min_top[1][1] - min_top[0].shape[0] // 2)
    bottom = (max_bottom[1][1] + max_bottom[0].shape[0] // 2)
    max_height = bottom - top

    bg_img = cv2.resize(bg_img, (max_width + 2, max_height + 2))  # +2 是 margin
    loc_list = list(map(lambda x: (x[0], x[1] - top), loc_list))

    for ch_img, loc in zip(ch_img_list, loc_list):
        mask = 255 * np.ones(ch_img.shape, ch_img.dtype)
        bg_img = cv2.seamlessClone(ch_img, bg_img, mask, loc, cv2.NORMAL_CLONE)

    cv2.imwrite(write_file, bg_img)


def generate_synthetic_crop():
    os.makedirs(output_image_path, exist_ok=True)

    output_lines = []
    datalen_probs = [sum(datalen_ratio[: i+1]) for i in range(len(datalen_ratio))]  # 欲產生資料長度的機率對應表
    blank_probs = [sum(blank_ratio[: i+1]) for i in range(len(blank_ratio))]  # 欲產生空白個數的機率對應表

    for mode in ['painting', 'handwrite', 'mixing']:
        for i in tqdm.tqdm(range(num_data), desc=mode):
            datalen = datalen_list[get_random_index(datalen_probs)]
            chars = [random.choice(char_list) for _ in range(datalen)]
            label = ''.join(chars)

            for _ in range(blank_list[get_random_index(blank_probs)]):
                chars.insert(random.randrange(0, len(chars) + 1), ' ')

            filename = f'{mode}_{i:06d}'
            output_lines.append(f'{filename},{label}\n')
            generate_crop_image(chars, pth.join(output_image_path, filename + '.jpg'), mode=mode)

    open(output_data_file, 'w').writelines(output_lines)


if __name__ == '__main__':
    generate_synthetic_crop()
