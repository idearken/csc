"""
從 public training data 裡挑出一些沒有倒置的序號圖片，然後以本程式將字元分割出來，之後會用來合成序號
"""
import os
import os.path as pth
import glob
import cv2
import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict


crop_image_path = '../csc-datasets/public_training_crop_rev/rotate180-rev'
label_file = '../data/public_training_data.csv'
output_path = '../csc-datasets/char_images'

cnt_map = defaultdict(lambda: 0)
X, Y, W, H = 0, 1, 2, 3
margin_x = 5


def pred_num_char(roi):
    ratio = roi[H] / roi[W]
    if ratio > 1.2: return 1
    elif ratio > 0.642: return 2
    elif ratio > 0.45: return 3
    elif ratio > 0.4: return 4
    else: return 5


def cut_chars(file, label):
    org = cv2.imread(file)
    img = org.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=3)

    kernel = np.ones((1, 5), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)

    roi_list = []
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    for c in cnts:
        roi_list.append(cv2.boundingRect(c))

    roi_list = list(filter(lambda x: x[H] / org.shape[0] > 0.4, roi_list))  # ROI 高度必須超過原圖高度的 40%
    roi_list = sorted(roi_list, key=lambda x: x[X])  # 從左排到右
    pred_total = sum([pred_num_char(roi) for roi in roi_list])  # 預測字元總數
    if pred_total != len(label):  # 預測字元總數與實際總數相同才輸出字元圖片
        return

    curr = 0
    for roi in roi_list:
        n = pred_num_char(roi)
        lbl = label[curr: curr + n]
        path = pth.join(output_path, lbl)
        os.makedirs(path, exist_ok=True)
        name = f'{lbl}_{cnt_map[lbl]}_{pth.splitext(pth.basename(file))[0]}.png'
        ch_img = org[roi[Y]: roi[Y] + roi[H], max(0, roi[X] - margin_x): roi[X] + roi[W] + margin_x]
        cv2.imwrite(pth.join(path, name), ch_img)
        cnt_map[lbl] += 1
        curr += n


def cut_char_images():
    df = pd.read_csv(label_file)
    files = glob.glob(pth.join(crop_image_path, '*.jpg'))
    for f in tqdm.tqdm(files):
        filename = pth.splitext(pth.basename(f))[0]
        label = df[df['filename'] == filename].iloc[0]['label']
        cut_chars(f, label)


if __name__ == '__main__':
    cut_char_images()