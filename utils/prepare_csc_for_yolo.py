"""
將有 bounding box 的訓練資料整理成 YOLO5 可以訓練的目錄結構及標籤結構。
另外將圖片都轉成灰階並以直方圖均衡化處理。

YOLO5 出處：https://github.com/ultralytics/yolov5
"""
import os
import os.path as pth
import cv2
import tqdm
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


csc_data_path = '../../csc-datasets/public_training_data'
csc_data_label = 'data/public_training_data.csv'
yolo_data_path = '../../csc-datasets/public_training_yolo'


def equalizeHistogram(src, dst):
    img = cv2.imread(src)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img2 = clahe.apply(gray)
    cv2.imwrite(dst, img2)


def prepare_csc_for_yolo():
    # 讀取資料的 bounding box 資訊，轉換成 yolo 的格式
    df = pd.read_csv(csc_data_label)
    df = df[~df['top right x'].isna()]  # 沒有座標資訊的資料不要
    all_data = []
    for _, row in tqdm.tqdm(df.iterrows(), desc='generate label'):
        max_x = max(row['top right x'], row['bottom right x'])
        min_x = min(row['top left x'], row['bottom left x'])
        max_y = max(row['bottom left y'], row['bottom right y'])
        min_y = min(row['top left y'], row['top right y'])
        cx = (max_x + min_x) / 2
        cy = (max_y + min_y) / 2
        img_file = pth.join(csc_data_path, row['filename'] + '.jpg')
        img = Image.open(img_file)
        yolo_x = cx / img.width
        yolo_y = cy / img.height
        yolo_w = abs(max_x - min_x) / img.width  # 有些圖片的標籤左右相反
        yolo_h = abs(max_y - min_y) / img.height

        label = f'0 {yolo_x:.6f} {yolo_y:.6f} {yolo_w:.6f} {yolo_h:.6f}'
        all_data.append((row['filename'], label))

    # 為 yolo5 準備好可以訓練的目錄結構
    img_trn_path = pth.join(yolo_data_path, 'images', 'train')
    img_val_path = pth.join(yolo_data_path, 'images', 'val')
    lbl_trn_path = pth.join(yolo_data_path, 'labels', 'train')
    lbl_val_path = pth.join(yolo_data_path, 'labels', 'val')

    os.makedirs(img_trn_path, exist_ok=True)
    os.makedirs(img_val_path, exist_ok=True)
    os.makedirs(lbl_trn_path, exist_ok=True)
    os.makedirs(lbl_val_path, exist_ok=True)

    # 將資料分割成訓練集及驗證集
    train, val = train_test_split(all_data, test_size=0.1, random_state=123)

    # 為 yolo5 準備好可以訓練的影像及標籤資料
    for name, label in tqdm.tqdm(train, desc='saving train data'):
        open(pth.join(lbl_trn_path, name + '.txt'), 'w').write(label)
        equalizeHistogram(pth.join(csc_data_path, name + '.jpg'), pth.join(img_trn_path, name + '.jpg'))

    for name, label in tqdm.tqdm(val, desc='saving val data'):
        open(pth.join(lbl_val_path, name + '.txt'), 'w').write(label)
        equalizeHistogram(pth.join(csc_data_path, name + '.jpg'), pth.join(img_val_path, name + '.jpg'))


if __name__ == '__main__':
    prepare_csc_for_yolo()
