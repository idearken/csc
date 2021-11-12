"""
將訓練資料分割成 5 個 fold，連同 label 儲存在 kfold_data.txt 裡
"""
import os.path as pth
import glob
import pandas as pd
from sklearn.model_selection import KFold


data_path = '../../csc-datasets/public_training_crop'
data_file = '../csc-datasets/public_training_data.csv'
num_fold = 5  # 分幾個 fold
output_file = '../data/kfold_data.txt'  # 輸出列表檔案


def generate_kfold_list():
    all_data = []
    df = pd.read_csv(data_file)
    for f in glob.glob(pth.join(data_path, '*.jpg')):
        filename = pth.splitext(pth.basename(f))[0]
        df2 = df[df['filename'] == filename]
        if len(df2) > 0:
            label = df2.iloc[0]['label']
            all_data.append((filename, label))
        else:
            print(f'Cannot find records in the csv file. file={f}')

    all_data = pd.Series(all_data)
    lines = []
    kfold = KFold(n_splits=num_fold, shuffle=True, random_state=369)
    for nth, (trn_idx, val_idx) in enumerate(kfold.split(all_data)):
        for filename, label in all_data[val_idx]:
            lines.append('{} {} {}\n'.format(nth, filename, label))

    open(output_file, 'w', encoding='utf8').writelines(lines)


if __name__ == '__main__':
    generate_kfold_list()
