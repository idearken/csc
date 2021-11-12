"""
用大量合成資料 (synthetic_training_1000k) 預訓練 Efficientnet 模型。
模型目標是識別圖片中是否有出現 ground truth 裡的字元，是一個 multi-label 任務。
"""
import sys
import os
import os.path as pth
import shutil
import tqdm
from PIL import Image
import numpy as np
import multiprocessing
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import imgaug.augmenters as iaa
import pytorch_lightning as pl
from backbone_efficientnet import EfficientNet


class cfg(object):

    # 模型相關參數
    char_classes = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'  # 欲分辦的字元。注意：沒有小寫字母及大寫「I」及「O」
    num_class = len(char_classes)  # 欲判別的類別總數

    input_height = 80  # 模型接受的輸入影像高度
    input_width = 480  # 模型接受的輸入影像寬度

    # 訓練相關參數
    train_batch_size = 48  # 訓練時批次大小
    val_batch_size = 256  # 評估時批次大小
    lr = 1e-4  # 學習率
    gpus = 1  # GPU 的數量；None for CPU
    val_check_interval = 0.2  # 每 0.2 個 epoch 做一次 validation
    num_workers_dataloader = multiprocessing.cpu_count() if sys.gettrace() is None else 0  # 在 debug 模式要設成 0 才行

    # 程式相關參數
    train_image_root = '../csc-datasets/synthetic_crop/synthetic_1000k'  # 合成序號圖片的放置目錄
    train_label_file = 'data/synthetic_training_1000k.csv'  # 合成序號的標籤檔
    num_val = 50000  # train_label_file 前面 num_val 筆資料做為 val data

    test_image_root = '../csc-datasets/public_training_crop'  # 將 public training data 當作測試資料
    test_label_file = 'data/kfold_data.txt'  # 測試資料標籤檔


class SynDataset(torch.utils.data.Dataset):

    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        file, target = self.data[index]
        return self.transform(Image.open(file)), torch.FloatTensor(target)

    def __len__(self):
        return len(self.data)


class SynDataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()

        # 將訓練資料載入
        all_data = []
        for line in open(cfg.train_label_file, 'r').readlines():
            filename, label = line.strip().split(',')
            pathfile = pth.join(cfg.train_image_root, filename + '.jpg')
            target = [int(ch in label) for ch in cfg.char_classes]
            all_data.append((pathfile, target))

        self.train_data = all_data[cfg.num_val: ]
        self.val_data = all_data[: cfg.num_val]

        # data augmentation。參考：https://github.com/aleju/imgaug
        aug_seq = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.Rotate(rotate=(180, 180))),  # 有一半的機率會旋轉 180 度
            iaa.SomeOf((1, 4), [  # 隨機挑 1 ~ 4 個
                iaa.ScaleX((0.8, 1.2)),  # 寬度縮放倍數
                iaa.GaussianBlur((0.0, 4.0)),  # 高斯模糊
                iaa.CLAHE((1, 8)),  # Contrast Limited Adaptive Histogram Equalization
                iaa.LogContrast((0.2, 1.4)),  # 調整對比
                iaa.CoarseDropout(p=0.1, size_percent=0.1),  # 有 10% 的面積會被遮罩
                iaa.JpegCompression(compression=(20, 30)),  # JPEG 壓縮 70% ~ 80%
            ])
        ])

        self.train_transform = T.Compose([
            T.Lambda(lambda img: Image.fromarray(aug_seq.augment_image(np.array(img)))),  # 使用 imgaug 提供的功能
            T.Grayscale(num_output_channels=3),  # 轉換成灰階
            T.Resize((cfg.input_height, cfg.input_width)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

        self.val_transform = T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize((cfg.input_height, cfg.input_width)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

    def train_dataloader(self):
        ds = SynDataset(self.train_data, self.train_transform)
        return DataLoader(ds, batch_size=cfg.train_batch_size, shuffle=True,
                          num_workers=cfg.num_workers_dataloader, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        ds = SynDataset(self.val_data, self.val_transform)
        return DataLoader(ds, batch_size=cfg.val_batch_size, shuffle=False,
                          num_workers=cfg.num_workers_dataloader, drop_last=False, pin_memory=True)

    def test_dataloader(self):
        test_data = []
        for line in open(cfg.test_label_file, 'r').readlines():
            _, filename, label = line.strip().split()
            pathfile = pth.join(cfg.test_image_root, filename + '.jpg')
            target = [int(ch in label) for ch in cfg.char_classes]
            test_data.append((pathfile, target))

        ds = SynDataset(test_data, self.val_transform)
        return DataLoader(ds, batch_size=cfg.val_batch_size, shuffle=False,
                          num_workers=cfg.num_workers_dataloader, drop_last=False, pin_memory=True), [f for f, _ in test_data]


class SynClassifier(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')

        # 修改 efficientnet-b4 部份結構，使當輸入圖像寬高為 480x80 時，輸出 shape=(batch, channel=1792, height=2, width=58)
        self.backbone._blocks[10]._depthwise_conv.stride = (2, 1)
        self.backbone._blocks[22]._depthwise_conv.stride = (2, 1)

        self.classifier = nn.Linear(1792 * 2, cfg.num_class)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, images):
        x = self.backbone(images)  # (batch, channel=1792, height=2, width=58)
        batch, channel, height, width = x.size()
        x = x.view(batch, channel * height, width)  # (batch, feature, width)
        x = x.mean(dim=2)  # (batch, feature)
        x = self.classifier(x)  # (batch, num_class)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=cfg.lr)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        loss = self.criterion(preds, targets)

        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        loss = self.criterion(preds, targets)

        self.log('val_loss', loss, prog_bar=True)

    def validation_epoch_end(self, val_step_outputs):
        if not self.trainer.sanity_checking:
            results = self.eval_dataloader(self.trainer.datamodule.val_dataloader())
            self.log('val_acc', np.mean(results), prog_bar=True)

    @torch.no_grad()
    def eval_dataloader(self, dataloader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.eval()

        results = []
        for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc=f'Calc accuracy'):
            images, targets = [data.to(device) for data in batch]
            preds = self(images)
            preds = torch.sigmoid(preds)
            preds = preds > 0.5
            results.extend(torch.all(preds == targets, dim=1).tolist())

        return results


def main():
    dm = SynDataModule()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename='{epoch} {loss:.4f} {val_loss:.4f} {val_acc:.4f}',  # 不設目錄，預設會儲存在 ./lightning_logs/version_X 裡
        monitor='val_loss',
        save_top_k=1)

    # 注意：不管 val_check_interval 設為多少，checkpoint 的儲存只會在每個 epoch 的第一次 val_check 進行
    trainer = pl.Trainer(gpus=cfg.gpus,
                         max_epochs=10,
                         reload_dataloaders_every_epoch=False,
                         distributed_backend=None if cfg.gpus in [1, None] else 'ddp',
                         callbacks=[checkpoint_callback],
                         val_check_interval=cfg.val_check_interval)

    model = SynClassifier()
    trainer.fit(model, dm)


def eval_test_dataset(ckpt, save_error_path=None):
    dm = SynDataModule()
    model = SynClassifier.load_from_checkpoint(ckpt)
    dataloader, files = dm.test_dataloader()
    results = model.eval_dataloader(dataloader)
    print(f'test_acc={np.mean(results):.2%}')

    if save_error_path is not None:
        os.makedirs(save_error_path, exist_ok=True)
        for corr, file in zip(results, files):
            if not corr:
                shutil.copy(file, pth.join(save_error_path, pth.basename(file)))


def get_backbone_from_ckpt(ckpt, dst):
    model = SynClassifier.load_from_checkpoint(ckpt)
    if not pth.exists(pth.dirname(dst)):
        os.makedirs(pth.dirname(dst))
    torch.save(model.backbone.state_dict(), dst)


if __name__ == '__main__':
    main()
