"""
模型主架構：Pretrained EfficientNet-b4 + Transformer
訓練資料：public training data + public testing data (240 張人工標籤，5664 張 pseudo label)
"""
import sys
import os.path as pth
import glob
import tqdm
import math
from PIL import Image
from munch import Munch
import numpy as np
import multiprocessing
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import imgaug.augmenters as iaa
import pytorch_lightning as pl
from collections import defaultdict
from backbone_efficientnet import EfficientNet


class cfg(object):

    # 模型相關參數
    char_classes = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'  # 欲分辦的字元。注意：沒有小寫字母及大寫「I」及「O」
    sos = torch.tensor([len(char_classes) + 0])  # Start Of Sentence
    eos = torch.tensor([len(char_classes) + 1])  # End Of Sentence
    pad = len(char_classes) + 2  # Padding, 用來填充同一 batch 裡較短的序號
    num_class = len(char_classes) + 3  # 欲判別的類別總數。加 3 是因為多了 sos, eos, pad
    inference_max_len = 13  # 推論時最大長度 (序號最長 12 碼，再加上 EOS)

    input_height = 80  # 模型接受的輸入影像高度
    input_width = 480  # 模型接受的輸入影像寬度
    d_model = 256  # Transformer 接受的 embed 大小
    positional_encoding_len = 58  # 設定為 Transformer 輸入 (58) 與輸出 (12) 最大長度

    # 訓練相關參數
    train_batch_size = 48  # 訓練時批次大小
    val_batch_size = 256  # 評估時批次大小
    lr = 5e-5  # 學習率
    gpus = 1  # GPU 的數量；None for CPU
    check_val_every_n_epoch = 1  # 每幾個 epoch 做一次 validation
    num_workers_dataloader = multiprocessing.cpu_count() if sys.gettrace() is None else 0  # 在 debug 模式要設成 0 才行

    # 程式相關參數
    dataset_root = '../csc-datasets/public_training_crop'  # 已截切過的「中鋼序號」圖片目錄
    kfold_list_file = 'data/kfold_data.txt'  # 訓練資料共分 5 個 fold，已事先劃分好放在此檔裡

    pretrained_model = 'weights/efficientnet_synthetic_1000k.pt'  # 已事先用「合成序號」預訓練過的 backbone 權重

    handwrite_list_file = 'data/handwrite_list.txt'  # 「中鋼序號」圖片裡有少數手寫序號，不容易判別，因此特別挑出來，用來觀察手寫序號的準確率
    public_testing_crop = '../csc-datasets/public_testing_crop'  # 已截切過的 public testing data 目錄

    public_testing_pseudo = 'data/public_testing_pseudo.csv'  # 包含 240 張人工標籤，5664 張 pseudo label (有 96 張圖人眼無法辦識直接丟棄)
    low_score_multiply = 10  # 由於 score 較低的標籤數量很少，因此將這些資料乘上這個倍數，放大這些標籤的數量
    low_score_threshold = 19  # 分數多少以下算低分


class CSCDataset(torch.utils.data.Dataset):

    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        file, target = self.data[index]
        return self.transform(Image.open(file)), torch.LongTensor(target)

    def __len__(self):
        return len(self.data)


class CSCDataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()

        self.char2index = {c: i for i, c in enumerate(cfg.char_classes)}
        self.index2char = {i: c for c, i in self.char2index.items()}

        # 將訓練資料依 fold 載入
        self.fold_map = defaultdict(lambda: [])
        for line in open(cfg.kfold_list_file, 'r', encoding='utf8').readlines():
            fold, file, label = line.strip().split()
            file = pth.join(cfg.dataset_root, file + '.jpg')
            target = [self.char2index[c] for c in label]
            self.fold_map[int(fold)].append((file, target))

        # 將 public testing pseudo 資料載入
        self.pseudo_data = []
        for line in open(cfg.public_testing_pseudo).readlines():
            filename, pseudo_label, score = line.strip().split(',')
            file = pth.join(cfg.public_testing_crop, filename + '.jpg')
            target = [self.char2index[c] for c in pseudo_label]
            if float(score) < cfg.low_score_threshold:
                self.pseudo_data.extend([(file, target)] * cfg.low_score_multiply)
            else:
                self.pseudo_data.append((file, target))

        self.num_fold = len(self.fold_map)  # fold 總數（共 5 個）
        self.curr_fold_index = 0  # 目前使用的 fold (current for val, others for train)

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

    @staticmethod
    def collate_fn(batch):
        images, targets = zip(*batch)
        max_len = max(targets, key=lambda x: len(x)).size(0)
        images = torch.stack(images, 0)
        targets = [torch.cat([cfg.sos, t, cfg.eos, torch.full((max_len - len(t),), cfg.pad)]) for t in targets]
        targets = torch.stack(targets, 0)
        return images, targets

    def set_fold_index(self, fold_index):
        self.curr_fold_index = fold_index

    def target2label(self, target):
        label = []
        for t in target:
            if t == cfg.sos:
                continue
            if t == cfg.eos or t == cfg.pad:
                break
            label.append(self.index2char[t.item()])
        return ''.join(label)

    def train_dataloader(self):
        # 將目前 fold 以外的其它 fold 整合成訓練資料
        train_data = []
        for fold, lst in self.fold_map.items():
            if fold != self.curr_fold_index:
                train_data.extend(lst)

        # 再加入 pseudo data
        train_data.extend(self.pseudo_data)

        ds = CSCDataset(train_data, self.train_transform)
        return DataLoader(ds, batch_size=cfg.train_batch_size, shuffle=True, collate_fn=self.collate_fn,
                          num_workers=cfg.num_workers_dataloader, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        val_data = self.fold_map[self.curr_fold_index]
        ds = CSCDataset(val_data, self.val_transform)
        return DataLoader(ds, batch_size=cfg.val_batch_size, shuffle=False, collate_fn=self.collate_fn,
                          num_workers=cfg.num_workers_dataloader, drop_last=False, pin_memory=True)

    def val_dataloader_source(self):
        return self.fold_map[self.curr_fold_index]

    def inference_dataloader(self, image_path):
        image_files = glob.glob(pth.join(image_path, '*.jpg'))
        ds = CSCDataset([(f, []) for f in image_files], self.val_transform)
        return DataLoader(ds, batch_size=cfg.val_batch_size, shuffle=False, collate_fn=self.collate_fn,
                          num_workers=cfg.num_workers_dataloader, drop_last=False, pin_memory=True), image_files


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, d_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0)) / d_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:  # accept (sequence, batch, d_model)
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class CSCIdentifier(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.backbone = EfficientNet.from_name('efficientnet-b4')

        # 修改 efficientnet-b4 部份結構，使當輸入圖像寬高為 480x80 時，輸出 shape=(batch, channel=1792, height=2, width=58)
        self.backbone._blocks[10]._depthwise_conv.stride = (2, 1)
        self.backbone._blocks[22]._depthwise_conv.stride = (2, 1)

        self.backbone.load_state_dict(torch.load(cfg.pretrained_model), strict=True)

        self.map_to_seq = torch.nn.Linear(1792 * 2, cfg.d_model)

        self.positional_encoder = PositionalEncoding(d_model=cfg.d_model, dropout_p=0.1, max_len=cfg.positional_encoding_len)
        self.embedding = nn.Embedding(cfg.num_class, cfg.d_model)

        self.transformer = nn.Transformer(
            d_model=cfg.d_model,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dropout=0.1,
        )

        self.dense = nn.Linear(cfg.d_model, cfg.num_class)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

        self.train_loss_sum = 0
        self.train_num_char = 0

    def forward(self, images, targets):
        conv = self.backbone(images)  # (batch, channel=1792, height=2, width=58)
        batch, channel, height, width = conv.size()
        conv = conv.view(batch, channel * height, width)  # (batch, feature, sequence)

        conv = conv.permute(2, 0, 1)  # (sequence, batch, feature)
        seq = self.map_to_seq(conv)  # (sequence, batch, d_model)
        src = self.positional_encoder(seq * math.sqrt(cfg.d_model))  # (sequence, batch, d_model)

        tgt = self.embedding(targets)  # (batch, sequence, d_model)
        tgt = tgt.permute(1, 0, 2)  # (sequence, batch, d_model)
        tgt = self.positional_encoder(tgt * math.sqrt(cfg.d_model))  # (sequence, batch, d_model)

        tgt_mask = self.get_tgt_mask(tgt.size(0))  # (sequence, sequence)

        out = self.transformer(src, tgt, tgt_mask=tgt_mask)  # (sequence, batch, d_model)
        out = self.dense(out)  # (sequence, batch, num_class)

        return out

    def get_tgt_mask(self, size):
        """
        Generates a square matrix where the each row allows one word more to be seen. example as follow (size=5):
        [[0., -inf, -inf, -inf, -inf],
         [0.,   0., -inf, -inf, -inf],
         [0.,   0.,   0., -inf, -inf],
         [0.,   0.,   0.,   0., -inf],
         [0.,   0.,   0.,   0.,   0.]]
        """
        mask = torch.tril(torch.ones(size, size)).to(self.device)  # Lower triangular matrix
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        return mask

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=cfg.lr)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        y_input = targets[:, :-1]  # (batch, sequence)
        y_expected = targets[:, 1:]  # (batch, sequence)

        preds = self(images, y_input)  # (sequence, batch, num_class)

        preds = preds.permute(1, 2, 0)  # (batch, num_class, sequence)
        loss = self.criterion(preds, y_expected)

        self.train_loss_sum += loss
        self.train_num_char += preds.size(0) * preds.size(2)  # total char in this batch

        return loss / (preds.size(0) * preds.size(2))

    def training_epoch_end(self, outs):
        self.log('train_loss', self.train_loss_sum / self.train_num_char, prog_bar=True)
        self.train_loss_sum = 0
        self.train_num_char = 0

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, val_step_outputs):
        if not self.trainer.sanity_checking:
            _, info = self.eval_dataloader(self.trainer.datamodule.val_dataloader())
            self.log('sn_acc', info.sn_acc, prog_bar=True)
            self.log('ch_acc', info.ch_acc, prog_bar=True)
            self.log('val_loss', info.val_loss, prog_bar=True)

    @torch.no_grad()
    def eval_dataloader(self, dataloader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.eval()

        results = []
        loss_sum = 0
        num_char = 0

        for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc=f'Evaluate dataloader'):
            images, targets = [data.to(device) for data in batch]

            y_input = targets[:, :-1]  # (batch, sequence)
            y_expected = targets[:, 1:]  # (batch, sequence)

            preds = self(images, y_input)  # (sequence, batch, num_class)
            preds = preds.permute(1, 2, 0)  # (batch, num_class, sequence)
            loss = self.criterion(preds, y_expected)

            preds = preds.argmax(dim=1)
            for pred, tgt in zip(preds, y_expected):
                results.append((tgt.tolist(), pred.tolist()))

            loss_sum += loss
            num_char += preds.nelement()  # total char in this batch

        sn_acc = sum([int(r[0] == r[1]) for r in results]) / len(results)
        ch_acc = np.mean([int(t == p) for r in results for t, p in zip(*r) if t < len(cfg.char_classes)])

        return results, Munch(sn_acc=sn_acc, ch_acc=ch_acc, val_loss=loss_sum / num_char)

    @torch.no_grad()
    def infer_dataloader(self, dataloader, datamodule):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.eval()

        results = []
        for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc=f'Infer dataloader'):
            images, targets = [data.to(device) for data in batch]
            y_input = targets[:, :-1]  # (batch, sequence)
            for i in range(cfg.inference_max_len):
                preds = self(images, y_input)  # (sequence, batch, num_class)
                preds = preds.permute(1, 2, 0)  # (batch, num_class, sequence)
                preds = preds[:, :, -1]  # 取最後一個預測值 (batch, num_class)
                next_item = preds.argmax(dim=1)
                y_input = torch.cat([y_input, next_item[:, None]], dim=1)

            for tgt in y_input.cpu():
                results.append(datamodule.target2label(tgt))

        return results


def main():
    dm = CSCDataModule()
    for fold_index in range(dm.num_fold):
        dm.set_fold_index(fold_index)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filename=f'nfold={fold_index}' + ' {epoch} {train_loss:.4f} {val_loss:.4f} {sn_acc:.4f} {ch_acc:.5f}',  # 不設目錄，預設會儲存在 ./lightning_logs/version_X 裡
            monitor='val_loss',
            save_top_k=100)

        # 注意：不管 val_check_interval 設為多少，checkpoint 的儲存只會在每個 epoch 的第一次 val_check 進行
        trainer = pl.Trainer(gpus=cfg.gpus,
                             max_epochs=100,
                             reload_dataloaders_every_epoch=False,
                             distributed_backend=None if cfg.gpus in [1, None] else 'ddp',
                             callbacks=[checkpoint_callback],
                             check_val_every_n_epoch=cfg.check_val_every_n_epoch)

        model = CSCIdentifier()
        trainer.fit(model, dm)


def eval_checkpoint(checkpoint_file):
    fold_index = int(pth.basename(checkpoint_file)[6])
    dm = CSCDataModule()
    dm.set_fold_index(fold_index)

    model = CSCIdentifier.load_from_checkpoint(checkpoint_file)
    results, info = model.eval_dataloader(dm.val_dataloader())

    handwrite_list = [line.strip() for line in open(cfg.handwrite_list_file, 'r').readlines()]
    handwrite_rslt = []
    for (real, pred), (file, target) in zip(results, dm.val_dataloader_source()):
        if pth.splitext(pth.basename(file))[0] in handwrite_list:
            handwrite_rslt.append(real == pred)

    print(f'The sn_acc={info.sn_acc:.2%}, ch_acc={info.ch_acc:.3%}, val_loss={info.val_loss:.4f}', end='')
    print(f', handwrite_acc={np.mean(handwrite_rslt):.1%} ({sum(handwrite_rslt)} / {len(handwrite_rslt)})')


def predict_questions(checkpoint_file, question_path):
    dm = CSCDataModule()
    dataloader, image_files = dm.inference_dataloader(question_path)

    print(f'==> {checkpoint_file}')
    model = CSCIdentifier.load_from_checkpoint(checkpoint_file)
    pred_labels = model.infer_dataloader(dataloader, dm)

    results = []
    for file, label in zip(image_files, pred_labels):
        results.append((pth.splitext(pth.basename(file))[0], label))

    return results


if __name__ == '__main__':
    main()
