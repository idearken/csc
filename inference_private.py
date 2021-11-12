"""
20 個模型集成預測，採用投票制，最多模型預測的答案為最終答案，若票數相同，以效能較高的模型勝出。共包含下列模型 (以效能排序)
    1. Inception_v3 + Transformer (fold 0 ~ 4)
    2. EfficientNet-b4 + Transformer (fold 0 ~ 4)
    3. VGG16 + Transformer (fold 0 ~ 4)
    4. DenseNet201 + Transformer (fold 0 ~ 4)

以上每個模型的訓練資料皆使用 public training data，再加上 public testing data (240 張人工標籤，5664 張 pseudo label)

訓練結果皆取最佳「字元準確度」，也就是「中鋼序號」裡的每個字元分開統計的準確度。

最終 Private testing score: 69.00189 (共 9915 題)
"""
from collections import defaultdict
import train_densenet_transformer as dsn
import train_efficientnet_transformer as eff
import train_inception_transformer as icp
import train_vgg_transformer as vgg


dsn_ckpt0 = 'weights/dsn/nfold=0 epoch=75 train_loss=0.0191 val_loss=0.0271 sn_acc=0.9705 ch_acc=0.99521.ckpt'
dsn_ckpt1 = 'weights/dsn/nfold=1 epoch=65 train_loss=0.0171 val_loss=0.0357 sn_acc=0.9677 ch_acc=0.99421.ckpt'
dsn_ckpt2 = 'weights/dsn/nfold=2 epoch=0 train_loss=0.0176 val_loss=0.0241 sn_acc=0.9713 ch_acc=0.99531.ckpt'
dsn_ckpt3 = 'weights/dsn/nfold=3 epoch=55 train_loss=0.0184 val_loss=0.0280 sn_acc=0.9685 ch_acc=0.99514.ckpt'
dsn_ckpt4 = 'weights/dsn/nfold=4 epoch=3 train_loss=0.0194 val_loss=0.0341 sn_acc=0.9657 ch_acc=0.99361.ckpt'

eff_ckpt0 = 'weights/eff/nfold=0 epoch=6 train_loss=0.0151 val_loss=0.0312 sn_acc=0.9713 ch_acc=0.99550.ckpt'
eff_ckpt1 = 'weights/eff/nfold=1 epoch=3 train_loss=0.0164 val_loss=0.0336 sn_acc=0.9682 ch_acc=0.99467.ckpt'
eff_ckpt2 = 'weights/eff/nfold=2 epoch=4 train_loss=0.0137 val_loss=0.0258 sn_acc=0.9738 ch_acc=0.99605.ckpt'
eff_ckpt3 = 'weights/eff/nfold=3 epoch=6 train_loss=0.0137 val_loss=0.0286 sn_acc=0.9721 ch_acc=0.99565.ckpt'
eff_ckpt4 = 'weights/eff/nfold=4 epoch=7 train_loss=0.0134 val_loss=0.0364 sn_acc=0.9715 ch_acc=0.99441.ckpt'

icp_ckpt0 = 'weights/icp/nfold=0 epoch=9 train_loss=0.0092 val_loss=0.0310 sn_acc=0.9729 ch_acc=0.99544.ckpt'
icp_ckpt1 = 'weights/icp/nfold=1 epoch=9 train_loss=0.0081 val_loss=0.0361 sn_acc=0.9705 ch_acc=0.99462.ckpt'
icp_ckpt2 = 'weights/icp/nfold=2 epoch=4 train_loss=0.0103 val_loss=0.0251 sn_acc=0.9743 ch_acc=0.99622.ckpt'
icp_ckpt3 = 'weights/icp/nfold=3 epoch=1 train_loss=0.0100 val_loss=0.0289 sn_acc=0.9743 ch_acc=0.99585.ckpt'
icp_ckpt4 = 'weights/icp/nfold=4 epoch=6 train_loss=0.0078 val_loss=0.0378 sn_acc=0.9680 ch_acc=0.99455.ckpt'

vgg_ckpt0 = 'weights/vgg/nfold=0 epoch=9 train_loss=0.0108 val_loss=0.0320 sn_acc=0.9699 ch_acc=0.99519.ckpt'
vgg_ckpt1 = 'weights/vgg/nfold=1 epoch=6 train_loss=0.0117 val_loss=0.0338 sn_acc=0.9680 ch_acc=0.99450.ckpt'
vgg_ckpt2 = 'weights/vgg/nfold=2 epoch=1 train_loss=0.0124 val_loss=0.0280 sn_acc=0.9702 ch_acc=0.99556.ckpt'
vgg_ckpt3 = 'weights/vgg/nfold=3 epoch=3 train_loss=0.0129 val_loss=0.0289 sn_acc=0.9699 ch_acc=0.99545.ckpt'
vgg_ckpt4 = 'weights/vgg/nfold=4 epoch=7 train_loss=0.0101 val_loss=0.0387 sn_acc=0.9674 ch_acc=0.99392.ckpt'

private_testing_crop = '../csc-datasets/private_data_v2_crop'
submission_file = 'submissions/submission_private.csv'


results_list = list()

results_list.append((dsn.predict_questions(dsn_ckpt0, private_testing_crop), 1.01))
results_list.append((dsn.predict_questions(dsn_ckpt1, private_testing_crop), 1.01))
results_list.append((dsn.predict_questions(dsn_ckpt2, private_testing_crop), 1.01))
results_list.append((dsn.predict_questions(dsn_ckpt3, private_testing_crop), 1.01))
results_list.append((dsn.predict_questions(dsn_ckpt4, private_testing_crop), 1.01))

results_list.append((eff.predict_questions(eff_ckpt0, private_testing_crop), 1.03))
results_list.append((eff.predict_questions(eff_ckpt1, private_testing_crop), 1.03))
results_list.append((eff.predict_questions(eff_ckpt2, private_testing_crop), 1.03))
results_list.append((eff.predict_questions(eff_ckpt3, private_testing_crop), 1.03))
results_list.append((eff.predict_questions(eff_ckpt4, private_testing_crop), 1.03))

results_list.append((icp.predict_questions(icp_ckpt0, private_testing_crop), 1.04))
results_list.append((icp.predict_questions(icp_ckpt1, private_testing_crop), 1.04))
results_list.append((icp.predict_questions(icp_ckpt2, private_testing_crop), 1.04))
results_list.append((icp.predict_questions(icp_ckpt3, private_testing_crop), 1.04))
results_list.append((icp.predict_questions(icp_ckpt4, private_testing_crop), 1.04))

results_list.append((vgg.predict_questions(vgg_ckpt0, private_testing_crop), 1.02))
results_list.append((vgg.predict_questions(vgg_ckpt1, private_testing_crop), 1.02))
results_list.append((vgg.predict_questions(vgg_ckpt2, private_testing_crop), 1.02))
results_list.append((vgg.predict_questions(vgg_ckpt3, private_testing_crop), 1.02))
results_list.append((vgg.predict_questions(vgg_ckpt4, private_testing_crop), 1.02))


answers = defaultdict(lambda: [])
for results, score in results_list:
    for filename, label in results:
        answers[filename].append((label, score))

outputs = []
for filename, preds in answers.items():
    vote = defaultdict(lambda: 0)
    for label, score in preds:
        vote[label] += score
    pred = max(vote.items(), key=lambda x: x[1])
    outputs.append((filename, pred[0], pred[1]))

outputs = sorted(outputs, key=lambda x: x[0])


with open(submission_file, 'w') as f:
    f.write('id,text\n')
    for filename, label, _ in outputs:
        f.write(f'{filename},{label}\n')
