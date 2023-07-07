# ライブラリの読み込み
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

from preprocess import make_filepath_list
from utils import ImageTransform
from dataset import DogDataset

# 画像データへのファイルパスを格納したリストを取得する
train_file_list, valid_file_list = make_filepath_list()

print('学習データ数 : ', len(train_file_list))
# 先頭3件だけ表示
# print(train_file_list[:3])

print('検証データ数 : ', len(valid_file_list))
# 先頭3件だけ表示
# print(valid_file_list[:3])

# 動作確認
img = Image.open('./images/n02085620-Chihuahua/n02085620_199.jpg')

# リサイズ先の画像サイズ
resize = 300

# 今回は簡易的に(0.5, 0.5, 0.5)で標準化
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

transform = ImageTransform(resize, mean, std)
img_transformed = transform(img, 'train')

# plt.imshow(img)
# plt.show()

# plt.imshow(img_transformed.numpy().transpose((1, 2, 0)))
# plt.show()

# 動作確認
# クラス名
dog_classes = [
    'Chihuahua',  'Shih-Tzu',
    'borzoi', 'Great_Dane', 'pug'
]

# リサイズ先の画像サイズ
resize = 300

# 今回は簡易的に(0.5, 0.5, 0.5)で標準化
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

# Datasetの作成
train_dataset = DogDataset(
    file_list=train_file_list, classes=dog_classes,
    transform=ImageTransform(resize, mean, std),
    phase='train'
)

valid_dataset = DogDataset(
    file_list=valid_file_list, classes=dog_classes,
    transform=ImageTransform(resize, mean, std),
    phase='valid'
)

# バッチサイズの指定
batch_size = 64

# DataLoaderを作成
train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

valid_dataloader = data.DataLoader(
    valid_dataset, batch_size=32, shuffle=False)

# 辞書にまとめる
dataloaders_dict = {
    'train': train_dataloader, 
    'valid': valid_dataloader
}

index = 0
print(train_dataset.__getitem__(index)[0].size())
# print(train_dataset.__getitem__(index)[1])

# 学習済みの重みを使用
use_pretrained = True

# モデルをロード
net = models.vgg16(pretrained=use_pretrained)

print(net)

print('変更前 : ', net.classifier[6])

net.classifier[6] = nn.Linear(in_features=4096, out_features=5)

print('変更前 : ', net.classifier[6])

criterion = nn.CrossEntropyLoss()

list(net.classifier[6].named_parameters())

# 出力内容の確認
for name, param in net.named_parameters():
    print('name : ', name)

# 転移学習で学習させるパラメータを、変数params_to_updateに格納
params_to_update = []

# 学習させるパラメータ名
update_param_names = ['classifier.6.weight', 'classifier.6.bias']

# 学習させるパラメータ以外は勾配計算をなくし、変化しないように設定
for name, param in net.named_parameters():
    
    if name in update_param_names:
        param.requires_grad = True
        params_to_update.append(param)
        print('name : ', name)
    else:
        param.requires_grad = False

# params_to_updateの中身を確認
print('--------------------')
print(params_to_update)

optimizer = optim.SGD(params_to_update, lr=0.01)

# エポック数
num_epochs = 30

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    print('-------------')
    
    for phase in ['train', 'valid']:
        if phase == 'train':
            # 学習モードに設定
            net.train()
        else:
            # 訓練モードに設定
            net.eval()
            
        # epochの損失和
        epoch_loss = 0.0
        # epochの正解数
        epoch_corrects = 0
        
        for inputs, labels in dataloaders_dict[phase]:

            # optimizerを初期化
            optimizer.zero_grad()
            
            # 学習時のみ勾配を計算させる設定にする
            with torch.set_grad_enabled(phase == 'train'):
                
                outputs = net(inputs)
                
                # 損失を計算
                loss = criterion(outputs, labels)
                
                # ラベルを予測
                _, preds = torch.max(outputs, 1)
                
                # 訓練時は逆伝搬の計算
                if phase == 'train':
                    # 逆伝搬の計算
                    loss.backward()
                    
                    # パラメータ更新
                    optimizer.step()
                    
                # イテレーション結果の計算
                # lossの合計を更新
                # PyTorchの仕様上各バッチ内での平均のlossが計算される。
                # データ数を掛けることで平均から合計に変換をしている。
                # 損失和は「全データの損失/データ数」で計算されるため、
                # 平均のままだと損失和を求めることができないため。
                epoch_loss += loss.item() * inputs.size(0)
                
                # 正解数の合計を更新
                epoch_corrects += torch.sum(preds == labels.data)

        # epochごとのlossと正解率を表示
        epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
        epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))