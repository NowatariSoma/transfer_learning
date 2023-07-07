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


class DogDataset(data.Dataset):
    """
    犬種のDataseクラス。
    PyTorchのDatasetクラスを継承させる。
    
    Attrbutes
    ---------
    file_list: list
        画像のファイルパスを格納したリスト
    classes: list
        犬種のラベル名
    transform: object
        前処理クラスのインスタンス
    phase: 'train' or 'valid'
        学習か検証化を設定
    """
    def __init__(self, file_list, classes, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.classes = classes
        self.phase = phase
    
    def __len__(self):
        """
        画像の枚数を返す
        """
        return len(self.file_list)
    
    def __getitem__(self, index):
        """
        前処理した画像データのTensor形式のデータとラベルを取得
        """
        # 指定したindexの画像を読み込む
        img_path = self.file_list[index]
        # print(img_path)
        img = Image.open(img_path)
        
        # 画像の前処理を実施
        img_transformed = self.transform(img, self.phase)

        # 画像ラベルをファイル名から抜き出す
        label = self.file_list[index].split('/')[2][10:]
        
        # print(label)
        # ラベル名を数値に変換
        label = self.classes.index(label)
        
        return img_transformed, label
