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

def make_filepath_list():
    """
    学習データ、検証データそれぞれのファイルへのパスを格納したリストを返す
    """
    train_file_list = []
    valid_file_list = []

    for top_dir in os.listdir('./images/'):
        file_dir = os.path.join('./images/', top_dir)
        file_list = os.listdir(file_dir)

        # 各犬種ごとに8割を学習データ、2割を検証データとする
        num_data = len(file_list)
        num_split = int(num_data * 0.8)

        train_file_list += [os.path.join('./images', top_dir, file) for file in file_list[:num_split]]
        valid_file_list += [os.path.join('./images', top_dir, file) for file in file_list[num_split:]]
    
    return train_file_list, valid_file_list

