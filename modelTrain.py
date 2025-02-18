# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : modelTrain.py
# Time       ：2025/2/18 16:53
# Author     ：zhou zihan
# version    ：python 3.6
# Description：
"""
import sys
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandZoomd,
    Compose,
    LoadImaged,
    Resized,
    EnsureTyped,
    EnsureType,
    NormalizeIntensityd,
)
import json
from pathlib import Path
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss,DiceLoss
from monai.data import CacheDataset, DataLoader, decollate_batch
import torch
import shutil
import time
import os
from statistics import mean
from datetime import datetime
from monai.config import print_config
import torch.nn as nn

print_config()

root_dir = './Dataset'
current_time = datetime.now().strftime('%m_%d_%H%M%S')#获取当前时间
# log_dir = os.path.join('runs2D', current_time)#以当前时间创建日志目录名
log_dir = os.path.join('runs2D', current_time)#以当前时间创建日志目录名

if not os.path.exists('runs2D'):
    os.makedirs('runs2D')#
if not os.path.exists(log_dir):
    os.makedirs(log_dir)#
root_logger = logging.getLogger()#日志对象初始化

for h in root_logger.handlers:
    root_logger.removeHandler(h)#移除老的日志对象变量
#设置日志格式和抬头信息
logging.basicConfig(filename=os.path.join(log_dir, 'log.txt'), level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

with open('dataset.json') as f:
    json_data = json.load(f)

trainDataDict = json_data['train']
validDataDict = json_data['test']
labelList = {'NC': 0, 'EMCI': 1, 'LMCI': 2, 'AD': 3}

train_files = [{"image": trainDataDict[file]['path'], "label": np.array([labelList[trainDataDict[file]['label']]]).astype(np.float32)} for file in
               trainDataDict]
val_files = [{"image": validDataDict[file]['path'], "label": np.array([labelList[validDataDict[file]['label']]]).astype(np.float32)} for file in
             validDataDict]

# train_files = train_files[0:int(len(train_files)/4)]
# val_files = val_files[0:int(len(val_files)/4)]


print(train_files)
print(val_files)

#对训练集做的预处理流程
train_transforms = Compose(
    [
        LoadImaged(keys=["image"]),#加载数据，keys参数指定image和label都需要加载
        EnsureChannelFirstd(keys=["image"]),#添加通道维度，并确保通道维度在第一维
        ScaleIntensityd(keys=["image"]),#数据矩阵归一化,标注不需要做 divisor=
        Resized(keys=["image"], spatial_size=(256, 256), mode=('nearest')),
        EnsureTyped(keys=["image", "label"])#将数据类型转化为Tensor
    ]
)
#对验证集做的预处理流程
val_transforms = Compose(
    [
        LoadImaged(keys=["image"]),  # 加载数据，keys参数指定image和label都需要加载
        EnsureChannelFirstd(keys=["image"]),  # 添加通道维度，并确保通道维度在第一维
        ScaleIntensityd(keys=["image"]),  # 数据矩阵归一化,标注不需要做 divisor=
        Resized(keys=["image"], spatial_size=(256, 256), mode=('nearest')),
        EnsureTyped(keys=["image", "label"])  # 将数据类型转化为Tensor
    ]
)
print("return 0")

#创建训练集数据加载器，batch_size可根据显卡情况自行加大。在DataLoader会对数据做预处理
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
# train_ds = Dataset(data=train_files, transform=train_transforms)#CacheDataset一次性加载所有数据到GPU,内存开销大，
#但后续训练块。Dataset训练一份数据加载一份数据，内存开销小，但训练效率底
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
val_loader = DataLoader(val_ds, batch_size=32)
print("return 0")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print('device =',device)
print(torch.cuda.get_device_name(0))
print("return 0")

# 修改输出通道数为 4
out_channel = 4

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=out_channel,  # 修改输出通道数为 4
    channels=(64, 128, 256, 512, 1024),
    strides=(2, 2, 2, 2),
    num_res_units=0,
).to(device)

#如果训练中断或者对训练结果不满意，可以从加载之前保存好的模型继续训练
# path_checkpoint = './runs2D/12_25_110226/best_metric_model.pth' # 12_25_110226 恒定的学习率 0.0001
# path_checkpoint = './runs2D/12_26_081035/best_metric_model.pth'# 12_26_081035 变化的学习率【0.001，0.0001】
# path_checkpoint = './runs2D/12_26_083922/best_metric_model.pth'# 12_26_083922 变化的学习率【0.0001，0.00001】
# path_checkpoint = './runs2D/12_26_091708/best_metric_model.pth'# 12_26_091708 恒定的学习率 0.00002
# path_checkpoint = './runs2D/12_27_140540/best_metric_model.pth'# 12_27_140540 恒定的学习率【0.00002】
# path_checkpoint = './runs/12_28_165333/best_metric_model.pth' # 12_28_165333 低深度训练结果 恒定的学习率 0.0002
# path_checkpoint = './runs2D/12_29_063156/best_metric_model.pth' # 12_29_063156 恒定的学习率 0.0002
# path_checkpoint = './runs2D/12_29_094706/best_metric_model.pth' # 12_29_094706 恒定的学习率 0.00002

# path_checkpoint = './runs2D/12_29_105322/best_metric_model.pth'# 12_29_105322 恒定的学习率 0.00002 进入局部最优解
# path_checkpoint = './runs2D/12_29_134054/best_metric_model.pth'# 12_29_134054 恒定的学习率 0.0002 重新训练
# path_checkpoint = './runs2D/01_01_070739/last_metric_model.pth'# 01_01_070739 恒定的学习率 0.00005 重新训练
# path_checkpoint = './runs2D/01_01_094427/best_metric_model.pth'# 01_01_094427 恒定的学习率 0.0001 重新训练
# checkpoint = torch.load(path_checkpoint, map_location=torch.device(device))
# model.load_state_dict(checkpoint)
# path_checkpoint = './runs2D/12_29_134054/best_metric_model.pth'# 12_29_134054 恒定的学习率 0.0002 重新训练
# checkpoint = torch.load(path_checkpoint, map_location=torch.device(device))
# model.load_state_dict(checkpoint)
for name, param in model.named_parameters():
    print(f"Parameter {name} has data type {param.dtype}")
print("return 0")

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)#设置损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)#设置优化器和学习率
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)#设置优化器和学习率 后期验证集dice系数基本不变，收敛于局部最优解
dice_metric = DiceMetric(include_background=True, reduction="mean")#设置验证集评估标准，此处为dice系数
train_dice_metric = DiceMetric(include_background=True, reduction="mean")#设置验证集评估标准，此处为dice系数
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.0001,step_size_up=5,mode="triangular2")
lrs = []

max_epochs = 120#设置最大训练代数
val_interval = 2#设置每隔多少代进行一次验证，数越大越节省开销
best_metric = -1#初始化最佳Dice大小
train_best_metric = -1#初始化最佳Dice大小
best_metric_epoch = -1#初始化最佳Dice的代数
epoch_loss_values = []#用于保存每代的Loss大小
metric_values = []#用于保存每次的Dice大小
train_metric_values = []#用于保存每次train的Dice大小,用于查看是否过拟合
loss_fn = nn.CrossEntropyLoss()
print("return 0")

post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=out_channel)])#后处理，即反转预处理
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=out_channel)])#后处理，即反转预处理
print("return 0")

import numpy as np

transform = AsDiscrete(argmax=True, to_onehot=out_channel)
out = transform(np.array([[[0.0, 1.0]], [[2.0, 3.0]]]))
print(out.shape, out.dtype)

writer = SummaryWriter(os.path.join(log_dir, 'log'))
# 写入并输出日志信息
logging.info(f"\nmodel:{model.__class__.__name__}\nmax epoch:{max_epochs}")
# -----------------------------------------------开始训练与验证
for epoch in range(max_epochs):
    print("==" * 20)
    logging.info("-" * 10)
    logging.info(f"epoch {epoch + 1}/{max_epochs}")

    model.train()
    epoch_loss = 0
    step = 0
    start_time = time.time()
    for batch_data in train_loader:  # 从数据加载器中取出数据
        torch.cuda.empty_cache()  # 清理显存
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        print(inputs.dtype, inputs.shape, torch.max(inputs), torch.min(inputs))
        print(labels.dtype, labels.shape, torch.max(labels), torch.min(labels))
        print(batch_data["label"].dtype)
        optimizer.zero_grad()  # 梯度归0
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)  # 计算损失函数
        # 梯度
        loss.sum().backward()
        # loss = loss_function(outputs, labels)  # 计算预测图与标签的损失
        # loss.backward()  # loss反向转播
        optimizer.step()  # 优化器调整
        epoch_loss += loss.item()

        outputs = torch.sigmoid(outputs)
        outputs = (outputs >= 0.5).float()
        labels = (labels >= 0.5).float()

        outputs = [i for i in decollate_batch(outputs)]
        labels = [i for i in decollate_batch(labels)]

        train_dice_metric(y_pred=outputs, y=labels)  # 计算Dice

    train_metric_mean = train_dice_metric.aggregate().item()  # 计算平均Dice数值
    _train_metric = train_dice_metric.aggregate(reduction="mean_batch").to('cpu').numpy().tolist()  # 计算所有标注类的Dice
    train_metric = [round(i, 4) for i in _train_metric]  # list元素取4位小数
    train_metric_mean = train_dice_metric.aggregate().item()  # 计算平均Dice数值
    train_dice_metric.reset()  # 重置Dice
    train_metric_values.append(train_metric_mean)
    logging.info(
        f"current train epoch: {epoch + 1} current train dice: {train_metric}"
        f"\ncurrent mean train dice: {train_metric_mean:.4f}"
    )

    epoch_loss /= step  # 计算每个epoch的平均dice
    epoch_loss_values.append(epoch_loss)
    logging.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    lrs.append(optimizer.param_groups[0]["lr"])
    # scheduler.step()

    if (epoch + 1) % val_interval == 0:  # 开始验证
        print("开始验证！")
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                torch.cuda.empty_cache()  # 清理显存
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )

                # val_outputs = model(val_inputs)
                # print(val_labels.shape, val_labels.dtype)
                # val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                # val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                # dice_metric(y_pred=val_outputs, y=val_labels)#计算Dice

                val_outputs = model(val_inputs)
                val_outputs = torch.sigmoid(val_outputs)
                val_outputs = (val_outputs >= 0.5).float()
                val_labels = (val_labels >= 0.5).float()
                print(val_labels.shape, val_labels.dtype)

                val_outputs = [i for i in decollate_batch(val_outputs)]
                val_labels = [i for i in decollate_batch(val_labels)]

                dice_metric(y_pred=val_outputs, y=val_labels)  # 计算Dice

            metric_mean = dice_metric.aggregate().item()  # 计算平均Dice数值
            _metric = dice_metric.aggregate(reduction="mean_batch").to('cpu').numpy().tolist()  # 计算所有标注类的Dice
            metric = [round(i, 4) for i in _metric]  # list元素取4位小数
            # metric_mean = dice_metric.aggregate().item()#计算平均Dice数值
            dice_metric.reset()  # 重置Dice

            metric_values.append(metric_mean)
            if metric_mean > best_metric:  # 如果当前Dice比之前的最佳Dice高，则保存模型
                best_metric = metric_mean
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(log_dir, f"best_metric_model.pth"))
                logging.info("saved new best metric model")
            logging.info(
                f"current epoch: {epoch + 1} current dice: {metric}"
                f"\ncurrent mean dice: {metric_mean:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )
    if train_metric_mean > train_best_metric:  # 如果当前Dice比之前的最佳Dice高，则保存模型
        train_best_metric = train_metric_mean
        # best_metric_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(log_dir, f"train_best_metric_model.pth"))
        logging.info("saved new train best metric model")

    torch.save(model.state_dict(), os.path.join(log_dir, f"last_metric_model.pth"))
    logging.info("saved the lastest metric model!")
    end_time = time.time()
    logging.info(f"time consuming: {end_time - start_time:.2f}s")
    logging.info(f"epoch = {epoch + 1}, Learning Rate = {optimizer.param_groups[0]['lr']}")

logging.info(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()
print("return 0")

import matplotlib.pyplot as plt
plt.figure("train", (12, 6))

plt.subplot(1, 3, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)

plt.subplot(1, 3, 2)
plt.title("Val DSC")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)

plt.subplot(1, 3, 3)
plt.title("Train DSC")
x = [i + 1 for i in range(len(train_metric_values))]
y = train_metric_values
plt.xlabel("epoch")
plt.plot(x, y)

# plt.subplot(1, 3, 3)
# plt.title("lrs per epoch")
# x = [(i + 1) for i in range(len(lrs))]
# y = lrs
# plt.xlabel("epoch")
# plt.plot(x, y)

plt.show()
print("return 0")

