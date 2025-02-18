# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : train.py
# Time       ：2025/2/17 17:54
# Author     ：zhou zihan
# version    ：python 3.6
# Description：
"""
import datetime
import time
import math
import os
import sys
import random
import timm
import torch
from torch import nn
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from pandas.core.frame import DataFrame


# 随机划分数据集为训练集和测试集
def divide_data():
    config = FConfig()
    csv_data = pd.read_csv('./dataset excel/BKH_data_all.csv', encoding='utf-8')
    test_data_path = './dataset excel/BKH_data_test.csv'
    val_data_path = './dataset excel/BKH_data_val.csv'
    train_data_path = './dataset excel/BKH_data_train.csv'
    all_data_path = './dataset excel/BKH_data_all_divide.csv'

    patient_ID_info = csv_data["patient_ID"].values.tolist()
    patient_ID_info = list(set(patient_ID_info))

    random.seed = config.seed
    random.shuffle(patient_ID_info)
    scale = [7, 1, 2]
    fen = 10
    train_patient = patient_ID_info[:int(len(patient_ID_info) * (scale[0] / fen))]
    val_patient = patient_ID_info[int(len(patient_ID_info) * (scale[0] / fen)):
                                  int(len(patient_ID_info) * ((scale[0] + scale[1]) / fen))]
    test_patient = patient_ID_info[int(len(patient_ID_info) * ((scale[0] + scale[1]) / fen)):]

    train_info = csv_data[csv_data["patient_ID"].isin(train_patient)]
    val_info = csv_data[csv_data["patient_ID"].isin(val_patient)]
    test_info = csv_data[csv_data["patient_ID"].isin(test_patient)]

    train_info = train_info.copy()
    val_info = val_info.copy()
    test_info = test_info.copy()

    train_info.loc[:, "remark"] = "train"
    val_info.loc[:, "remark"] = 'val'
    test_info.loc[:, "remark"] = "test"

    train_info.to_csv(train_data_path, index=False)  # 训练集表格
    val_info.to_csv(val_data_path, index=False)  # 验证集表格
    test_info.to_csv(test_data_path, index=False)  # 测试集表格

    all_data = pd.concat([train_info, val_info, test_info])
    all_data.to_csv(all_data_path, index=False)   # 合并表格

#加载数据集
class GenerateData(Dataset):
    def __init__(self, mode="train"):
        config = FConfig()

        self.mode = mode
        self.classes = ["Benign", "Malignant"]
        self.data = []
        csv_data = pd.read_csv(config.all_data_path, encoding='utf-8')

        train_data = csv_data[csv_data["remark"] == "train"]
        test_data = csv_data[csv_data["remark"] == "val"]

        self.train = train_data
        self.test = test_data

        train_patient_list = list(set(train_data["patient_ID"].values.tolist()))
        test_patient_list = list(set(test_data["patient_ID"].values.tolist()))

        # 排序
        list.sort(train_patient_list)
        list.sort(test_patient_list)

        self.info = {
            "Benign:Malignant": "%d : %d" % (
                len(csv_data[csv_data["label"] == 0]),
                len(csv_data[csv_data["label"] == 1]),),
            "train:val": "%d : %d" % (
                len(train_data),
                len(test_data)),
        }

        self.train_info = {"patient_list": train_patient_list,
                           "patient_len": len(train_patient_list),
                           "data_len": len(train_data),
                           "Benign:Malignant": "%d : %d" % (
                               len(train_data[train_data["label"] == 0]),
                               len(train_data[train_data["label"] == 1]),),
                           }

        self.test_info = {"patient_list": test_patient_list,
                          "patient_len": len(test_patient_list),
                          "data_len": len(test_data),
                          "Benign:Malignant": "%d : %d" % (
                              len(test_data[test_data["label"] == 0]),
                              len(test_data[test_data["label"] == 1]),),
                          }

        #不同情况加载对应数据集
        if mode == "train":
            self.data = train_data.values.tolist()
            self.transform = config.train_transform

        elif mode == "test":
            self.data = test_data.values.tolist()
            self.transform = config.test_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image_path = self.data[item][2]
        label = self.data[item][0]
        img_data = Image.open(image_path).convert('RGB')

        label = np.int64(label)
        label = torch.tensor(label)
        img_data = self.transform(img_data)
        return img_data, label


# 模型调用, pretrined 定义是否需要预训练数据集 imagenet
class ModelGenerator:
    def __init__(self, model_name, class_num=1000):
        self.model = None
        if model_name == "vgg19":
            import torchvision.models as modelss
            temp_model = modelss.vgg19(pretrained=True)
            temp_model.classifier[6] = nn.Linear(in_features=4096, out_features=class_num)
            self.model = temp_model

        elif model_name == "resnet50":
            import torchvision.models as modelss
            temp_model = modelss.resnet50(pretrained=True)
            temp_model.fc = nn.Linear(temp_model.fc.in_features, class_num)
            self.model = temp_model

        elif model_name == "densenet121":
            import torchvision.models as modelss
            temp_model = modelss.densenet121(pretrained=True)
            temp_model.classifier = nn.Linear(temp_model.classifier.in_features, class_num)
            self.model = temp_model

        elif model_name == 'vit':
            temp_model = timm.create_model('vit_base_patch16_224', pretrained=True)
            temp_model.head = nn.Linear(temp_model.head.in_features, class_num)
            self.model = temp_model


# 上下文管理器,日志输出
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout   # 将当前系统输出储存到临时变量
        self.log = open(fileN, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()  # 每次写入后刷新到文件中,防止程序意外结束

    def flush(self):
        self.log.flush()


# 训练时间计算
def minNums(startTime, endTime):
    startTime = datetime.datetime.strptime(startTime, "%Y-%m-%d %H:%M:%S")
    endTime = datetime.datetime.strptime(endTime, "%Y-%m-%d %H:%M:%S")
    total_seconds = (endTime - startTime).total_seconds()
    time = changeTime(total_seconds)
    return str(time)


# 将秒换算成天时分秒
def changeTime(allTime):
    day = 24 * 60 * 60
    hour = 60 * 60
    min = 60
    if allTime < 60:
        return "%d sec" % math.ceil(allTime)
    elif allTime > day:
        days = divmod(allTime, day)
        return "%d days, %s" % (int(days[0]), changeTime(days[1]))
    elif allTime > hour:
        hours = divmod(allTime, hour)
        return '%d hours, %s' % (int(hours[0]), changeTime(hours[1]))
    else:
        mins = divmod(allTime, min)
        return "%d mins, %d sec" % (int(mins[0]), math.ceil(mins[1]))


# top-k准确率计算函数
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# 计算并存储平均值和当前值
class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main(train_loader, test_loader):
    global device, model, model_name
    Fconfig = FConfig()

    # GPU使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型优化与并行-----------------------------------------------------------------------------------
    model = ModelGenerator(Fconfig.model_name, class_num=Fconfig.num_classes).model
    criterion = nn.CrossEntropyLoss().cuda()     # 交叉熵损失函数。.cuda（）转入gpu
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=Fconfig.learning_rate,
                                momentum=Fconfig.momentum,
                                weight_decay=Fconfig.weight_decay)
    # 学习率调整
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model = torch.nn.DataParallel(model, device_ids=list(range(Fconfig.ngpu)))
    model.to(device)
    # ----------------------------------------------------------------------------------------------

    train_acc_all = []
    train_loss_all = []
    test_acc_all = []
    test_loss_all = []

    best_acc = 0
    for epoch in range(Fconfig.start_epoch, Fconfig.epochs):
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print("当前学习率:", now_lr)

        # 训练验证与验证-----------------------------------------------------------------------------------
        model.train()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)
        train_acc_all.append(round(train_acc, 3))
        train_loss_all.append(round(train_loss, 3))

        model.eval()
        test_loss, test_prec = test(test_loader, model, criterion, 'Val')
        test_acc_all.append(round(test_prec, 3))
        test_loss_all.append(round(test_loss, 3))
        # ----------------------------------------------------------------------------------------------

        scheduler.step()

        acc_epoch = test_prec
        if acc_epoch > best_acc:
            best_acc = acc_epoch
            best_model = model

            model_dir = Fconfig.result_path
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            best_acc = round(best_acc, 4)
            model_name = Fconfig.save_model_name + '_best' + ".pth"
            torch.save(best_model, os.path.join(model_dir, model_name))  # 模型保存

    print('训练集损失:\n', train_loss_all)
    print('训练集准确率:\n', train_acc_all)
    print('验证集损失:\n', test_loss_all)
    print('验证集准确率:\n', test_acc_all)

    return model


# 模型训练函数
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top_avg = AverageMeter()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        data_time.update(time.time() - end)

        input, target = input.to(device), target.to(device)

        output = model(input)
        loss = criterion(output, target)

        prec, _ = accuracy(output.data, target, topk=(1, Fconfig.num_classes))
        losses.update(loss.item(), input.size(0))
        top_avg.update(prec[0], input.size(0))

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 损失回传
        optimizer.step()  # 参数更新

        batch_time.update(time.time() - end)
        end = time.time()

        if i % Fconfig.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top_avg))

    print('* 训练集准确率： {top1.avg:.3f}%'.format(top1=top_avg))
    return float(losses.avg), float(top_avg.avg)


# 模型测试函数
def test(val_loader, model, criterion_val, mode='Test'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top_avg = AverageMeter()
    end = time.time()

    class_correct = list(0. for i in range(Fconfig.num_classes))
    class_total = list(0. for i in range(Fconfig.num_classes))

    for i, (input, target) in enumerate(val_loader):
        input, target = input.to(device), target.to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        with torch.no_grad():
            output = model(input_var)
            loss = criterion_val(output, target_var)

            # 计算单分类准确率------------------------------------
            out = F.softmax(output, dim=1)
            pred = torch.max(out, 1)[1]
            if len(target) != 1:
                true_pred = (pred == target).squeeze()
                for ii in range(len(target)):
                    _label = target[ii].item()
                    class_correct[_label] += true_pred[ii].item()
                    class_total[_label] += 1
            else:
                index = target.item()
                class_total[index] += 1
                if pred == target:
                    class_correct[index] += 1
            # -------------------------------------------------

            prec, _ = accuracy(output.data, target, topk=(1, Fconfig.num_classes))
            losses.update(loss.item(), input.size(0))
            top_avg.update(prec[0], input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % Fconfig.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses, top1=top_avg))
    print('-----------------------------------------------------------')
    print('{}准确率： {top1.avg:.3f}%'.format(mode, top1=top_avg))

    for i in range(Fconfig.num_classes):
        print('Accuracy of %5s : %.3f %%' % (
            Fconfig.classes[i], 100 * class_correct[i] / class_total[i]))
    print('-----------------------------------------------------------')

    return float(losses.avg), float(top_avg.avg)


# 用于模型最终测试, 返回真实标签, 预测标签与预测概率分数
def validate(val_loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, (input, target) in enumerate(val_loader):
        input, target = input.to(device), target.to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        with torch.no_grad():
            output = model(input_var)
            out = F.softmax(output, dim=1)
            batch_predict = torch.max(out, 1)[1]

            if torch.cuda.is_available():
                out = out.cuda().data.cpu().numpy()
                batch_labels = target_var.cuda().data.cpu().numpy()
                batch_predict = batch_predict.cuda().data.cpu().numpy()

            if i == 0:
                labels = batch_labels
                predict = batch_predict
                pre_score = out

            else:
                labels = np.append(labels, batch_labels)
                predict = np.append(predict, batch_predict)
                pre_score = np.append(pre_score, out, axis=0)

    return labels, predict, pre_score


# 将列表随机划分为几份
def bisector_list(tabulation: list, num: int):
    new_list = []
    if len(tabulation) >= num:
        remainder = len(tabulation) % num
        if remainder == 0:
            merchant = int(len(tabulation) / num)
            for i in range(1, num + 1):
                if i == 1:
                    new_list.append(tabulation[:merchant])
                else:
                    new_list.append(tabulation[(i - 1) * merchant:i * merchant])
            return new_list
        else:
            merchant = int(len(tabulation) // num)
            remainder = int(len(tabulation) % num)
            for i in range(1, num + 1):
                if i == 1:
                    new_list.append(tabulation[:merchant])
                else:
                    new_list.append(tabulation[(i - 1) * merchant:i * merchant])
                    if int(len(tabulation) - i * merchant) <= merchant:
                        for j in tabulation[-remainder:]:
                            new_list[tabulation[-remainder:].index(j)].append(j)
            return new_list
    else:
        for i in range(1, len(tabulation) + 1):
            tabulation_subset = []
            tabulation_subset.append(tabulation[i - 1])
            new_list.append(tabulation_subset)
        return new_list


# 交叉验证列表划分
def get_pidlist(config):
    csv_data = pd.read_csv(config.data_k_fold, encoding="utf-8")
    patient_ID_info = list(set(csv_data["patient_ID"].values.tolist()))

    patient_ID_info.sort()
    random.shuffle(patient_ID_info)

    patient_ID_info_klist = bisector_list(patient_ID_info, config.k_fold)
    val_csvs, train_csvs = [], []
    for i in range(config.k_fold):
        val_csv = csv_data[csv_data["patient_ID"].isin(patient_ID_info_klist[i])]
        temp = csv_data._append(val_csv)
        train_csv = temp.drop_duplicates(subset=['image_path'], keep=False)
        val_csvs.append(val_csv)
        train_csvs.append(train_csv)
    return val_csvs, train_csvs


# 交叉验证数据集装载
class GenerateCrossValidationData(Dataset):
    def __init__(self, csv_data, mode, config):

        self.mode = mode
        if mode == "train":
            self.data = csv_data.values.tolist()
            self.transform = config.train_transform

        elif mode == "val":
            self.data = csv_data.values.tolist()
            self.transform = config.test_transform

        self.classes = ["Benign", "Malignant"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image_path = self.data[item][2]
        label = self.data[item][0]
        img_data = Image.open(image_path).convert('RGB')

        # 压力测试====================================================
        # brightEnhancer = ImageEnhance.Brightness(img_data)
        # imageBright = brightEnhancer.enhance(0.1)
        # img_data = imageBright
        # contrastEnhancer = ImageEnhance.Contrast(img_data)
        # imageContrast = contrastEnhancer.enhance(2.0)
        # img_data = imageContrast
        # colorEnhancer = ImageEnhance.Color(img_data)
        # imageColor = colorEnhancer.enhance(1.9)
        # img_data = imageColor
        # ===========================================================

        img_data = self.transform(img_data)
        label = np.int64(label)
        label = torch.tensor(label)

        return img_data, label


# 将预测结果存储到表格中
def save_results(model, test_dataset, model_name=None, is_cross=False):
    if not os.path.exists("./CSV Results"):
        os.makedirs("./CSV Results")
    config = FConfig()
    datas = test_dataset.data
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=config.workers)
    labels, predict, pre_score = validate(test_loader, model)
    result = []
    for i, (info) in enumerate(datas):
        result.append([np.int64(info[0]), info[1], info[2], predict[i], pre_score[i][0], pre_score[i][1]])

    csv_data = DataFrame(result, columns=["label", "patient_ID", "image_path", "labels_pred", "Benign", "Malignant"])

    if config.cross_validataion and is_cross:
        path = './K fold excel/' + model_name + '_test_results.csv'
        csv_data.to_csv(path, index=False)
    else:
        csv_data.to_csv(config.test_result, index=False)

    accuracy_score_whole = accuracy_score(y_true=csv_data["label"].values.tolist(),
                                          y_pred=csv_data["labels_pred"].values.tolist())

    # 绘制混淆矩阵：
    conf_matrix = confusion_matrix(csv_data["label"].values.tolist(), csv_data["labels_pred"].values.tolist())

    score = csv_data['Malignant'].values.tolist()
    fpr, tpr, thresholds = roc_curve(csv_data["label"].values.tolist(), score, pos_label=1)
    temp_auc = auc(fpr, tpr)  # 计算AUC

    print("----------------------- 计算结果 -----------------------")
    print("准确率:{0}%".format(round(accuracy_score_whole * 100, 3)))
    print("混淆矩阵:")
    print(conf_matrix)
    print('AUC:', round(temp_auc, 3))
    print("----------------------- 测试结束 -----------------------")

#训练/测试
def train_and_test():
    config = FConfig()
    print("------------------------------- 数据集信息 -------------------------------")
    train_dataset = GenerateData(mode="train")
    test_dataset = GenerateData(mode="test")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)

    print("train:\t*\tBenign:Malignant =", train_dataset.train_info["Benign:Malignant"])
    print("val:\t*\tBenign:Malignant =", train_dataset.test_info["Benign:Malignant"])
    print("all \t*\tBenign:Malignant =", train_dataset.info["Benign:Malignant"])
    print("\t\t*\ttrain:val =", train_dataset.info["train:val"])

    print("-------------------------------- 训练开始 --------------------------------")
    main(train_loader, test_loader)
    print("-------------------------------- 测试部分 --------------------------------")
    model_path = './Experimental Results/' + config.date + '/' + model_name
    net = torch.load(model_path)
    save_results(net.cuda(), test_dataset, model_name=None, is_cross=False)

#交叉验证
def cross_validataion():
    global prec, device
    config = FConfig()
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    val_csvs, train_csvs = get_pidlist(config)

    ACC = []
    model_paths = []
    for k in range(config.start_k_fold, config.k_fold):
        model = ModelGenerator(config.model_name, class_num=config.num_classes).model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.nn.DataParallel(model, device_ids=list(range(Fconfig.ngpu)))#多卡可取列表
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(),
                                    config.learning_rate,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        criterion = nn.CrossEntropyLoss().cuda()

        vvk = val_csvs[k]
        vvk.to_csv('./K fold excel/test-' + str(k + 1) + '.csv', index=False)
        val_datasets = GenerateCrossValidationData(val_csvs[k], "val", config)
        val_loader = DataLoader(val_datasets, batch_size=config.batch_size, shuffle=False, num_workers=config.workers)

        train_datasets = GenerateCrossValidationData(train_csvs[k], "train", config)
        train_loader = DataLoader(train_datasets, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)

        print("val:%d  train:%d" % (len(val_datasets), len(train_datasets)))  # 输出验证和训练信息

        print("================================= kFold=%s ===================================" % (k + 1))

        best_acc = 0
        model_save_path = None
        model_name = None
        for epoch in range(config.start_epoch, config.epochs):

            model.train()
            train(train_loader, model, criterion, optimizer, epoch)
            model.eval()
            val_loss, prec = test(val_loader, model, criterion, 'Test')
            scheduler.step()

            model_name = config.save_model_name + "--" + str(k + 1) + '_fold.pth'

            acc_epoch = prec
            if acc_epoch > best_acc:
                best_acc = acc_epoch
                best_model = model
                model_dir = config.result_path + "/kfold"
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                model_name = config.save_model_name + "--" + str(k + 1) + '_fold.pth'
                model_save_path = os.path.join(model_dir, model_name)
                torch.save(best_model, model_save_path)

        model_paths.append(model_save_path)

        ACC.append(float(best_acc))

        for index, (acc) in enumerate(ACC):
            print("第%d折准确率: %f" % (index + 1, acc))
        if k >= 1:
            print("mean = %f" % np.mean(ACC))

        net_path = model_save_path
        net = torch.load(net_path)
        save_results(net, val_datasets, model_name, is_cross=True)

    for path in model_paths:
         print(path)


# 参数配置类
class FConfig:
    def __init__(self):

        self.divide = True
        self.classes = ["Benign", "Malignant"]
        self.seed = '900108'
        self.date = '20230220'
        self.ngpu = 1
        self.num_classes = 2

        self.cross_validataion = True
        self.k_fold = 5
        self.start_k_fold = 0

        # 数据路径----------------------------------------------------------------------------------------
        self.all_data_path = './dataset excel/BKH_data_all_divide.csv'
        self.data_k_fold = './dataset excel/BKH_data_train.csv'
        # ----------------------------------------------------------------------------------------------

        # 数据预处理--------------------------------------------------------------------------------------
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        # ----------------------------------------------------------------------------------------------

        # 网络参数与优化参数--------------------------------------------------------------------------------
        self.model_name = "resnet50"
        self.batch_size = 64
        self.epochs = 50
        self.start_epoch = 0
        self.print_freq = 20
        self.workers = 0
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.weight_decay = 1e-5
        # ----------------------------------------------------------------------------------------------

        # 结果保存----------------------------------------------------------------------------------------
        dt = datetime.datetime.now()
        self.save_model_name = dt.strftime('%m%d%H%M_') + self.model_name + "_S" + self.date + "_C" + str(
            self.num_classes)
        self.result_path = "./Experimental Results/%s" % self.date
        self.test_result = './CSV Results/BKH_all_test_result_' + self.model_name + '.csv'
        # ----------------------------------------------------------------------------------------------


if __name__ == '__main__':
    Fconfig = FConfig()

    if Fconfig.divide:
        divide_data()
        print('dataset divided finished!')

    # ---------------------------------------------------------------------------
    # dataset = pd.read_csv(Fconfig.all_data_path)
    # train = dataset[dataset['remark'] == 'train']
    # val = dataset[dataset['remark'] == 'val']
    # test = dataset[dataset['remark'] == 'test']
    # print('train良性有：', len(set(train[train['label'] == 0]['patient_ID'])))
    # print('train恶性有：', len(set(train[train['label'] == 1]['patient_ID'])))
    # print('val良性有：', len(set(val[val['label'] == 0]['patient_ID'])))
    # print('val恶性有：', len(set(val[val['label'] == 1]['patient_ID'])))
    # print('test良性有：', len(set(test[test['label'] == 0]['patient_ID'])))
    # print('test恶性有：', len(set(test[test['label'] == 1]['patient_ID'])))
    # ---------------------------------------------------------------------------

    if not os.path.exists(Fconfig.result_path):
        os.makedirs(Fconfig.result_path)
    if not os.path.exists("K fold excel"):
        os.makedirs("K fold excel")
    sys.stdout = Logger(Fconfig.result_path + "/" + Fconfig.save_model_name + ".txt")

    startTime_1 = time.strftime("%Y-%m-%d %H:%M:%S")
    print("开始训练时间:" + time.strftime("%Y-%m-%d %H:%M:%S"))

    dic = Fconfig.__dict__
    list_dic = list(dic.keys())
    print("---------------------------  参数 parameters  ---------------------------")
    for k in list_dic:
        print("*\t" + k + ":\t" + str(dic[k]))

    if Fconfig.cross_validataion:
        cross_validataion()
    else:
        train_and_test()

    endTime_1 = time.strftime("%Y-%m-%d %H:%M:%S")
    print("结束训练时间:" + time.strftime("%Y-%m-%d %H:%M:%S"))
    print("总共用时：" + minNums(startTime_1, endTime_1))