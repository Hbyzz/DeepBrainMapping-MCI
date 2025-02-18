# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : dfc_nii2np.py
# Time       ：2025/2/16 19:48
# Author     ：zhou zihan
# version    ：python 3.6
# Description：
"""

import nibabel as nib
import numpy as np
import os
import json
import cv2


class DFC2NP:
    def __init__(self, jsonFile, dfcRoot, npyPath, npy2DPath):
        '''
        初始化配置
        :param jsonFile: json文件地址（imgMassage.json）
        :param dfcRoot: 动态功能连接分析结果（DYN_01）
        :param npyPath: 存放npy目录
        '''
        self.json = jsonFile
        with open(self.json, 'r') as j:
            self.jsonMessage = json.load(j)
        self.nameList = []
        self.root = dfcRoot
        self.npyPath = npyPath
        if not os.path.exists(self.npyPath):
            os.makedirs(self.npyPath)
        self.npy2DPath = npy2DPath
        if not os.path.exists(self.npy2DPath):
            os.makedirs(self.npy2DPath)
        self.labelList = {}
        self.jsonDataset = {'train':{}, 'test':{}}

    def getNameList(self):
        for name in self.jsonMessage.keys():
            for time in self.jsonMessage[name].keys():
                self.nameList.append(f'{name}_{time}')
                self.labelList[f'{name}_{time}'] = self.jsonMessage[name][time]["Resting State fMRI"]["researchGroup"]
        self.nameList.sort()

    def nii2np(self, niiFile):
        nii_img = nib.load(niiFile)
        nii_data = nii_img.get_fdata()
        nii_data = np.squeeze(nii_data)
        return nii_data

    def findNii(self):
        niiPathList = []
        for Root, _, filees in os.walk(self.root):
            for file in filees:
                firstName, lastName = os.path.splitext(file)
                if lastName == '.nii':
                    niiPathList.append(os.path.join(Root, file))
        niiPathList.sort()
        return niiPathList

    def makeNpy(self):
        self.getNameList()
        niiPathList = self.findNii()
        if len(self.nameList) != len(niiPathList):
            raise IndexError('json文件中读取的subject数量与dfc结果中检索到的nii文件数量不一致')
        for i in range(len(self.nameList)):
            npyData = self.nii2np(niiPathList[i])
            npyFilePath = str(os.path.join(self.npyPath, self.nameList[i]))
            np.save(npyFilePath, npyData)
            print(f'{npyFilePath} OK!')

    def make2Dnpy(self):
        for npyFile in self.nameList:
            npyFilePath = str(os.path.join(self.npyPath, npyFile))
            if not npyFilePath:
                raise FileExistsError(f'不存在对应的npy文件！{npyFilePath}')
            npyData = np.load(f'{npyFilePath}.npy', 'r')
            for i in range(npyData.shape[-1]-1):
                npyData2D = (np.triu(self.revalueNpy(npyData[:, :, i+1] - npyData[:, :, i], 0, 1))
                             +np.tril(self.revalueNpy(npyData[:, :, i+1], 0, 1)))
                np.fill_diagonal(npyData2D, 1)
                dataType = str(os.path.join(self.npy2DPath, self.labelList[f'{npyFile}']))
                if not os.path.exists(dataType):
                    os.makedirs(dataType)
                npyData2DPath = os.path.join(dataType, f'{npyFile}_{i:0{3}d}.npy')
                np.save(npyData2DPath, npyData2D)
                print(f'{npyData2DPath} OK!')

    def revalueNpy(self, npyFile, arrmin, arrmax):
        old_min = np.min(npyFile)
        old_max = np.max(npyFile)
        # 处理原区间最大值和最小值相等的情况，避免除零错误
        if old_max - old_min == 0:
            return np.full_like(npyFile, arrmin)
        scaled = arrmin + (npyFile - old_min) * (arrmax - arrmin) / (old_max - old_min)
        return scaled

    def showNpy(self, npy2Dpath):
        arrData = np.load(npy2Dpath, 'r')
        # self.revalueNpy(arrData, arrmin=0, arrmax=255)
        cv2.imwrite('img1.png', self.revalueNpy(arrData, arrmin=0, arrmax=255), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        bigimg = cv2.resize(arrData, (720, 720),
                            interpolation=cv2.INTER_NEAREST)
        cv2.imshow('DFC', bigimg)
        print(arrData)
        cv2.waitKey(0)

    def getJson(self):
        for root, _, files in os.walk(self.npy2DPath):
            if files:
                label = os.path.basename(root)
                totallen = len(files)
                print(label)
            else:
                continue
            for i, fileName in enumerate(files):
                if i<=np.floor(totallen*0.8):
                    mode = 'train'
                else:
                    mode = 'test'
                    i = int(i - np.floor(totallen*0.8))

                self.jsonDataset[mode][i] = {'path':os.path.join(root, fileName),
                                                    'label':label}
        with open('dataset.json', 'w') as j:
            json.dump(self.jsonDataset, j, ensure_ascii=False, indent=4)



if __name__=="__main__":
    jsonFile = r'Dataset_selected/run0/imgMassage.json'
    root = r'F:\MATLAB\project\MCI_fMRI\conn_mat\conn_project03\results\firstlevel\DYN_01'
    npDir = './dfcNumpy'
    npy2DPath = './Dataset'
    dfc2npy = DFC2NP(jsonFile=jsonFile,
                     dfcRoot=root,
                     npyPath=npDir,
                     npy2DPath=npy2DPath)
    # dfc2npy.makeNpy()
    # dfc2npy.make2Dnpy()
    dfc2npy.getJson()

