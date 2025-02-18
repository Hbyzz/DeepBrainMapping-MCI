# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : dcm2nii.py
# Time       ：2025/1/13 13:37
# Author     ：zhou zihan
# version    ：python 3.6
# Description：
"""

import dicom2nifti
import os
import gzip
import nibabel as nib
import xml.etree.ElementTree as ET
import json


def print_folder_tree(path, parent_is_last=1, depth_limit=-1, tab_width=1, max_files=3):
    """
    以树状打印输出文件夹下的文件, 并返回文件夹内的所有文件
    :param path: 文件夹路径
    :param parent_is_last: 递归调用上级文件夹是否是最后一个文件(夹), 控制输出 │ 树干
    :param depth_limit: 要输出文件夹的层数, -1为输出全部文件及文件夹
    :param tab_width: 空格宽度
    :param max_files：文件夹下打印文件的最大数量
    """
    # 打印根目录名称
    if len(str(parent_is_last)) == 1:
        print(os.path.basename(path))

    items = [item for item in os.listdir(path) if not item.startswith('.')]  # 过滤掉隐藏文件和文件夹
    for index, i in enumerate(items):
        if index >= max_files and index < len(items) - 1:
            # 打印省略号并跳过剩余项
            for k in str(parent_is_last)[1:]:
                if k == "0":
                    print("│" + "\t" * tab_width, end="")
                if k == "1":
                    print("\t" * tab_width, end="")
            print("├── ...")
            break

        is_last = index == len(items) - 1
        i_path = os.path.join(path, i)

        for k in str(parent_is_last)[1:]:
            if k == "0":
                print("│" + "\t" * tab_width, end="")
            if k == "1":
                print("\t" * tab_width, end="")

        if is_last:
            print("└── ", end="")
        else:
            print("├── ", end="")

        if os.path.isdir(i_path):
            print(i)
            if depth_limit != 0:  # 添加深度限制检查
                print_folder_tree(
                    path=i_path,
                    depth_limit=depth_limit - 1 if depth_limit > 0 else -1,
                    parent_is_last=(parent_is_last * 10 + 1) if is_last else (parent_is_last * 10),
                    max_files=max_files,
                    tab_width=tab_width
                )
        else:
            print(i)


class DCM2NII:
    """
    数据转换类
    self.root：根目录
    self.outDataset：输出路径
    self.imgDict：存储图像基本信息
    self.peopleList：存储病人编号（索引_时间）
    self.translation_dict：翻译

    self.dcm2nii:具体的转换过程
    self.print_all_elements：打印xml文件元素
    self.getJson：生成json文件
    self.getMatlab：生成json文件（matlab）
    """
    def __init__(self, root=r"F:\MATLAB\project\MCI_fMRI\MCI_AD_Dataset\MRI\ADNI",
                 metadataRoot=r'F:\MATLAB\project\MCI_fMRI\MCI_AD_Dataset\_fMRI_metadata\ADNI',
                 outDataset='./outDataset'):
        """
        初始化
        :param root: 根目录
        :param metadataRoot: 元数据目录
        :param outDataset: 输出目录
        """
        self.root = root
        if not os.path.exists(outDataset):
            os.makedirs(outDataset)
        for i in range(999):
            output = os.path.join(outDataset, 'run' + str(i))
            if not os.path.exists(output):
                os.makedirs(output)
                self.outDataset = output
                break
        if not os.path.exists(self.outDataset):
            os.makedirs(self.outDataset)
        self.imgDict = {}
        self.peopleList = []
        self.translation_dict = {
            "projectIdentifier": "项目标识符",
            "projectDescription": "项目描述",
            "siteKey": "站点密钥",
            "subjectIdentifier": "受试者标识符",
            "researchGroup": "研究组",
            "subjectSex": "受试者性别",
            "subjectInfo": "受试者信息",
            "visitIdentifier": "访视标识符",
            "MMSCORE": "MMSE(简易精神状态检查表) 得分",
            "GDTOTAL": "GDS(老年抑郁量表) 总分值",
            "CDGLOBAL": "CDR(临床痴呆评定量表) 总体评分",
            "FAQTOTAL": "功能评估问卷总分值",
            "studyIdentifier": "研究标识符",
            "subjectAge": "受试者年龄",
            "ageQualifier": "年龄限定符",
            "weightKg": "体重（千克）",
            "postMortem": "是否尸检",
            "seriesIdentifier": "序列标识符",
            "modality": "成像模态",
            "dateAcquired": "采集日期",
            "imageUID": "图像唯一标识符",
            "description": "描述",
            "Acquisition Type": "采集类型",
            "Weighting": "加权",
            "Coil": "线圈",
            "Acquisition Plane": "采集平面",
            "Manufacturer": "制造商",
            "Mfg Model": "制造型号",
            "Field Strength": "磁场强度",
            "Pulse Sequence": "脉冲序列",
            "TE": "回波时间（TE）",
            "TR": "重复时间（TR）",
            "TI": "反转时间",
            "Flip Angle": "翻转角",
            "Matrix X": "矩阵X",
            "Matrix Y": "矩阵Y",
            "Matrix Z": "矩阵Z",
            "Slices": "切片数量",
            "Pixel Spacing X": "像素间距X",
            "Pixel Spacing Y": "像素间距Y",
            "Slice Thickness": "层厚"
        }
        for Root, dirs, files in os.walk(self.root):
            if not dirs:
                self.path = Root
                self.fmriNum = os.path.basename(self.path)   # 获取编号
                self.path = os.path.dirname(self.path)
                self.fmriTime = os.path.basename(self.path)   # 获取时间
                self.path = os.path.dirname(self.path)
                self.fmriType = os.path.basename(self.path)   # 获取扫描类型
                self.path = os.path.dirname(self.path)
                self.fmriPeople = os.path.basename(self.path)   # 获取患者编号
                if self.fmriPeople not in self.peopleList:
                    self.peopleList.append(self.fmriPeople)
                self.imgDict[self.fmriNum] = {'path': Root,
                                              'num': self.fmriNum,
                                              'time': self.fmriTime,
                                              'type': self.fmriType,
                                              'people': self.fmriPeople}
        self.xmlPathList = []
        for Root, dirs, files in os.walk(metadataRoot):
            if Root == metadataRoot:
                for file in files:
                    filePath = os.path.join(Root, file)
                    self.xmlPathList.append(filePath)

        self.imgMassageJson = {} # 索引：仿照ADNI：病人编号-时间-图像类型
        self.imgMassage = {}
        self.outputDirsList = []
        self.imgMatlabJson = {}
        self.imgMatlab = {}

    def dcm2nii(self):
        """
        dcm转nii过程
        :return:
        """
        '''
        for i in range(999):
            output = os.path.join(outDataset, 'run'+str(i))
            if not os.path.exists(output):
                os.makedirs(output)
                self.outDataset = output
                break
        '''
        numList = self.imgDict.keys()   # 索引列表（图像编号）
        for i in numList:
            # 分别转换每个图像
            dcmType = self.imgDict[i]['type']   # 类型
            dcmPeople = self.imgDict[i]['people']   # 病人编号
            dcmTime = self.imgDict[i]['time']   # 时间
            output = os.path.join(self.outDataset, dcmType)  # 输出路径
            # self.outputDirsList.append(os.path.join(self.outDataset, dcmPeople+'_'+dcmTime[0:10]))
            # 没有此路径？创建
            if not os.path.exists(output):
                os.makedirs(output)
            dicom2nifti.convert_directory(self.imgDict[i]['path'], output)
            print(f'{i} is OK!')
            # 找到生成的.nii.gz 文件, 然后解压
            for root, _, files in os.walk(output):
                for file in files:
                    if file.endswith('.nii.gz'):
                        nii_gz_file = os.path.join(root, file)
                        # 打开.nii.gz 文件
                        with gzip.open(nii_gz_file, 'rb') as f_in:
                            # 生成.nii 文件的路径
                            nii_file = nii_gz_file[:-3]
                            # 解压.nii.gz 文件
                            with open(nii_file, 'wb') as f_out:
                                f_out.write(f_in.read())
                        # 删除原始的.nii.gz 文件
                        os.remove(nii_gz_file)

            # 接下来对nii文件裁剪，丢弃前十时间点
            if dcmType == 'Resting_State_fMRI':
                # 读取原始4D NIfTI文件
                original_img = nib.load(nii_file)
                original_data = original_img.get_fdata()
                # 抛弃前十时间点
                selectedData = original_data[..., 10:]
                # 创建新的NIfTI图像对象
                new_img = nib.Nifti1Image(selectedData, original_img.affine, original_img.header)
                # 保存新的4D NIfTI文件
                nib.save(new_img, os.path.join(self.outDataset, dcmType, dcmPeople+'_'+dcmTime[0:10]+'.nii'))
                # 删除原始的.nii 文件
                os.remove(nii_file)
            elif dcmType == 'MPRAGE':
                os.rename(nii_file, os.path.join(self.outDataset, dcmType, dcmPeople+'_'+dcmTime[0:10]+'.nii'))

    def initImgMassage(self):
        self.imgMassage = {}

    def getOutputDirs(self, dcm2niiResultPath=None):
        if not dcm2niiResultPath:
            dcm2niiResultPath = self.outDataset
        for Root, dirs, files in os.walk(dcm2niiResultPath):
            if not dirs:
                if not files:
                    raise ValueError("读不到任何nii图像，检查参数dcm2niiResultPath是否正确设置（指向run系列目录）或目录内有空目录")
                for niiFiles in files:
                    niiFiles = niiFiles[-25:]
                    if niiFiles not in self.outputDirsList and niiFiles[-4:] == '.nii':
                        self.outputDirsList.append(niiFiles)

    def print_all_elements(self, element, level=0, mode=0, txt=''):
        """
        打印xml文件所有元素（mode==0）；精简打印（mode==1）
        :param element: 需要打印的元素（已读取好的xml结构）
        :param level: 缩进
        :param mode: 模式
        :return:
        """
        indent = "  " * level
        pas = 0 if element.text.strip() else 1
        try:
            label = element.tag if not element.attrib else element.attrib['term']
        except KeyError:
            label = element.tag if not element.attrib else element.attrib.get('attribute', element.tag)

        if mode==0:
            print(f"{indent}{self.translation_dict[label]}: {element.text.strip() if element.text else None}")
            txt = txt+f"{indent}{self.translation_dict[label]}: {element.text.strip() if element.text else None}"+'\n'
        elif mode==1 and pas != 1:
            self.imgMassage[label] = element.text.strip()
            print(f"{self.translation_dict[label]}: {element.text.strip() if element.text else None}")
            txt = txt + f"{self.translation_dict[label]}: {element.text.strip() if element.text else None}" + '\n'

        for child in element:
            txt = self.print_all_elements(child, level + 1, mode, txt)

        return txt

    def getJson(self, xmlPath=None, isCopy=True, sampleMode=1, isSavePast=True, lastJsonPath=None):
        """
        生成json文件，包含图像的全部信息；后续会用...可能...
        :param xmlPath: 单个xml文件，为None时处理元数据文件夹下第一层的全部xml文件；为单个xml文件地址时，打印单个文件的信息，且不生成json
        :param isCopy: 是否打印信息
        :param sampleMode: 打印信息详细（0）or简略（1）
        :return:
        """
        if xmlPath:
            if xmlPath[-3:] != 'xml':
                print("这不是xml文件")
                return
            # 单个文件不生成任何文件
            # 解析XML文件
            tree = ET.parse(xmlPath)
            # 获取根元素
            root = tree.getroot()
            if isCopy:  # 是否打印全文内容
                self.print_all_elements(root, mode=sampleMode)
                txt = self.print_all_elements(root, mode=sampleMode)
                if not os.path.exists(os.path.join(self.outDataset, 'massageCN')):
                    os.makedirs(os.path.join(self.outDataset, 'massageCN'))
                with open(os.path.join(self.outDataset, 'massageCN', f'{os.path.basename(xmlPath)[:-4]}.txt'),
                          'w') as f:
                    f.write(txt)
            return

        if isSavePast:
            if lastJsonPath is None:
                raise ValueError("要读取的json文件路径不能为None")
            with open(lastJsonPath, 'r')as j:
                self.imgMassageJson = json.load(j)

        for filePath in self.xmlPathList:
            if filePath[-3:] != 'xml':
                raise ValueError("这不是xml文件, 或者存放xml文件的目录内存在非xml文件")
            print('\n\t', '正在处理：', os.path.basename(filePath))
            self.initImgMassage()   # 清空上一个图像信息，以填入新的
            # 解析XML文件
            tree = ET.parse(filePath)
            # 获取根元素
            root = tree.getroot()
            if isCopy:  # 是否打印全文内容
                txt = self.print_all_elements(root, mode=sampleMode)
                if not os.path.exists(os.path.join(self.outDataset, 'massageCN')):
                    os.makedirs(os.path.join(self.outDataset, 'massageCN'))
                with open(os.path.join(self.outDataset, 'massageCN', f'{os.path.basename(filePath)[:-4]}.txt'), 'w') as f:
                    f.write(txt)
            people = self.imgMassage['subjectIdentifier']
            dateAcquired = self.imgMassage['dateAcquired']
            imgType = self.imgMassage['description']
            try:
                self.imgMassageJson[people][dateAcquired][imgType] = self.imgMassage
            except KeyError:
                self.imgMassageJson[people] = {} if people not in self.imgMassageJson.keys() else self.imgMassageJson[people]
                self.imgMassageJson[people][dateAcquired] = {} if people not in self.imgMassageJson[people].keys() else self.imgMassageJson[people][dateAcquired]
                self.imgMassageJson[people][dateAcquired][imgType] = self.imgMassage

        with open(os.path.join(self.outDataset, 'imgMassage.json'), 'w') as imgjson:
            json.dump(self.imgMassageJson, imgjson, ensure_ascii=False, indent=4)

    def getMatlab(self, dcm2niiResultPath=None):
        """
        生成matlab处理时可用的json信息包括病人编号，日期、mprage和静息态MRI的绝对地址（所以移动输出文件夹matlab处理时会报错）
        :param dcm2niiResultPath: dcm文件转换nii文件的存放目录
        :return:
        """
        self.getOutputDirs(dcm2niiResultPath)
        if not self.outputDirsList:
            raise KeyError(
                "读取MPRAGE或静息态fmri内文件列表时出错，请确保dcm2nii转换结果路径正确，或其目录内没有其他无关文件")
        for i, nii in enumerate(self.outputDirsList):
            mprage = os.path.join(self.outDataset, 'MPRAGE')
            rsmri = os.path.join(self.outDataset, 'Resting_State_fMRI')
            people = nii[0:10]
            imgDate = nii[11:-4]
            try:
                tr = self.imgMassageJson[people][imgDate]['Resting State fMRI']['TR']
            except KeyError:
                raise KeyError("读取MPRAGE或静息态fmri内文件列表时出错，请确保dcm2nii转换结果路径正确，或其目录内没有其他无关文件")
            self.imgMatlab = {'people': people,
                              'time': imgDate,
                              'mprage': os.path.abspath(mprage),
                              'rsmri': os.path.abspath(rsmri),
                              'niiName': '^'+nii+'$',
                              'tr': tr}
            self.imgMatlabJson[i] = self.imgMatlab
        with open(os.path.join(self.outDataset, 'imgMatlab.json'), 'w') as f:
            json.dump(self.imgMatlabJson, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    root = r"F:\MATLAB\project\MCI_fMRI\MCI_AD_Dataset\_fMRI\ADNI"  # 数据集的根目录
    metadataRoot = r'F:\MATLAB\project\MCI_fMRI\MCI_AD_Dataset\_fMRI_metadata\ADNI' # 数据集元数据根目录
    outDataset = r'./Dataset_selected'  # 输出目录
    # 是否有打印目录结构需求
    isprint_folder_tree = False  # 是否打印目录结构
    path = r'Dataset_selected'                # 目录地址
    depth_limit = -1            # 深度限制（-1无限制）
    max_files = 99               # 每个文件夹内最多打印max_files个文件
    # 是否创建转换类
    is_D2N = True              # 是否创建D2N转换类
    # 是否有dic转nii的需求
    is_dcm2nii = False                 # 是否dic转nii
    # 是否生成json文件
    is_json = True              # 是否生成json文件
    isCopy = True               # 是否打印写入的信息（中文）
    samplemode = 1              # 打印xml文件元素模式 详细（0）or 简略（1）
    xmlPath = None              # 是单个xml文件地址，会打印其信息；None会处理输出路径下所有xml文件
    isSavePast = False           # 继续上次保存的json文件，继续保存
    lastJsonPath = None         # 上次保存的json文件目录，如果isSavePast = True，则lastJsonPath不能为None
    dcm2niiResultPath = r'Dataset_selected/run0'    # dcm文件转换nii文件的存放目录， 当is_dcm2nii = False且生成json文件时，应用上次转换输出目录

    if isprint_folder_tree:
        print_folder_tree(path=path, depth_limit=depth_limit, max_files=max_files)
    if is_D2N:
        D2N = DCM2NII(root=root, metadataRoot=metadataRoot, outDataset=outDataset)
        if is_dcm2nii:
            D2N.dcm2nii()
        if is_json:
            D2N.getJson(xmlPath=xmlPath,
                        isCopy=isCopy,
                        sampleMode=samplemode,
                        isSavePast=isSavePast,
                        lastJsonPath=lastJsonPath)
            D2N.getMatlab(dcm2niiResultPath=dcm2niiResultPath)
