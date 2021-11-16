import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.siamese import Siamese
from utils.dataloader import SiameseDataset, dataset_collate
from utils.utils_fit import fit_one_epoch

#----------------------------------------------------#
#   计算图片总数
#----------------------------------------------------#
def get_image_num(path, train_own_data):
    num = 0
    if train_own_data:
        train_path = os.path.join(path, 'images_background')
        for character in os.listdir(train_path):
            #----------------------------------------------------#
            #   在大众类下遍历小种类。
            #----------------------------------------------------#
            character_path = os.path.join(train_path, character)
            num += len(os.listdir(character_path))
    else:
        train_path = os.path.join(path, 'images_background')
        for alphabet in os.listdir(train_path):
            #-------------------------------------------------------------#
            #   然后遍历images_background下的每一个文件夹，代表一个大种类
            #-------------------------------------------------------------#
            alphabet_path = os.path.join(train_path, alphabet)
            for character in os.listdir(alphabet_path):
                #----------------------------------------------------#
                #   在大众类下遍历小种类。
                #----------------------------------------------------#
                character_path = os.path.join(alphabet_path, character)
                num += len(os.listdir(character_path))
    return num

if __name__ == "__main__":
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    Cuda            = True
    #----------------------------------------------------#
    #   数据集存放的路径
    #----------------------------------------------------#
    dataset_path    = "./datasets"
    #----------------------------------------------------#
    #   输入图像的大小，默认为105,105,3
    #----------------------------------------------------#
    input_shape     = [105,105,3]
    #----------------------------------------------------#
    #   当训练Omniglot数据集时设置为False
    #   当训练自己的数据集时设置为True
    #
    #   训练自己的数据和Omniglot数据格式不一样。
    #   详情可看README.md
    #----------------------------------------------------#
    train_own_data  = False
    #-------------------------------#
    #   用于指定是否使用VGG预训练权重
    #   有两种获取方式
    #   1、利用百度网盘下载后放入
    #      ./model_data/
    #   2、直接运行自动下载
    #-------------------------------#
    pretrained      = True
    #----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的，pretrain不影响此处的权值加载。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，pretrain = True，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，pretrain = Fasle，此时从0开始训练。
    #   一般来讲，从0开始训练效果会很差，因为权值太过随机，特征提取效果不明显。
    #
    #   网络一般不从0开始训练，至少会使用主干部分的权值，有些论文提到可以不用预训练，主要原因是他们 数据集较大 且 调参能力优秀。
    #   如果一定要训练网络的主干部分，可以了解imagenet数据集，首先训练分类模型，分类模型的 主干部分 和该模型通用，基于此进行训练。
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = ""

    model = Siamese(input_shape, pretrained)
    if model_path != '':
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    loss = nn.BCELoss()
    #----------------------------------------------------#
    #   训练集和验证集的比例。
    #----------------------------------------------------#
    train_ratio         = 0.9
    images_num          = get_image_num(dataset_path, train_own_data)
    num_train           = int(images_num * train_ratio)
    num_val             = images_num - num_train
    
    #-------------------------------------------------------------#
    #   训练分为两个阶段，两阶段初始的学习率不同，手动调节了学习率
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #-------------------------------------------------------------#
    if True:
        Batch_size      = 32
        Lr              = 1e-4
        Init_epoch      = 0
        Freeze_epoch    = 50

        epoch_step          = num_train // Batch_size
        epoch_step_val      = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
        
        optimizer       = optim.Adam(model_train.parameters(), Lr)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.96)

        train_dataset   = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=True, train_own_data=train_own_data)
        val_dataset     = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=False, train_own_data=train_own_data)
        gen             = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate)
        gen_val         = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True, 
                                drop_last=True, collate_fn=dataset_collate)

        for epoch in range(Init_epoch, Freeze_epoch):
            fit_one_epoch(model_train, model, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Freeze_epoch, Cuda)
            lr_scheduler.step()

    if True:
        Batch_size      = 32
        Lr              = 1e-5
        Freeze_epoch    = 50
        Unfreeze_epoch  = 100

        epoch_step          = num_train // Batch_size
        epoch_step_val      = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        optimizer       = optim.Adam(model_train.parameters(), Lr)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.96)

        train_dataset   = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=True, train_own_data=train_own_data)
        val_dataset     = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=False, train_own_data=train_own_data)
        gen             = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate)
        gen_val         = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True, 
                                drop_last=True, collate_fn=dataset_collate)

        for epoch in range(Freeze_epoch, Unfreeze_epoch):
            fit_one_epoch(model_train, model, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Unfreeze_epoch, Cuda)
            lr_scheduler.step()
