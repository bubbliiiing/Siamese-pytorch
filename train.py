import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.siamese import Siamese
from utils.dataloader import SiameseDataset, dataset_collate
from utils.dataloader_own_dataset import \
    SiameseDataset as SiameseDataset_own_dataset
from utils.dataloader_own_dataset import dataset_collate


def get_image_num(path, train_own_data):
    num = 0
    if train_own_data:
        train_path = os.path.join(path, 'images_background')
        for character in os.listdir(train_path):
            # 在大众类下遍历小种类。
            character_path = os.path.join(train_path, character)
            num += len(os.listdir(character_path))
    else:
        train_path = os.path.join(path, 'images_background')
        for alphabet in os.listdir(train_path):
            # 然后遍历images_background下的每一个文件夹，代表一个大种类
            alphabet_path = os.path.join(train_path, alphabet)
            for character in os.listdir(alphabet_path):
                # 在大众类下遍历小种类。
                character_path = os.path.join(alphabet_path, character)
                num += len(os.listdir(character_path))
    return num

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(net,loss,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    total_loss = 0
    val_loss = 0
    total_accuracy = 0
    val_total_accuracy = 0

    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = Variable(torch.from_numpy(targets).type(torch.FloatTensor)).cuda()
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = Variable(torch.from_numpy(targets).type(torch.FloatTensor))

            optimizer.zero_grad()
            outputs = nn.Sigmoid()(net(images))
            
            output = loss(outputs, targets)
            output.backward()
            optimizer.step()

            with torch.no_grad():
                equal = torch.eq(torch.round(outputs),targets)
                accuracy = torch.mean(equal.float())

            total_loss += output.item()
            total_accuracy += accuracy.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'acc'       : total_accuracy / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = Variable(torch.from_numpy(targets_val).type(torch.FloatTensor)).cuda()
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = Variable(torch.from_numpy(targets_val).type(torch.FloatTensor))
                optimizer.zero_grad()
                outputs = nn.Sigmoid()(net(images_val))
                output = loss(outputs, targets_val)

                equal = torch.eq(torch.round(outputs),targets_val)
                accuracy = torch.mean(equal.float())

            val_loss += output.item()
            val_total_accuracy += accuracy.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1), 
                                'acc'       : val_total_accuracy / (iteration + 1)})
            pbar.update(1)
            
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))


if __name__ == "__main__":
    dataset_path = "./datasets"
    #----------------------------------------------------#
    #   输入图像的大小，默认为105,105,3
    #----------------------------------------------------#
    input_shape = [105,105,3]
    #----------------------------------------------------#
    #   训练自己的数据的话需要把train_own_data设置成true
    #   训练自己的数据和训练omniglot数据格式不一样
    #----------------------------------------------------#
    train_own_data = False
    #-------------------------------#
    #   用于指定是否使用VGG预训练权重
    #   有两种获取方式
    #   1、利用百度网盘下载后放入
    #      ./model_data/
    #   2、直接运行自动下载
    #-------------------------------#
    pretrained = True

    Cuda = True

    model = Siamese(input_shape, pretrained)

    #------------------------------------------#
    #   注释部分可用于断点续练
    #   将训练好的模型重新载入
    #------------------------------------------#
    # # 加快模型训练的效率
    # model_path = "model_data/Omniglot_vgg.pth"
    # print('Loading weights into state dict...')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load(model_path, map_location=device)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    
    net = model.train()
    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    loss = nn.BCELoss()

    train_ratio = 0.9
    images_num = get_image_num(dataset_path, train_own_data)
    num_train = int(images_num*0.9)
    num_val = int(images_num*0.1)
    
    if True:
        Batch_size = 32
        lr = 1e-3
        Init_Epoch = 0
        Freeze_Epoch = 50
        
        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.92)

        if train_own_data:
            train_dataset = SiameseDataset_own_dataset(input_shape, dataset_path, num_train, num_val, train=True)
            val_dataset = SiameseDataset_own_dataset(input_shape, dataset_path, num_train, num_val, train=False)
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                    drop_last=True, collate_fn=dataset_collate)

        else:
            train_dataset = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=True)
            val_dataset = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=False)
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                    drop_last=True, collate_fn=dataset_collate)

        epoch_size = max(1, num_train//Batch_size)
        epoch_size_val = num_val//Batch_size

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_one_epoch(net,loss,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda)
            lr_scheduler.step()

    
    if True:
        Batch_size = 32
        lr = 1e-4
        Freeze_Epoch = 50
        Unfreeze_Epoch = 100

        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.92)

        if train_own_data:
            train_dataset = SiameseDataset_own_dataset(input_shape, dataset_path, num_train, num_val, train=True)
            val_dataset = SiameseDataset_own_dataset(input_shape, dataset_path, num_train, num_val, train=False)
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                    drop_last=True, collate_fn=dataset_collate)

        else:
            train_dataset = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=True)
            val_dataset = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=False)
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                    drop_last=True, collate_fn=dataset_collate)

        epoch_size = max(1, num_train//Batch_size)
        epoch_size_val = num_val//Batch_size

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            fit_one_epoch(net,loss,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda)
            lr_scheduler.step()
