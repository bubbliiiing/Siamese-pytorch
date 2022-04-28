## Siamese：孪生神经网络在Pytorch当中的实现
---

## 目录
1. [仓库更新 Top News](#仓库更新)
2. [注意事项 Attention](#注意事项)
3. [所需环境 Environment](#所需环境)
4. [文件下载 Download](#文件下载)
5. [预测步骤 How2predict](#预测步骤)
6. [训练步骤 How2train](#训练步骤)
7. [参考资料 Reference](#Reference)

## Top News
**`2022-04`**:**进行了大幅度的更新，支持step、cos学习率下降法、支持adam、sgd优化器选择、支持学习率根据batch_size自适应调整。**  
BiliBili视频中的原仓库地址为：https://github.com/bubbliiiing/siamese-pytorch/tree/bilibili

## 注意事项
**训练Omniglot数据集和训练自己的数据集可以采用两种不同的格式**。需要注意格式的摆放噢！  

该仓库实现了孪生神经网络（Siamese network），该网络常常用于检测输入进来的两张图片的相似性。该仓库所使用的主干特征提取网络（backbone）为VGG16。  

## 所需环境
torch==1.2.0

## 文件下载
训练所需的vgg16-397923af.pth可在百度网盘中下载。    
链接: https://pan.baidu.com/s/14SFoKX6xTDPx2XG9rcUTDQ 提取码: 44en    

Omniglot数据集下载地址为：    
链接: https://pan.baidu.com/s/1pYp6vqiLLRFLn1tVeRk8ZQ 提取码: 5sa7      

人脸数据集下载地址为（格式还需要简单修改一下才可以使用，请参考下方“训练自己相似性比较的模型”的格式进行修改）：   
链接: https://pan.baidu.com/s/1OvEFXTUZrvu4T5qSPkHOJw 提取码: aqhg     

我一共会提供两个权重，分别是vgg16-397923af.pth和Omniglot_vgg.pth。   
其中:  
Omniglot_vgg.pth是Omniglot训练好的权重，可直接使用进行下面的预测步骤。  
vgg16-397923af.pth是vgg的权重，可以用于训练其它的数据集。   

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，在百度网盘下载Omniglot_vgg.pth，放入model_data，运行predict.py，依次输入    
```python
img/Angelic_01.png
```
```python
img/Angelic_02.png
```  
### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在siamese.py文件里面，在如下部分修改model_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件**。    
```python
_defaults = {
    "model_path": 'model_data/Omniglot_vgg.pth',
    "input_shape" : (105, 105, 3),
}
```
3. 运行predict.py，输入   
```python
img/Angelic_01.png
```
```python
img/Angelic_02.png
```

## 训练步骤
可参考我的CSDN博客https://blog.csdn.net/weixin_44791964/article/details/107343394
### a、训练Omniglot例子  
Omniglot数据集中数据存放格式有三级：
```python
- image_background
	- Alphabet_of_the_Magi
		- character01
			- 0709_01.png
			- 0709_02.png
			- ……
		- character02
		- character03
		- ……
	- Anglo-Saxon_Futhorc
	- ……
```
训练步骤为：  
1. 下载数据集，放在根目录下的dataset文件夹下。     
2. 运行train.py开始训练。   
### b、训练自己相似性比较的模型
如果大家想要训练自己的数据集，可以将数据集按照如下格式进行摆放。    
```python
- image_background
	- character01
		- 0709_01.png
		- 0709_02.png
		- ……
	- character02
	- character03
	- ……
```
相比Omniglot少了一级。每一个chapter里面放同类型的图片。    
训练步骤为：  
1. 按上述格式放置数据集，放在根目录下的dataset文件夹下。     
2. 之后将train.py当中的train_own_data设置成True。  
3. 运行train.py开始训练。 

### Reference
https://github.com/tensorfreitas/Siamese-Networks-for-One-Shot-Learning
