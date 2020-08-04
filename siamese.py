import os
import torch
import numpy as np
import colorsys
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from nets.siamese import Siamese as siamese
from PIL import Image,ImageFont, ImageDraw

class Siamese(object):
    _defaults = {
        "model_path"    : 'model_data/Omniglot_vgg.pth',
        "input_shape"   : (105, 105, 3),
        "cuda"          : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化Siamese
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()
        
    def generate(self):
        
        # 载入模型
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = siamese(self.input_shape)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = model.eval()

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()
            
        print('{} model, anchors, and classes loaded.'.format(self.model_path))
    
    def letterbox_image(self, image, size):
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        if self.input_shape[-1]==1:
            new_image = new_image.convert("L")
        return new_image
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image_1, image_2):
        # 图片预处理，归一化
        with torch.no_grad():
            image_1 = self.letterbox_image(image_1,[self.input_shape[1],self.input_shape[0]])
            image_2 = self.letterbox_image(image_2,[self.input_shape[1],self.input_shape[0]])
            
            photo_1 = np.asarray(image_1).astype(np.float64)/255
            photo_2 = np.asarray(image_2).astype(np.float64)/255
            
            if self.input_shape[-1]==1:
                photo_1 = np.expand_dims(photo_1,-1)
                photo_2 = np.expand_dims(photo_2,-1)

            photo_1 = Variable(torch.from_numpy(np.expand_dims(np.transpose(photo_1,(2,0,1)),0)).type(torch.FloatTensor))
            photo_2 = Variable(torch.from_numpy(np.expand_dims(np.transpose(photo_2,(2,0,1)),0)).type(torch.FloatTensor))
            
            if self.cuda:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()
                
            output = self.net([photo_1,photo_2])[0]
            output = torch.nn.Sigmoid()(output)

                
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(image_1))

        plt.subplot(1, 2, 2)
        plt.imshow(np.array(image_2))
        plt.text(-12, -12, 'Similarity:%.3f' % output, ha='center', va= 'bottom',fontsize=11)
        plt.show()
        return output
