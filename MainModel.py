import torch
import torch.nn as nn

import datetime
import math
import os
import random
import re

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import cv2

ONNX_EXPORT = False

class YOLOLayer(nn.Module):
    def __init__(self, mask, nc, img_size, yolo_index, layers, stride):
        super(YOLOLayer, self).__init__()
        anchors = [[10,13],[16,30],[33,23],
                [30,61],[62,45],[59,119],
                [116,90],[156,198],[373,326]]  
        self.anchors = torch.Tensor(anchors)
        self.anchors = self.anchors[mask]

        #self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.stride = stride  # layer stride
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(self.anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p, out):
        ASFF = False  # https://arxiv.org/abs/1911.09516
        if ASFF:
            i, n = self.index, self.nl  # index in layers, number of layers
            p = out[self.layers[i]]
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

            # outputs and weights
            # w = F.softmax(p[:, -n:], 1)  # normalized weights
            w = torch.sigmoid(p[:, -n:]) * (2 / n)  # sigmoid weights (faster)
            # w = w / w.sum(1).unsqueeze(1)  # normalize across layer dimension

            # weighted ASFF sum
            p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    p += w[:, j:j + 1] * \
                         F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)

        elif ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny
            ng = 1 / self.ng.repeat((m, 1))
            grid = self.grid.repeat((1, self.na, 1, 1, 1)).view(m, 2)
            anchor_wh = self.anchor_wh.repeat((1, 1, self.nx, self.ny, 1)).view(m, 2) * ng

            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
                torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            return p_cls, xy * ng, wh

        else:  # inference
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


def _make_encoder(backbone, use_pretrained):
    if backbone == "resnext101_wsl":
        pretrained = _make_pretrained_resnext101_wsl(use_pretrained)    
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False
        
    return pretrained
    
def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained
    
def _make_pretrained_resnext101_wsl(use_pretrained):
    resnet = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    return _make_resnet_backbone(resnet)
    
class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.
        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device('cpu'))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)
        
class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False
        )

        return x
       

class MainModel(BaseModel):
    def __init__(self, path=None, non_negative=True):
        print("Loading weights: ", path)

        super(MainModel, self).__init__()

        use_pretrained = False if path is None else True

        self.pretrained = _make_encoder(backbone="resnext101_wsl", use_pretrained=use_pretrained)

        self.yolo_head = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=2, padding=1, bias=False),  #208
            nn.BatchNorm2d(256, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),#104
            nn.BatchNorm2d(256, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),#52
            nn.BatchNorm2d(512, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),#26
            nn.BatchNorm2d(512, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False),#13
            nn.BatchNorm2d(1024, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),#13
            nn.BatchNorm2d(1024, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1, bias=False),#13
            nn.BatchNorm2d(2048, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1, bias=False),#13
            nn.BatchNorm2d(2048, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.yolo4_1 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.yolo4_hack = nn.Sequential(
            nn.Conv2d(256, 27, kernel_size=1, stride=1, padding=0)              #model.module_list[88]       
        )

        self.yolo4_class = YOLOLayer(mask=[6,7,8], nc=4, img_size=(416, 416), yolo_index=0, layers=[], stride = 32)      ##model.module_list[89]
        
        self.yolo3_1 = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear"),                             #model.module_list[92]
            nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0, bias=False),     #model.module_list[91]
            nn.BatchNorm2d(256, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.yolo3_2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),    #model.module_list[90]
            nn.BatchNorm2d(512, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.yolo3_3 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0, bias=False),          #model.module_list[94]
            nn.BatchNorm2d(256, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),         #model.module_list[95]
            nn.BatchNorm2d(512, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),          #model.module_list[96]
            nn.BatchNorm2d(256, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),          #model.module_list[97]
            nn.BatchNorm2d(512, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),          #model.module_list[98]
            nn.BatchNorm2d(256, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),          #model.module_list[99]
            nn.BatchNorm2d(512, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.yolo3_hack = nn.Sequential(
            nn.Conv2d(512, 27, kernel_size=1, stride=1, padding=0)                        #model.module_list[100]  
        )

        self.yolo3_class = YOLOLayer(mask=[3,4,5], nc=4, img_size=(416, 416), yolo_index=1, layers=[], stride = 16)     #model.module_list[101]

        self.yolo2_1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),          #model.module_list[102]
            nn.BatchNorm2d(256, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.yolo2_2 =  nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear"),                               #model.module_list[104]
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),        #model.module_list[103]
            nn.BatchNorm2d(128, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.yolo2_3 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0, bias=False),         #model.module_list[106]
            nn.BatchNorm2d(128, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),         #model.module_list[107]
            nn.BatchNorm2d(256, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),         #model.module_list[108]
            nn.BatchNorm2d(128, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),        #model.module_list[109]
            nn.BatchNorm2d(256, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),        #model.module_list[110]
            nn.BatchNorm2d(128, momentum=0.03, eps=1E-4), 
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),        #model.module_list[111]
            nn.BatchNorm2d(256, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.yolo2_hack = nn.Sequential(
            nn.Conv2d(256, 27, kernel_size=1, stride=1, padding=0)                    #model.module_list[112]
        )

        self.yolo2_class = YOLOLayer(mask=[0,1,2], nc=4, img_size=(416, 416), yolo_index=2, layers=[], stride = 8)   #model.module_list[113]

    
        if path:
            self.load(path)

    def forward(self, x, augment=False, verbose=False):
        if not augment:
            return self.forward_once(x)
        else:  # Augment images (inference and test only) https://github.com/ultralytics/yolov3/issues/931
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0], same_shape=False),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1], same_shape=False),  # scale
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale
            
            y = torch.cat(y, 1)
            return y, None

    def forward_once(self, x, augment=False, verbose=False):
        img_size = x.shape[-2:]  # height, width
        yolo_out, out = [], []
        if verbose:
            print('0', x.shape)
            str = ''

        # Augment images (inference and test only)
        if augment:  # https://github.com/ultralytics/yolov3/issues/931
            nb = x.shape[0]  # batch size
            s = [0.83, 0.67]  # scales
            x = torch.cat((x,
                           torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                           torch_utils.scale_img(x, s[1]),  # scale
                           ), 0)
        yolo_out, out = [], []
        #print(x)
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)
        
        #print(layer_4)
        yolo_layer_head = self.yolo_head(x)                           #13 x 13 x 2048
        yolo_layer4_1 = self.yolo4_1(yolo_layer_head + layer_4)                         # 13 x 13 x 256
        yolo_layer4_2 = self.yolo4_hack(yolo_layer4_1)                # 13 x 13 x 27
        yolo_layer4_out = self.yolo4_class(yolo_layer4_2, out)        # 3 x 13 x 13 x 9

        yolo_layer3_1 = self.yolo3_1(layer_4)                         # 26 x 26 x 256
        yolo_layer3_2 = self.yolo3_2(layer_3)                         # 26 x 26 x 512
        yolo_layer3_3 = torch.cat((yolo_layer3_1,yolo_layer3_2),1)    # 26 x 26 x 768             #model.module_list[93]
        yolo_layer3_4 = self.yolo3_3(yolo_layer3_3)                   # 26 x 26 x 512
        yolo_layer3_5 = self.yolo3_hack(yolo_layer3_4)                # 26 x 26 x 27
        yolo_layer3_out = self.yolo3_class(yolo_layer3_5, out)        # 3 x 26 x 26 x 9

        yolo_layer2_1 = self.yolo2_1(layer_2)                         # 52 x 52 x 256
        yolo_layer2_2 = self.yolo2_2(yolo_layer3_4)                   # 52 x 52 x 128
        yolo_layer2_3 = torch.cat((yolo_layer2_1,yolo_layer2_2),1)    # 52 x 52 x 384
        yolo_layer2_4 = self.yolo2_3(yolo_layer2_3)                   # 52 x 52 x 256
        yolo_layer2_5 = self.yolo2_hack(yolo_layer2_4)                # 52 x 52 x 27
        yolo_layer2_out = self.yolo2_class(yolo_layer2_5, out)        # 3 x 52 x 52 x 9

        #yolo2_out_temp = [torch.unsqueeze(item, dim=0) for item in yolo_layer2_out]
        #yolo2_out_final = torch.stack(yolo2_out_temp, dim=1)
        #yolo2_out_final = torch.squeeze(yolo2_out_final, dim=0)
            
        #yolo3_out_temp = [torch.unsqueeze(item, dim=0) for item in yolo_layer3_out]
        #yolo3_out_final = torch.stack(yolo3_out_temp, dim=1)
        #yolo3_out_final = torch.squeeze(yolo3_out_final, dim=0)
            
        #yolo4_out_temp = [torch.unsqueeze(item, dim=0) for item in yolo_layer4_out]
        #yolo4_out_final = torch.stack(yolo4_out_temp, dim=1)
        #yolo4_out_final = torch.squeeze(yolo4_out_final, dim=0)
        
        yolo_out.append(yolo_layer4_out)
        yolo_out.append(yolo_layer3_out)
        yolo_out.append(yolo_layer2_out)
        
        if self.training:  # train
            return yolo_out
        elif ONNX_EXPORT:  # export
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            if augment:  # de-augment results
                x = torch.split(x, nb, dim=0)
                x[1][..., :4] /= s[0]  # scale
                x[1][..., 0] = img_size[1] - x[1][..., 0]  # flip lr
                x[2][..., :4] /= s[1]  # scale
                x = torch.cat(x, 1)
            return x, p

        
