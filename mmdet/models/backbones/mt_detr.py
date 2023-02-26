
from re import X
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from ..builder import BACKBONES

from mmcv.runner import BaseModule

from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .trident_resnet import TridentResNet
from .swin_transformer import SwinTransformer, DoubleSwinTransformer, TripleSwinTransformer
from .cbnet import CBResNet, CBRes2Net, CBSwinTransformer, TwoCBResNet, ThreeCBResNet, CB2ResNet, CBConvNeXt
from .convnext import ConvNeXt
from .fusion_module import SimpleFusionModule, ResidualFusionModule, ConfidenceFusionModule, EnhancementModule





@BACKBONES.register_module()
class CameraOnly(BaseModule):

    def __init__(self,net1,net2,net3,args1,args2,args3,**kwargs):
        super(CameraOnly,self).__init__()
        self.model1=globals()[net1](**args1)
        print("backbones loaded!!!!")


    def _freeze_stages(self):
        if '_freeze_stages' in dir(self.model1):
            self.model1._freeze_stages()

    def init_weights(self):
        self.model1.init_weights()

    def forward(self, x):
        x1=x[:,:3,:,:]
        outs1=self.model1(x1)
        return tuple(outs1)

    def train(self, mode=True):
        self.model1.train(mode)


@BACKBONES.register_module()
class MT_DETR_two(BaseModule):

    def __init__(self,net1,net2,net3,net4,net5,args1,args2,args3,args4,args5,loss_type=0,**kwargs):
        super(MT_DETR_two,self).__init__()

        self.model1=globals()[net1](**args1)
        self.model2=globals()[net2](**args2)
        self.fusion=globals()[net3](**args3)
        self.scatter1=globals()[net4](**args4)
        self.scatter2=globals()[net5](**args5)

        self.type=loss_type

        print("backbones loaded!!!!")

    def init_weights(self):
        self.model1.init_weights()
        self.model2.init_weights()

    def forward(self, x):

        x1=x[:,:3,:,:]
        x2=x[:,3:6,:,:]

        outs_list=[]
        outs1=[]
        outs2=[]
        outs3=[]

        for i in range(4):
            x1 = self.model1.forward_stage(i,x1)
            x2 = self.model2.forward_stage(i,x2)

            # fusion
            x_fusion=self.fusion.forward_stage(i,x1,x2)

            if self.type==1:
                # c+l, c
                if i in self.fusion.out_indices:
                    out1 = self.model1.forward_norm(i,x1)
                    outs1.append(out1)

                if i in self.fusion.out_indices:
                    out2 = self.fusion.forward_norm(i,x_fusion)
                    outs2.append(out2)

            elif self.type==2:
                # c+l, l
                if i in self.fusion.out_indices:
                    out1 = self.model2.forward_norm(i,x2)
                    outs1.append(out1)

                if i in self.fusion.out_indices:
                    out2 = self.fusion.forward_norm(i,x_fusion)
                    outs2.append(out2)

            elif self.type==3:
                # c+l, c, l
                if i in self.fusion.out_indices:
                    out1 = self.model2.forward_norm(i,x2)
                    outs1.append(out1)

                if i in self.fusion.out_indices:
                    out2 = self.model1.forward_norm(i,x1)
                    outs2.append(out2)

                if i in self.fusion.out_indices:
                    out3 = self.fusion.forward_norm(i,x_fusion)
                    outs3.append(out3)

            # scatter
            if i!=3:
                x1 = self.scatter1.forward_stage(i,x1,x_fusion)
                x2 = self.scatter2.forward_stage(i,x2,x_fusion)

        outs_list.append(tuple(outs1))
        outs_list.append(tuple(outs2))
        if outs3 != []:
            outs_list.append(tuple(outs3))
        return tuple(outs_list)


    def train(self, mode=True):
        self.model1.train(mode)
        self.model2.train(mode)
        self.fusion.train(mode)
        self.scatter1.train(mode)
        self.scatter2.train(mode)



@BACKBONES.register_module()
class MT_DETR_three(BaseModule):

    def __init__(self,net1,net2,net3,net4,net5,net6,net7,net8,args1,args2,args3,args4,args5,args6,args7,args8,loss_type,**kwargs):
        super(MT_DETR_three,self).__init__()

        self.loss_type=loss_type

        self.model1=globals()[net1](**args1)
        self.model2=globals()[net2](**args2)
        self.model3=globals()[net3](**args3)
        self.fusion1=globals()[net4](**args4)
        self.fusion2=globals()[net5](**args5)
        self.scatter1=globals()[net6](**args6)
        self.scatter2=globals()[net7](**args7)
        self.scatter3=globals()[net8](**args8)

        print("backbones loaded!!!!")

    def init_weights(self):
        self.model1.init_weights()
        self.model2.init_weights()
        self.model3.init_weights()

    def forward(self, x):
        x1=x[:,:3,:,:]
        x2=x[:,3:6,:,:]
        x3=x[:,6:,:,:]

        outs_list=[]
        outs1=[]
        outs2=[]
        outs3=[]
        for i in range(4):
            x1 = self.model1.forward_stage(i,x1)
            x2 = self.model2.forward_stage(i,x2)
            x3 = self.model3.forward_stage(i,x3)

            # fusion
            x_fusion1=self.fusion1.forward_stage(i,x2,x3)
            x_fusion2=self.fusion2.forward_stage(i,x1,x_fusion1)

            if self.loss_type==0:
                if i in self.fusion2.out_indices:
                    out1 = self.fusion1.forward_norm(i,x_fusion1)
                    outs1.append(out1)

                    out2 = self.fusion2.forward_norm(i,x_fusion2)
                    outs2.append(out2)

            elif self.loss_type==1:
                if i in self.fusion2.out_indices:
                    out1 = self.model1.forward_norm(i,x1)
                    outs1.append(out1)

                    out2 = self.fusion2.forward_norm(i,x_fusion2)
                    outs2.append(out2)

            elif self.loss_type==2:
                if i in self.fusion2.out_indices:
                    out1 = self.fusion1.forward_norm(i,x_fusion1)
                    outs1.append(out1)

                    out2 = self.model1.forward_norm(i,x1)
                    outs2.append(out2)

                    out3 = self.fusion2.forward_norm(i,x_fusion2)
                    outs3.append(out3)

            # scatter
            if i!=3:
                # use fusion 1,2
                x1 = self.scatter1.forward_stage(i,x1,x_fusion2)
                x2 = self.scatter2.forward_stage(i,x2,x_fusion1)
                x3 = self.scatter3.forward_stage(i,x3,x_fusion1)

        outs_list.append(tuple(outs1))
        outs_list.append(tuple(outs2))
        if outs3 != []:
            outs_list.append(tuple(outs3))
        return tuple(outs_list)


    def train(self, mode=True):
        self.model1.train(mode)
        self.model2.train(mode)
        self.model3.train(mode)
        self.fusion1.train(mode)
        self.fusion2.train(mode)
        self.scatter1.train(mode)
        self.scatter2.train(mode)
        self.scatter3.train(mode)


@BACKBONES.register_module()
class MT_DETR_four(BaseModule):

    def __init__(self,bb1,bb2,bb3,bb4,fu1,fu2,sca1,sca2,sca3,sca4,args_bb1,args_bb2,args_bb3,args_bb4,args_fu1,args_fu2,args_sca1,args_sca2,args_sca3,args_sca4,loss_type,num_layer=4,scatter=True,**kwargs):
        super(MT_DETR_four,self).__init__()

        self.loss_type=loss_type
        self.scatter=scatter
        self.num_layer=num_layer

        self.model1=globals()[bb1](**args_bb1)
        self.model2=globals()[bb2](**args_bb2)
        self.model3=globals()[bb3](**args_bb3)
        self.model4=globals()[bb4](**args_bb4)
        self.fusion1=globals()[fu1](**args_fu1)
        self.fusion2=globals()[fu2](**args_fu2)
        self.scatter1=globals()[sca1](**args_sca1)
        self.scatter2=globals()[sca2](**args_sca2)
        self.scatter3=globals()[sca3](**args_sca3)
        self.scatter4=globals()[sca4](**args_sca4)

        print("backbones loaded!!!!")

    def init_weights(self):
        self.model1.init_weights()
        self.model2.init_weights()
        self.model3.init_weights()
        self.model4.init_weights()

    def forward(self, x):
        """Forward function."""
        # print(x.size())
        x1=x[:,:3,:,:]
        x2=x[:,3:6,:,:]
        x3=x[:,6:9,:,:]
        x4=x[:,9:,:,:]

        outs_list=[]
        outs1=[]
        outs2=[]
        outs3=[]
        for i in range(self.num_layer):
            x1 = self.model1.forward_stage(i,x1)
            x2 = self.model2.forward_stage(i,x2)
            x3 = self.model3.forward_stage(i,x3)
            x4 = self.model4.forward_stage(i,x4)

            # fusion
            x_fusion1=self.fusion1.forward_stage(i,x2,x3)
            x_fusion2=self.fusion2.forward_stage(i,x1,x_fusion1,x4)

            if self.loss_type==0:
                if i in self.fusion2.out_indices:
                    out1 = self.fusion1.forward_norm(i,x_fusion1)
                    outs1.append(out1)

                    out2 = self.fusion2.forward_norm(i,x_fusion2)
                    outs2.append(out2)

            elif self.loss_type==1:
                if i in self.fusion2.out_indices:
                    out1 = self.model1.forward_norm(i,x1)
                    outs1.append(out1)

                    out2 = self.fusion2.forward_norm(i,x_fusion2)
                    outs2.append(out2)

            elif self.loss_type==2:
                if i in self.fusion2.out_indices:
                    out1 = self.fusion1.forward_norm(i,x_fusion1)
                    outs1.append(out1)

                    out2 = self.model1.forward_norm(i,x1)
                    outs2.append(out2)

                    out3 = self.fusion2.forward_norm(i,x_fusion2)
                    outs3.append(out3)
            
            elif self.loss_type==3:
                if i in self.fusion2.out_indices:
                    out1 = self.fusion2.forward_norm(i,x_fusion2)
                    outs1.append(out1)

            # scatter
            if self.scatter==True:
                if i!=3:
                    # use fusion 1,2
                    x1 = self.scatter1.forward_stage(i,x1,x_fusion2)
                    x2 = self.scatter2.forward_stage(i,x2,x_fusion1)
                    x3 = self.scatter3.forward_stage(i,x3,x_fusion1)
                    x4 = self.scatter4.forward_stage(i,x4,x_fusion2)

        outs_list.append(tuple(outs1))
        if outs2 != []:
            outs_list.append(tuple(outs2))
        if outs3 != []:
            outs_list.append(tuple(outs3))
        return tuple(outs_list)


    def train(self, mode=True):
        self.model1.train(mode)
        self.model2.train(mode)
        self.model3.train(mode)
        self.model4.train(mode)
        self.fusion1.train(mode)
        self.fusion2.train(mode)
        self.scatter1.train(mode)
        self.scatter2.train(mode)
        self.scatter3.train(mode)
        self.scatter4.train(mode)



@BACKBONES.register_module()
class EarlyFusion(BaseModule):
    # convolution first and add
    def __init__(self,bb_type,in_chans,args):
        super(EarlyFusion,self).__init__()

        self.num_sensor=int(in_chans/3)
        print(f"this early fusion model have {self.num_sensor} sensor")

        self.model1=globals()[bb_type](**args)
        if self.num_sensor >=2:
            self.model2=globals()[bb_type](**args)
        if self.num_sensor >=3:
            self.model3=globals()[bb_type](**args)
        if self.num_sensor >=4:
            self.model4=globals()[bb_type](**args)

        print("backbones loaded!!!!")

    def init_weights(self):
        self.model1.init_weights()
        if self.num_sensor >=2:
            self.model2.init_weights()
        if self.num_sensor >=3:
            self.model3.init_weights()
        if self.num_sensor >=4:
            self.model4.init_weights()

    def forward(self, x):
        x1=x[:,:3,:,:]
        if self.num_sensor >=2:
            x2=x[:,3:6,:,:]
        if self.num_sensor >=3:
            x3=x[:,6:9,:,:]
        if self.num_sensor >=4:
            x4=x[:,9:12,:,:]

        outs=[]
        for i in range(4):
            if i == 0:
                x1 = self.model1.forward_stage(i,x1)
                if self.num_sensor >=2:
                    x2 = self.model2.forward_stage(i,x2)
                if self.num_sensor >=3:
                    x3 = self.model3.forward_stage(i,x3)
                if self.num_sensor >=4:
                    x4 = self.model4.forward_stage(i,x4)

                if self.num_sensor ==1:
                    x1=x1
                elif self.num_sensor ==2:
                    x1=x1+x2
                elif self.num_sensor ==3:
                    x1=x1+x2+x3
                elif self.num_sensor ==4:
                    x1=x1+x2+x3+x4
            else:
                x1 = self.model1.forward_stage(i,x1)


            if i in self.model1.out_indices:
                out = self.model1.forward_norm(i,x1)
                outs.append(out)

        return tuple(outs)


    def train(self, mode=True):
        self.model1.train(mode)
        if self.num_sensor >=2:
            self.model2.train(mode)
        if self.num_sensor >=3:
            self.model3.train(mode)
        if self.num_sensor >=4:
            self.model4.train(mode)



@BACKBONES.register_module()
class MiddleFusion_two(BaseModule):

    def __init__(self,net1,net2,net3,args1,args2,args3,**kwargs):
        super(MiddleFusion_two,self).__init__()

        self.model1=globals()[net1](**args1)
        self.model2=globals()[net2](**args2)
        self.fusion=globals()[net3](**args3)

        print("backbones loaded!!!!")

    def init_weights(self):
        self.model1.init_weights()
        self.model2.init_weights()

    def forward(self, x):
        x1=x[:,:3,:,:]
        x2=x[:,3:6,:,:]

        outs=[]
        for i in range(4):
            x1 = self.model1.forward_stage(i,x1)
            x2 = self.model2.forward_stage(i,x2)

             # ready for fusion
            if i in self.fusion.out_indices:
                out=self.fusion.forward_stage(i,x1,x2)
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        self.model1.train(mode)
        self.model2.train(mode)
        self.fusion.train(mode)


@BACKBONES.register_module()
class MiddleFusion_three(BaseModule):

    def __init__(self,net1,net2,net3,args1,args2,args3,**kwargs):
        super(MiddleFusion_three,self).__init__()

        self.model1=globals()[net1](**args1)
        self.model2=globals()[net2](**args2)
        self.model3=globals()[net3](**args3)

        print("backbones loaded!!!!")


    def init_weights(self):
        self.model1.init_weights()
        self.model2.init_weights()
        self.model3.init_weights()

    def forward(self, x):
        x1=x[:,:3,:,:]
        x2=x[:,3:6,:,:]
        x3=x[:,6:,:,:]
        outs1=self.model1(x1)
        outs2=self.model2(x2)
        outs3=self.model3(x3)

        outs=[]
        for i in range(len(outs1)):
            if outs1[i].size(2)!=outs2[i].size(2) or outs1[i].size(3)!=outs2[i].size(3):
                new_outs1=F.interpolate(outs1[i],size=(outs2[i].size(2),outs2[i].size(3)))
                outs.append(torch.cat((new_outs1,outs2[i],outs3[i]),1))
            else:
                outs.append(torch.cat((outs1[i],outs2[i],outs3[i]),1))
        
        return tuple(outs)

    def train(self, mode=True):
        self.model1.train(mode)
        self.model2.train(mode)
        self.model3.train(mode)

