# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial

from numpy import pad
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from mmcv_custom import load_checkpoint
from mmdet.utils import get_root_logger
from ..builder import BACKBONES

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

@BACKBONES.register_module()
class FusionModule_1(nn.Module):
    
    def __init__(self, dims=[128, 256, 512, 1024], num_sensor=3, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.fusion_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        for i in range(len(dims)):
            layer = nn.Sequential(
                LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i]*num_sensor, dims[i], kernel_size=1),
            )
            # # increase size
            # layer = nn.Sequential(
            #     LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
            #     nn.Conv2d(dims[i]*num_sensor, dims[i]*num_sensor, kernel_size=3, padding=1),
            #     nn.ReLU(),
            #     nn.Conv2d(dims[i]*num_sensor, dims[i]*num_sensor, kernel_size=3, padding=1),
            #     nn.Conv2d(dims[i]*num_sensor, dims[i], kernel_size=1),
            # )

            self.fusion_layers.append(layer)

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(len(dims)):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # print("into _init_weight!!!!")
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_stage(self, i,x1,x2,x3=None,x4=None):
        # concat
        if x4 is not None:
            x=torch.cat((x1,x2,x3,x4),1)
        elif x3 is not None:
            x=torch.cat((x1,x2,x3),1)
        else:
            x=torch.cat((x1,x2),1)
        x = self.fusion_layers[i](x)
        # residual
        x_out=x1+x
        if i in self.out_indices:
            norm_layer = getattr(self, f'norm{i}')
            x_out = norm_layer(x_out)

        return x_out

    # def forward_stage(self, i,x1,x2):
    #     # concat
    #     x=torch.cat((x1,x2),1)
    #     x = self.fusion_layers[i](x)
    #     # residual
    #     x_out=x1+x
    #     if i in self.out_indices:
    #         norm_layer = getattr(self, f'norm{i}')
    #         x_out = norm_layer(x_out)

    #     return x_out

    # def forward(self, x):
    #     x = self.forward_features(x)
    #     return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


@BACKBONES.register_module()
class FusionModule_2(nn.Module):
    
    def __init__(self, dims=[128, 256, 512, 1024], num_sensor=3, out_indices=[0, 1, 2, 3],type=0):
        super().__init__()

        self.fusion_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers

        # different type fusion module
        if type==0:
            for i in range(len(dims)):
                layer = nn.Sequential(
                    LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i]*num_sensor, dims[i], kernel_size=1),
                )
                self.fusion_layers.append(layer)

        elif type==1:
            for i in range(len(dims)):
                layer = nn.Sequential(
                    nn.Conv2d(dims[i]*num_sensor, dims[i], kernel_size=1),
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                )
                self.fusion_layers.append(layer)

        elif type==2:
            for i in range(len(dims)):
                layer = nn.Sequential(
                    LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i]*num_sensor, dims[i]*num_sensor, kernel_size=3,padding=1),
                    nn.GELU(),
                    LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i]*num_sensor, dims[i], kernel_size=1),
                )
                self.fusion_layers.append(layer)

        elif type==3:
            for i in range(len(dims)):
                layer = nn.Sequential(
                    LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i]*num_sensor, dims[i]*num_sensor, kernel_size=3,padding=1),
                    nn.GELU(),
                    LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i]*num_sensor, dims[i]*num_sensor, kernel_size=3,padding=1),
                    nn.GELU(),
                    LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i]*num_sensor, dims[i], kernel_size=1),
                )
                self.fusion_layers.append(layer)
        
        elif type==4:
            for i in range(len(dims)):
                layer = nn.Sequential(
                    nn.Conv2d(dims[i]*num_sensor, dims[i], kernel_size=3,padding=1),
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.GELU(),
                )
                self.fusion_layers.append(layer)
        
        elif type==5:
            for i in range(len(dims)):
                layer = nn.Sequential(
                    nn.Conv2d(dims[i]*num_sensor, dims[i]*num_sensor, kernel_size=3,padding=1),
                    LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                    nn.GELU(),
                    nn.Conv2d(dims[i]*num_sensor, dims[i], kernel_size=1),
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                )
                self.fusion_layers.append(layer)

        elif type==6:
            for i in range(len(dims)):
                layer = nn.Sequential(
                    nn.Conv2d(dims[i]*num_sensor, dims[i]*num_sensor, kernel_size=3,padding=1),
                    LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                    nn.GELU(),
                    nn.Conv2d(dims[i]*num_sensor, dims[i]*num_sensor, kernel_size=3,padding=1),
                    LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                    nn.GELU(),
                    nn.Conv2d(dims[i]*num_sensor, dims[i], kernel_size=1),
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                )
                self.fusion_layers.append(layer)

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(len(dims)):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # print("into _init_weight!!!!")
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_norm(self, i,x):
        norm_layer = getattr(self, f'norm{i}')
        x_out = norm_layer(x)

        return x_out


    def forward_stage(self, i,x1,x2,x3=None):
        # concat
        if x3 is not None:
            x=torch.cat((x1,x2,x3),1)
        else:
            x=torch.cat((x1,x2),1)
        x_out = self.fusion_layers[i](x)
        # residual
        x_out=x_out+x1

        return x_out

    # def forward_stage(self, i,x1,x2):
    #     # concat
    #     x=torch.cat((x1,x2),1)
    #     x_out = self.fusion_layers[i](x)
    #     # residual
    #     x_out=x_out+x1

    #     return x_out

@BACKBONES.register_module()
class ScatterModule(nn.Module):
    
    def __init__(self, dims=[128, 256, 512, 1024]):
        super().__init__()

        self.fusion_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        for i in range(len(dims)):
            layer = nn.Sequential(
                LayerNorm(dims[i]*2, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i]*2, dims[i], kernel_size=1),
            )
            # # increase size
            # layer = nn.Sequential(
            #     LayerNorm(dims[i]*2, eps=1e-6, data_format="channels_first"),
            #     nn.Conv2d(dims[i]*2, dims[i]*2, kernel_size=3, padding=1),
            #     nn.ReLU(),
            #     nn.Conv2d(dims[i]*2, dims[i]*2, kernel_size=3, padding=1),
            #     nn.Conv2d(dims[i]*2, dims[i], kernel_size=1),
            # )
            self.fusion_layers.append(layer)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # print("into _init_weight!!!!")
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)


    def forward_stage(self, i,x1,x2):
        # concat
        x=torch.cat((x1,x2),1)
        x_out = self.fusion_layers[i](x)
        # residual
        x_out=x_out+x1

        return x_out

@BACKBONES.register_module()
class FusionModule_3(nn.Module):
    
    def __init__(self, dims=[128, 256, 512, 1024], num_sensor=3, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.fusion_layers = nn.ModuleList()

        ######## model
        # for i in range(4):
        #     layer = nn.Sequential(
        #         LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
        #         nn.Conv2d(dims[i]*num_sensor, dims[i], kernel_size=1),
        #     )
        #     self.fusion_layers.append(layer)

        ########

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # print("into _init_weight!!!!")
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_norm(self, i,x):
        norm_layer = getattr(self, f'norm{i}')
        x_out = norm_layer(x)

        return x_out

    def fusion_1(self,i,x1,x2,x3):
        ## (lidar * radar) * camera

        x_new=torch.mul(x2,x3)
        x_out=torch.mul(x1,x_new)

        return x_out
    
    def fusion_2(self,i,x1,x2,x3):
        ## sigmoid(lidar * sigmoid(radar)) * camera

        x_new=torch.mul(x2,F.sigmoid(x3))
        x_out=torch.mul(x1,F.sigmoid(x_new))

        return x_out

    def fusion_3(self,i,x1,x2,x3):
        ## sigmoid(lidar * sigmoid(radar)) * camera
        if x3 is not None:
            x_new=x2+torch.mul(x2,F.sigmoid(x3))
        else:
            x_new=x2
        x_out=x1+torch.mul(x1,F.sigmoid(x_new))

        return x_out
    
    # def fusion_3(self,i,x1,x2):
    #     ## sigmoid(lidar * sigmoid(radar)) * camera

    #     x_out=x1+torch.mul(x1,F.sigmoid(x2))

    #     return x_out



    def forward_stage(self, i,x1,x2,x3=None):
        # attention by element-wise multiply
        # x_out = self.fusion_1(i,x1,x2,x3)
        # x_out = self.fusion_2(i,x1,x2,x3)
        x_out = self.fusion_3(i,x1,x2,x3)

        return x_out

    # def forward_stage(self, i,x1,x2):
    #     # attention by element-wise multiply
    #     # x_out = self.fusion_1(i,x1,x2,x3)
    #     # x_out = self.fusion_2(i,x1,x2,x3)
    #     x_out = self.fusion_3(i,x1,x2)

    #     return x_out


@BACKBONES.register_module()
class FusionModule_confidence(nn.Module):
    
    def __init__(self, dims=[128, 256, 512, 1024], num_sensor=3, out_indices=[0, 1, 2, 3],type=0, channel_wise=False):
        super().__init__()

        self.fusion_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.channel=channel_wise

        # different type fusion module
        if type==0:
            for i in range(len(dims)):
                layer = nn.Sequential(
                    LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i]*num_sensor, dims[i], kernel_size=1),
                )
                self.fusion_layers.append(layer)

        elif type==1:
            for i in range(len(dims)):
                layer = nn.Sequential(
                    nn.Conv2d(dims[i]*num_sensor, dims[i], kernel_size=1),
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                )
                self.fusion_layers.append(layer)

        elif type==2:
            for i in range(len(dims)):
                layer = nn.Sequential(
                    LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i]*num_sensor, dims[i]*num_sensor, kernel_size=3,padding=1),
                    nn.GELU(),
                    LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i]*num_sensor, dims[i], kernel_size=1),
                )
                self.fusion_layers.append(layer)

        elif type==3:
            for i in range(len(dims)):
                layer = nn.Sequential(
                    LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i]*num_sensor, dims[i]*num_sensor, kernel_size=3,padding=1),
                    nn.GELU(),
                    LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i]*num_sensor, dims[i]*num_sensor, kernel_size=3,padding=1),
                    nn.GELU(),
                    LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i]*num_sensor, dims[i], kernel_size=1),
                )
                self.fusion_layers.append(layer)
        
        elif type==4:
            for i in range(len(dims)):
                layer = nn.Sequential(
                    nn.Conv2d(dims[i]*num_sensor, dims[i], kernel_size=1),
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.GELU(),
                )
                self.fusion_layers.append(layer)
        
        elif type==5:
            for i in range(len(dims)):
                layer = nn.Sequential(
                    nn.Conv2d(dims[i]*num_sensor, dims[i]*num_sensor, kernel_size=3,padding=1),
                    LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                    nn.GELU(),
                    nn.Conv2d(dims[i]*num_sensor, dims[i], kernel_size=1),
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                )
                self.fusion_layers.append(layer)

        elif type==6:
            for i in range(len(dims)):
                layer = nn.Sequential(
                    nn.Conv2d(dims[i]*num_sensor, dims[i]*num_sensor, kernel_size=3,padding=1),
                    LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                    nn.GELU(),
                    nn.Conv2d(dims[i]*num_sensor, dims[i]*num_sensor, kernel_size=3,padding=1),
                    LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                    nn.GELU(),
                    nn.Conv2d(dims[i]*num_sensor, dims[i], kernel_size=1),
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                )
                self.fusion_layers.append(layer)

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(len(dims)):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # print("into _init_weight!!!!")
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_norm(self, i,x):
        norm_layer = getattr(self, f'norm{i}')
        x_out = norm_layer(x)

        return x_out


    def forward_stage(self, i,x1,x2,x3=None):
        # concat
        if x3 == None:
            x=torch.cat((x1,x2),1)
        else:
            x=torch.cat((x1,x2,x3),1)
        confidence = self.fusion_layers[i](x)
        if self.channel==True:
            confidence=F.avg_pool2d(confidence,confidence.size()[2:])
        confidence = F.sigmoid(confidence)
        x_out =  x1 + confidence * x2
        # residual
        # x_out=x_out+x1

        return x_out


@BACKBONES.register_module()
class FusionModule_rgbt(nn.Module):
    def __init__(self, dims=[128, 256, 512, 1024], num_sensor=3, out_indices=[0, 1, 2, 3], init_cam=False, fusebyconv=False, nofusion=False):
        super(FusionModule_rgbt, self).__init__()
        self.out_indices = out_indices
        self.init_cam=init_cam
        self.fusebyconv=fusebyconv
        self.nofusion=nofusion

        c1 = dims[0]
        c2 = dims[1]
        c3 = dims[2]
        c4 = dims[3]

        if self.fusebyconv == False:
            self.block = nn.ModuleList()
            self.block.append(Block_rgbt(c1, first_block=True, init_cam=self.init_cam))
            self.block.append(Block_rgbt(c2))
            self.block.append(Block_rgbt(c3))
            self.block.append(Block_rgbt(c4))

        elif self.nofusion == True:
            self.block = nn.ModuleList()
            self.block.append(Block_rgbt_nofuse(c1, first_block=True, init_cam=self.init_cam))
            self.block.append(Block_rgbt_nofuse(c2))
            self.block.append(Block_rgbt_nofuse(c3))
            self.block.append(Block_rgbt_nofuse(c4))

        else:
            self.block = nn.ModuleList()
            self.block.append(Block_rgbt_2to1(c1, first_block=True, init_cam=self.init_cam))
            self.block.append(Block_rgbt_2to1(c2))
            self.block.append(Block_rgbt_2to1(c3))
            self.block.append(Block_rgbt_2to1(c4))

        self._initialize_weights()

    def forward_stage(self, i,x1,x2,x3,x4=None):

        x1,x2,x3,x4 = self.block[i](x1,x2,x3,x4)

        return x1,x2,x3,x4


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, std=0.01)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Block_rgbt(nn.Module):
    def __init__(self, channels, first_block=False, init_cam=False):
        super(Block_rgbt, self).__init__()
        
        self.first_block = first_block
        self.init_cam=init_cam

        self.m1_msc = MSC(channels)
        self.m2_msc = MSC(channels)
        self.m3_msc = MSC(channels)
        if first_block is False or init_cam is True:
            self.shared_fuse_msc = MSC(channels)
        self.shared_distribute_msc = MSC(channels)

        self.m1_fuse_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.m2_fuse_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.m3_fuse_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)

        self.m1_distribute_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.m2_distribute_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.m3_distribute_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x1, x2, x3, shared):
        if self.first_block:
            if self.init_cam==False:
                shared = torch.zeros(x1.shape).cuda()
            elif self.init_cam==True:
                shared = x1


        new_x1, new_x2, new_x3, new_shared = self.fuse(x1, x2, x3, shared, self.init_cam)
        return new_x1, new_x2, new_x3, new_shared

    def fuse(self, x1, x2, x3, shared, init_cam):

        x1_m = self.m1_msc(x1)
        x2_m = self.m2_msc(x2)
        x3_m = self.m3_msc(x3)

        if self.first_block and init_cam == False:
            shared_m = shared  # zero
        else:
            shared_m = self.shared_fuse_msc(shared)

        x1_s = self.m1_fuse_1x1conv(x1_m - shared_m)
        x1_fuse_gate = torch.sigmoid(x1_s)

        x2_s = self.m2_fuse_1x1conv(x2_m - shared_m)
        x2_fuse_gate = torch.sigmoid(x2_s)

        x3_s = self.m3_fuse_1x1conv(x3_m - shared_m)
        x3_fuse_gate = torch.sigmoid(x3_s)

        new_shared = shared + (x1_m - shared_m) * x1_fuse_gate + (x2_m - shared_m) * x2_fuse_gate + (x3_m - shared_m) * x3_fuse_gate

        # new_shared_m = self.shared_distribute_msc(new_shared)

        # s_x1 = self.m1_distribute_1x1conv(new_shared_m - x1_m)
        # x1_distribute_gate = torch.sigmoid(s_x1)

        # s_x2 = self.m2_distribute_1x1conv(new_shared_m - x2_m)
        # x2_distribute_gate = torch.sigmoid(s_x2)

        # s_x3 = self.m3_distribute_1x1conv(new_shared_m - x3_m)
        # x3_distribute_gate = torch.sigmoid(s_x3)

        # new_x1 = x1 + (new_shared_m - x1_m) * x1_distribute_gate
        # new_x2 = x2 + (new_shared_m - x2_m) * x2_distribute_gate
        # new_x3 = x3 + (new_shared_m - x3_m) * x3_distribute_gate

        return x1, x2, x3, new_shared

        return new_x1, new_x2, new_x3, new_shared

        


class Block_rgbt_2to1(nn.Module):
    def __init__(self, channels, first_block=False, init_cam=False):
        super(Block_rgbt_2to1, self).__init__()
        
        self.first_block = first_block
        self.init_cam=init_cam

        self.m1_msc = MSC(channels)
        self.m2_msc = MSC(channels)
        self.m3_msc = MSC(channels)
        if first_block is False or init_cam is True:
            self.shared_fuse_msc = MSC(channels)
        self.shared_distribute_msc = MSC(channels)

        self.m1_fuse_1x1conv = nn.Conv2d(channels*2, channels, kernel_size=1)
        self.m2_fuse_1x1conv = nn.Conv2d(channels*2, channels, kernel_size=1)
        self.m3_fuse_1x1conv = nn.Conv2d(channels*2, channels, kernel_size=1)

        self.m1_distribute_1x1conv = nn.Conv2d(channels*2, channels, kernel_size=1)
        self.m2_distribute_1x1conv = nn.Conv2d(channels*2, channels, kernel_size=1)
        self.m3_distribute_1x1conv = nn.Conv2d(channels*2, channels, kernel_size=1)

    def forward(self, x1, x2, x3, shared):
        if self.first_block:
            if self.init_cam==False:
                shared = torch.zeros(x1.shape).cuda()
            elif self.init_cam==True:
                shared = x1


        new_x1, new_x2, new_x3, new_shared = self.fuse(x1, x2, x3, shared, self.init_cam)
        return new_x1, new_x2, new_x3, new_shared

    def fuse(self, x1, x2, x3, shared, init_cam):

        x1_m = x1+self.m1_msc(x1)
        x2_m = x2+self.m2_msc(x2)
        x3_m = x3+self.m3_msc(x3)

        if self.first_block and init_cam == False:
            shared_m = shared  # zero
        else:
            shared_m = shared+self.shared_fuse_msc(shared)
        
        x1_s = self.m1_fuse_1x1conv(torch.cat((x1_m,shared_m),1))
        x1_fuse_gate = torch.sigmoid(x1_s)

        x2_s = self.m2_fuse_1x1conv(torch.cat((x2_m,shared_m),1))
        x2_fuse_gate = torch.sigmoid(x2_s)

        x3_s = self.m3_fuse_1x1conv(torch.cat((x3_m,shared_m),1))
        x3_fuse_gate = torch.sigmoid(x3_s)

        new_shared = shared + (x1_m) * x1_fuse_gate + (x2_m) * x2_fuse_gate + (x3_m) * x3_fuse_gate

        new_shared_m = new_shared+self.shared_distribute_msc(new_shared)

        s_x1 = self.m1_distribute_1x1conv(torch.cat((new_shared_m,x1_m),1))
        x1_distribute_gate = torch.sigmoid(s_x1)

        s_x2 = self.m2_distribute_1x1conv(torch.cat((new_shared_m,x2_m),1))
        x2_distribute_gate = torch.sigmoid(s_x2)

        s_x3 = self.m3_distribute_1x1conv(torch.cat((new_shared_m,x3_m),1))
        x3_distribute_gate = torch.sigmoid(s_x3)

        new_x1 = x1 + (new_shared_m) * x1_distribute_gate
        new_x2 = x2 + (new_shared_m) * x2_distribute_gate
        new_x3 = x3 + (new_shared_m) * x3_distribute_gate

        return new_x1, new_x2, new_x3, new_shared


class Block_rgbt_nofuse(nn.Module):
    def __init__(self, channels, first_block=False, init_cam=False):
        super(Block_rgbt_nofuse, self).__init__()
        
        self.first_block = first_block
        self.init_cam=init_cam


        self.m1_msc = MSC(channels)
        self.m2_msc = MSC(channels)
        self.m3_msc = MSC(channels)
        if first_block is False or init_cam is True:
            self.shared_fuse_msc = MSC(channels)
        self.shared_distribute_msc = MSC(channels)

        self.m1_fuse_1x1conv = nn.Conv2d(channels*2, channels, kernel_size=1)
        self.m2_fuse_1x1conv = nn.Conv2d(channels*2, channels, kernel_size=1)
        self.m3_fuse_1x1conv = nn.Conv2d(channels*2, channels, kernel_size=1)

        self.m1_distribute_1x1conv = nn.Conv2d(channels*2, channels, kernel_size=1)
        self.m2_distribute_1x1conv = nn.Conv2d(channels*2, channels, kernel_size=1)
        self.m3_distribute_1x1conv = nn.Conv2d(channels*2, channels, kernel_size=1)

    def forward(self, x1, x2, x3, shared):
        if self.first_block:
            if self.init_cam==False:
                shared = torch.zeros(x1.shape).cuda()
            elif self.init_cam==True:
                shared = x1
                
        new_x1, new_x2, new_x3, new_shared = self.fuse(x1, x2, x3, shared, self.init_cam)
        return new_x1, new_x2, new_x3, new_shared


    def fuse(self, x1, x2, x3, shared, init_cam):

        x1_m = x1+self.m1_msc(x1)
        x2_m = x2+self.m2_msc(x2)
        x3_m = x3+self.m3_msc(x3)

        if self.first_block and init_cam == False:
            shared_m = shared  # zero
        else:
            shared_m = shared+self.shared_fuse_msc(shared)
        
        x1_s = self.m1_fuse_1x1conv(torch.cat((x1_m,shared_m),1))
        x1_fuse_gate = torch.sigmoid(x1_s)

        x2_s = self.m2_fuse_1x1conv(torch.cat((x2_m,shared_m),1))
        x2_fuse_gate = torch.sigmoid(x2_s)

        x3_s = self.m3_fuse_1x1conv(torch.cat((x3_m,shared_m),1))
        x3_fuse_gate = torch.sigmoid(x3_s)

        new_shared = shared + (x1_m) * x1_fuse_gate + (x2_m) * x2_fuse_gate + (x3_m) * x3_fuse_gate

        new_shared_m = new_shared+self.shared_distribute_msc(new_shared)

        s_x1 = self.m1_distribute_1x1conv(torch.cat((new_shared_m,x1_m),1))
        x1_distribute_gate = torch.sigmoid(s_x1)

        s_x2 = self.m2_distribute_1x1conv(torch.cat((new_shared_m,x2_m),1))
        x2_distribute_gate = torch.sigmoid(s_x2)

        s_x3 = self.m3_distribute_1x1conv(torch.cat((new_shared_m,x3_m),1))
        x3_distribute_gate = torch.sigmoid(s_x3)

        new_x1 = x1 + (new_shared_m) * x1_distribute_gate
        new_x2 = x2 + (new_shared_m) * x2_distribute_gate
        new_x3 = x3 + (new_shared_m) * x3_distribute_gate

        # return x1, x2, x3, new_shared
        # return x1, x2, x3, shared
        return new_x1, new_x2, new_x3, new_shared


class MSC(nn.Module):
    def __init__(self, channels):
        super(MSC, self).__init__()
        self.channels = channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv = nn.Sequential(
            nn.Conv2d(3*channels, channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = nn.functional.interpolate(self.pool1(x), x.shape[2:])
        x2 = nn.functional.interpolate(self.pool2(x), x.shape[2:])
        concat = torch.cat([x, x1, x2], 1)
        fusion = self.conv(concat)
        return fusion


@BACKBONES.register_module()
class FusionModule_confidence_four(nn.Module):
    
    def __init__(self, dims=[128, 256, 512, 1024], num_sensor=3, out_indices=[0, 1, 2, 3],type=0, channel_wise=False):
        super().__init__()

        self.fusion_layers1 = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.fusion_layers2 = nn.ModuleList()
        self.channel=channel_wise

        # different type fusion module
        if type==0:
            for i in range(len(dims)):
                layer = nn.Sequential(
                    LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i]*num_sensor, dims[i], kernel_size=1),
                )
                self.fusion_layers1.append(layer)

            for i in range(len(dims)):
                layer = nn.Sequential(
                    LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i]*num_sensor, dims[i], kernel_size=1),
                )
                self.fusion_layers2.append(layer)


        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(len(dims)):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # print("into _init_weight!!!!")
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_norm(self, i,x):
        norm_layer = getattr(self, f'norm{i}')
        x_out = norm_layer(x)

        return x_out


    def forward_stage(self, i,x1,x2,x3,x4):
        # concat
        x=torch.cat((x1,x2,x3,x4),1)
        confidence1 = self.fusion_layers1[i](x)
        confidence2 = self.fusion_layers2[i](x)

        confidence1 = F.sigmoid(confidence1)
        confidence2 = F.sigmoid(confidence2)
        x_out =  x1 + confidence1 * x2 + confidence1 * x3

        return x_out


@BACKBONES.register_module()
class FusionModule_def(nn.Module):
    
    def __init__(self, dims=[128, 256, 512, 1024], num_sensor=3, out_indices=[0, 1, 2, 3],type=0):
        super().__init__()

        self.fusion_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.entropy_layers_1 = nn.ModuleList()
        self.entropy_layers_2 = nn.ModuleList()
        self.entropy_layers_3 = nn.ModuleList()
        self.dims=dims

        # different type fusion module
        for i in range(len(dims)):
            layer = nn.Sequential(
                LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i]*num_sensor, dims[i], kernel_size=1),
            )
            self.fusion_layers.append(layer)

        for i in range(len(dims)):
            layer = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )
            self.entropy_layers_1.append(layer)
        
        for i in range(len(dims)):
            layer = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )
            self.entropy_layers_2.append(layer)
        
        for i in range(len(dims)):
            layer = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )
            self.entropy_layers_3.append(layer)

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(len(dims)):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # print("into _init_weight!!!!")
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_norm(self, i,x):
        norm_layer = getattr(self, f'norm{i}')
        x_out = norm_layer(x)

        return x_out


    def forward_stage(self, i,x1,x2,x3,x1_e,x2_e,x3_e):
        # entropy
        x1_entro=self.entropy_layers_1[i](x1_e)
        x2_entro=self.entropy_layers_2[i](x2_e)
        x3_entro=self.entropy_layers_3[i](x3_e)

        # gating
        x1=x1*x1_entro
        x2=x2*x2_entro
        x3=x3*x3_entro

        # concat entropy map
        x1=x1[:,:self.dims[i]-1,:,:]
        x1=torch.cat((x1,x1_e),1)
        x2=x2[:,:self.dims[i]-1,:,:]
        x2=torch.cat((x2,x2_e),1)
        x3=x3[:,:self.dims[i]-1,:,:]
        x3=torch.cat((x3,x3_e),1)


        # concat
        x=torch.cat((x1,x2,x3),1)
        x_out = self.fusion_layers[i](x)
        # residual
        x_out=x_out+x1

        x_out=x_out[:,:self.dims[i]-1,:,:]
        x_out=torch.cat((x_out,x1_e),1)

        return x_out, x2, x3