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

@BACKBONES.register_module()
class SimpleFusionModule(nn.Module):
    
    def __init__(self, dims=[128, 256, 512, 1024], num_sensor=3, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.fusion_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        for i in range(len(dims)):
            layer = nn.Sequential(
                LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i]*num_sensor, dims[i], kernel_size=1),
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
class ResidualFusionModule(nn.Module):
    
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

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(len(dims)):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
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



@BACKBONES.register_module()
class ConfidenceFusionModule(nn.Module):
    
    def __init__(self, dims=[128, 256, 512, 1024], num_sensor=3, out_indices=[0, 1, 2, 3],type=0, channel_wise=False):
        super().__init__()

        self.fusion_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.channel=channel_wise

        for i in range(len(dims)):
            layer = nn.Sequential(
                LayerNorm(dims[i]*num_sensor, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i]*num_sensor, dims[i], kernel_size=1),
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

        return x_out

@BACKBONES.register_module()
class EnhancementModule(nn.Module):
    
    def __init__(self, dims=[128, 256, 512, 1024]):
        super().__init__()

        self.fusion_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        for i in range(len(dims)):
            layer = nn.Sequential(
                LayerNorm(dims[i]*2, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i]*2, dims[i], kernel_size=1),
            )
            
            self.fusion_layers.append(layer)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
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


