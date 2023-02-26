import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..builder import NECKS
from .fpn import FPN
from .. import builder

from .channel_mapper import ChannelMapper

@NECKS.register_module()
class CBFPN(FPN):
    '''
    FPN with weight sharing
    which support mutliple outputs from cbnet
    '''
    def forward(self, inputs):
        # print("start")
        # print(len(inputs))
        # print(len(inputs[0]))
        # print(inputs[0][0].size())

        if not isinstance(inputs[0], (list, tuple)):
            inputs = [inputs]
            
        if self.training:
            outs = []
            for x in inputs:
                out = super().forward(x)
                outs.append(out)

            # print("start cbfpn")
            # print(len(inputs))
            # print(len(inputs[0]))
            # print(inputs[0][0].size())

            return outs
        else:
            out = super().forward(inputs[-1])
            return out

@NECKS.register_module()
class CBChannelMapper(ChannelMapper):
    '''
    ChannelMapper with weight sharing
    which support mutliple outputs from cbnet
    '''
    def forward(self, inputs):
        if not isinstance(inputs[0], (list, tuple)):
            inputs = [inputs]
            
        if self.training:
            outs = []
            for x in inputs:
                out = super().forward(x)
                outs.append(out)

            # print("start cbchannelmapper")
            # print(len(inputs))
            # print(len(inputs[0]))
            # print(inputs[0][0].size())

            return outs
            # to be solved
            # return outs
        else:
            out = super().forward(inputs[-1])
            return out
