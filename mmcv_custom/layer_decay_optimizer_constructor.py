# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
from mmcv.runner import get_dist_info

def get_num_layer_layer_wise_multi(var_name, num_max_layer=12):
    
    if var_name in ("backbone.model1.cls_token", "backbone.model1.mask_token", "backbone.model1.pos_embed","backbone.model2.cls_token", "backbone.model2.mask_token", "backbone.model2.pos_embed","backbone.model3.cls_token", "backbone.model3.mask_token", "backbone.model3.pos_embed", "backbone.model4.mask_token", "backbone.model4.pos_embed"):
        return 0
    elif var_name.startswith("backbone.model1.downsample_layers") or var_name.startswith("backbone.model2.downsample_layers") or var_name.startswith("backbone.model3.downsample_layers") or var_name.startswith("backbone.model4.downsample_layers"):
        stage_id = int(var_name.split('.')[3])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3
        elif stage_id == 3:
            layer_id = num_max_layer
        return layer_id
    elif var_name.startswith("backbone.model1.stages") or var_name.startswith("backbone.model2.stages") or var_name.startswith("backbone.model3.stages") or var_name.startswith("backbone.model4.stages"):
        stage_id = int(var_name.split('.')[3])
        block_id = int(var_name.split('.')[4])
        if stage_id == 0:
            layer_id = 1
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3 + block_id // 3
        elif stage_id == 3:
            layer_id = num_max_layer
        return layer_id
    else:
        return num_max_layer + 1

def get_num_layer_layer_wise_multi_fusion(var_name, num_max_layer=12):
    
    if var_name in ("backbone.model1.cls_token", "backbone.model1.mask_token", "backbone.model1.pos_embed","backbone.model2.cls_token", "backbone.model2.mask_token", "backbone.model2.pos_embed","backbone.model3.cls_token", "backbone.model3.mask_token", "backbone.model3.pos_embed"):
        return 0
    elif var_name.startswith("backbone.model1.downsample_layers") or var_name.startswith("backbone.model2.downsample_layers") or var_name.startswith("backbone.model3.downsample_layers"):
        stage_id = int(var_name.split('.')[3])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3
        elif stage_id == 3:
            layer_id = num_max_layer
        return layer_id
    elif var_name.startswith("backbone.model1.stages") or var_name.startswith("backbone.model2.stages") or var_name.startswith("backbone.model3.stages"):
        stage_id = int(var_name.split('.')[3])
        block_id = int(var_name.split('.')[4])
        if stage_id == 0:
            layer_id = 1
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3 + block_id // 3
        elif stage_id == 3:
            layer_id = num_max_layer
        return layer_id
    elif var_name.startswith("backbone.fusion.fusion_layers") or var_name.startswith("backbone.scatter1.fusion_layers") or var_name.startswith("backbone.scatter2.fusion_layers") or var_name.startswith("backbone.scatter3.fusion_layers"):
        # fusion layer
        stage_id = int(var_name.split('.')[3])
        block_id = int(var_name.split('.')[4])
        if stage_id == 0:
            layer_id = 1
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = num_max_layer-1
        elif stage_id == 3:
            layer_id = num_max_layer
        return layer_id
    else:
        return num_max_layer + 1


def get_num_layer_layer_wise_multi_rgbt(var_name, num_max_layer=12):
    
    if var_name in ("backbone.model1.cls_token", "backbone.model1.mask_token", "backbone.model1.pos_embed","backbone.model2.cls_token", "backbone.model2.mask_token", "backbone.model2.pos_embed","backbone.model3.cls_token", "backbone.model3.mask_token", "backbone.model3.pos_embed"):
        return 0
    elif var_name.startswith("backbone.model1.downsample_layers") or var_name.startswith("backbone.model2.downsample_layers") or var_name.startswith("backbone.model3.downsample_layers") or var_name.startswith("backbone.model4.downsample_layers"):
        stage_id = int(var_name.split('.')[3])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3
        elif stage_id == 3:
            layer_id = num_max_layer
        return layer_id
    elif var_name.startswith("backbone.model1.stages") or var_name.startswith("backbone.model2.stages") or var_name.startswith("backbone.model3.stages") or var_name.startswith("backbone.model4.stages"):
        stage_id = int(var_name.split('.')[3])
        block_id = int(var_name.split('.')[4])
        if stage_id == 0:
            layer_id = 1
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3 + block_id // 3
        elif stage_id == 3:
            layer_id = num_max_layer
        return layer_id
    elif var_name.startswith("backbone.fusion.block"):
        # fusion layer
        stage_id = int(var_name.split('.')[3])
        if stage_id == 0:
            layer_id = 1
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3
        elif stage_id == 3:
            layer_id = num_max_layer
        return layer_id
    else:
        return num_max_layer + 1

def get_num_layer_layer_wise_cbnet(var_name, num_max_layer=12):
    
    if var_name in ("backbone.cls_token", "backbone.mask_token", "backbone.pos_embed"):
        return 0
    elif var_name.startswith("backbone.cb_modules.0.downsample_layers") or var_name.startswith("backbone.cb_modules.1.downsample_layers"):
        stage_id = int(var_name.split('.')[4])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3
        elif stage_id == 3:
            layer_id = num_max_layer
        return layer_id
    elif var_name.startswith("backbone.cb_modules.0.stages") or var_name.startswith("backbone.cb_modules.1.stages"):
        stage_id = int(var_name.split('.')[4])
        block_id = int(var_name.split('.')[5])
        if stage_id == 0:
            layer_id = 1
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3 + block_id // 3
        elif stage_id == 3:
            layer_id = num_max_layer
        return layer_id
    else:
        return num_max_layer + 1



def get_num_layer_layer_wise(var_name, num_max_layer=12):
    
    if var_name in ("backbone.cls_token", "backbone.mask_token", "backbone.pos_embed"):
        return 0
    elif var_name.startswith("backbone.downsample_layers"):
        stage_id = int(var_name.split('.')[2])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3
        elif stage_id == 3:
            layer_id = num_max_layer
        return layer_id
    elif var_name.startswith("backbone.stages"):
        stage_id = int(var_name.split('.')[2])
        block_id = int(var_name.split('.')[3])
        if stage_id == 0:
            layer_id = 1
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3 + block_id // 3
        elif stage_id == 3:
            layer_id = num_max_layer
        return layer_id
    else:
        return num_max_layer + 1


def get_num_layer_stage_wise(var_name, num_max_layer):
    if var_name in ("backbone.cls_token", "backbone.mask_token", "backbone.pos_embed"):
        return 0
    elif var_name.startswith("backbone.downsample_layers"):
        return 0
    elif var_name.startswith("backbone.stages"):
        stage_id = int(var_name.split('.')[2])
        return stage_id + 1
    else:
        return num_max_layer - 1
        

@OPTIMIZER_BUILDERS.register_module()
class LearningRateDecayOptimizerConstructor(DefaultOptimizerConstructor):
    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.
        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        print("start construct optimizer!!!!!!!!!!")
        parameter_groups = {}
        print(self.paramwise_cfg)
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        decay_rate = self.paramwise_cfg.get('decay_rate')
        decay_type = self.paramwise_cfg.get('decay_type', "layer_wise")
        print("Build LearningRateDecayOptimizerConstructor %s %f - %d" % (decay_type, decay_rate, num_layers))
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in ('pos_embed', 'cls_token'):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

            if decay_type == "layer_wise":
                layer_id = get_num_layer_layer_wise(name, self.paramwise_cfg.get('num_layers'))
                print(f"set param {name} as id {layer_id}")

            elif decay_type == "layer_wise_multi":
                layer_id = get_num_layer_layer_wise_multi(name, self.paramwise_cfg.get('num_layers'))
                print(f"set param {name} as id {layer_id}")
            
            elif decay_type == "layer_wise_multi_rgbt":
                layer_id = get_num_layer_layer_wise_multi_rgbt(name, self.paramwise_cfg.get('num_layers'))
                print(f"set param {name} as id {layer_id}")
            
            elif decay_type == "layer_wise_cbnet":
                layer_id = get_num_layer_layer_wise_cbnet(name, self.paramwise_cfg.get('num_layers'))
                print(f"set param {name} as id {layer_id}")
            
            ## no help
            elif decay_type == "layer_wise_multi_fusion":
                layer_id = get_num_layer_layer_wise_multi_fusion(name, self.paramwise_cfg.get('num_layers'))
                print(f"set param {name} as id {layer_id}")

            elif decay_type == "stage_wise":
                layer_id = get_num_layer_stage_wise(name, num_layers)
                print(f"set param {name} as id {layer_id}")
                
            group_name = "layer_%d_%s" % (layer_id, group_name)

            if group_name not in parameter_groups:
                scale = decay_rate ** (num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "param_names": [], 
                    "lr_scale": scale, 
                    "group_name": group_name, 
                    "lr": scale * self.base_lr, 
                }

            parameter_groups[group_name]["params"].append(param)
            parameter_groups[group_name]["param_names"].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    "param_names": parameter_groups[key]["param_names"], 
                    "lr_scale": parameter_groups[key]["lr_scale"], 
                    "lr": parameter_groups[key]["lr"], 
                    "weight_decay": parameter_groups[key]["weight_decay"], 
                }
            print("Param groups = %s" % json.dumps(to_display, indent=2))
        
        # state_dict = module.state_dict()
        # for group_name in parameter_groups:
        #     group = parameter_groups[group_name]
        #     for name in group["param_names"]:
        #         group["params"].append(state_dict[name])
        params.extend(parameter_groups.values())