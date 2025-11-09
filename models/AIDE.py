# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import clip
import open_clip
from .srm_filter_kernel import all_normalized_hpf_list
import numpy as np
import sys
import os
# 添加algos目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from algos.cbp_conv import CBPConv

class HPF(nn.Module):
  def __init__(self):
    super(HPF, self).__init__()

    #Load 30 SRM Filters
    all_hpf_list_5x5 = []

    for hpf_item in all_normalized_hpf_list:
      if hpf_item.shape[0] == 3:
        hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

      all_hpf_list_5x5.append(hpf_item)

    hpf_weight = torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5).contiguous()
    hpf_weight = torch.nn.Parameter(hpf_weight.repeat(1, 3, 1, 1), requires_grad=False)
   

    self.hpf = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
    self.hpf.weight = hpf_weight


  def forward(self, input):

    output = self.hpf(input)

    return output



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True, use_cbp=False, cbp_params=None):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(30, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # CBP related attributes
        self.use_cbp = use_cbp
        self.cbp_params = cbp_params or {}
        self.cbp_layers = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        
        # Add CBP layers if enabled
        # 1108
        self._cbp_monitored_parameters_ids = set()

        if self.use_cbp:
            self._add_cbp_layers()
    
    # def _add_cbp_layers(self):
    #     """Add CBP layers to the ResNet architecture, adapted for ResNet50."""
    #     # Get default CBP parameters
    #     replacement_rate = self.cbp_params.get('replacement_rate', 1e-5)
    #     maturity_threshold = self.cbp_params.get('maturity_threshold', 1000)
    #     decay_rate = self.cbp_params.get('decay_rate', 0.99)
    #     util_type = self.cbp_params.get('util_type', 'contribution')
        
    #     # CBP after layer2 - monitor layer2 output and layer3 input
    #     if hasattr(self, 'layer2') and hasattr(self, 'layer3'):
    #         # Get the first block of layer3 to monitor its input
    #         layer3_first_block = self.layer3[0]
    #         in_layer = self.layer2[-1].conv3 if hasattr(self.layer2[-1], 'conv3') else self.layer2[-1].conv2
    #         out_layer = layer3_first_block.conv1
    #         self._register_cbp_module_params(in_layer)
    #         self._register_cbp_module_params(out_layer)
 
    #         self.cbp_layer2 = CBPConv(
    #             #1108 in_layer=self.layer2[-1].conv3 if hasattr(self.layer2[-1], 'conv3') else self.layer2[-1].conv2,
    #             #1108 out_layer=layer3_first_block.conv1,
    #             in_layer=in_layer,
    #             out_layer=out_layer,
    #             replacement_rate=replacement_rate,
    #             maturity_threshold=maturity_threshold,
    #             decay_rate=decay_rate,
    #             util_type=util_type,
    #             # 新增频域敏感度参数
    #             frequency_sensitivity_enabled=self.cbp_params.get('frequency_sensitivity_enabled', True),
    #             lambda_freq=self.cbp_params.get('lambda_freq', 0.1),
    #             frequency_cutoff=self.cbp_params.get('frequency_cutoff', 15),
    #             sensitivity_update_interval=self.cbp_params.get('sensitivity_update_interval', 100),
    #             sensitivity_alpha=self.cbp_params.get('sensitivity_alpha', 0.1)
    #         )
    #         self.cbp_layer2.set_model_info(self, 'layer2')
    #         self.cbp_layers.append(self.cbp_layer2)

    #     # CBP after layer3 - monitor layer3 output and layer4 input
    #     if hasattr(self, 'layer3') and hasattr(self, 'layer4'):
    #         layer4_first_block = self.layer4[0]
    #         #1108
    #         in_layer = self.layer3[-1].conv3 if hasattr(self.layer3[-1], 'conv3') else self.layer3[-1].conv2
    #         out_layer = layer4_first_block.conv1
    #         self._register_cbp_module_params(in_layer)
    #         self._register_cbp_module_params(out_layer)
            
    #         self.cbp_layer3 = CBPConv(
    #             #in_layer=self.layer3[-1].conv3 if hasattr(self.layer3[-1], 'conv3') else self.layer3[-1].conv2,
    #             #out_layer=layer4_first_block.conv1,
    #             in_layer = self.layer3[-1].conv3 if hasattr(self.layer3[-1], 'conv3') else self.layer3[-1].conv2
    #             out_layer = layer4_first_block.conv1
    #             self._register_cbp_module_params(in_layer)
    #             self._register_cbp_module_params(out_layer)
            
    #             replacement_rate=replacement_rate,
    #             maturity_threshold=maturity_threshold,
    #             decay_rate=decay_rate,
    #             util_type=util_type,
    #             # 新增频域敏感度参数
    #             frequency_sensitivity_enabled=self.cbp_params.get('frequency_sensitivity_enabled', True),
    #             lambda_freq=self.cbp_params.get('lambda_freq', 0.1),
    #             frequency_cutoff=self.cbp_params.get('frequency_cutoff', 15),
    #             sensitivity_update_interval=self.cbp_params.get('sensitivity_update_interval', 100),
    #             sensitivity_alpha=self.cbp_params.get('sensitivity_alpha', 0.1)
    #         )
    #         self.cbp_layer3.set_model_info(self, 'layer3')
    #         self.cbp_layers.append(self.cbp_layer3)

    # #1108
    # def _register_cbp_module_params(self, module):
    #     if module is None:
    #         return
    #     for _, param in module.named_parameters(recurse=False):
    #         self._cbp_monitored_parameters_ids.add(id(param))

    # def get_cbp_monitored_param_names(self, prefix=""):
    #     names = []
    #     for name, param in self.named_parameters():
    #         if id(param) in self._cbp_monitored_parameters_ids:
    #             names.append(f"{prefix}{name}" if prefix else name)
    #     return names
    def _add_cbp_layers(self):
        """Add CBP layers to the ResNet architecture, adapted for ResNet50."""
        # Get default CBP parameters
        replacement_rate = self.cbp_params.get('replacement_rate', 1e-5)
        maturity_threshold = self.cbp_params.get('maturity_threshold', 1000)
        decay_rate = self.cbp_params.get('decay_rate', 0.99)
        util_type = self.cbp_params.get('util_type', 'contribution')
        
        # CBP after layer2 - monitor layer2 output and layer3 input
        if hasattr(self, 'layer2') and hasattr(self, 'layer3'):
            # Get the first block of layer3 to monitor its input
            layer3_first_block = self.layer3[0]
            last_block_layer2 = self.layer2[-1]
            if hasattr(last_block_layer2, 'conv3'):
                in_layer = last_block_layer2.conv3
            else:
                in_layer = last_block_layer2.conv2
            out_layer = layer3_first_block.conv1
            self._register_cbp_module_params(in_layer)
            self._register_cbp_module_params(out_layer)
            self.cbp_layer2 = CBPConv(
                in_layer=in_layer,
                out_layer=out_layer,
                replacement_rate=replacement_rate,
                maturity_threshold=maturity_threshold,
                decay_rate=decay_rate,
                util_type=util_type,
                # 新增频域敏感度参数
                frequency_sensitivity_enabled=self.cbp_params.get('frequency_sensitivity_enabled', True),
                lambda_freq=self.cbp_params.get('lambda_freq', 0.1),
                frequency_cutoff=self.cbp_params.get('frequency_cutoff', 15),
                sensitivity_update_interval=self.cbp_params.get('sensitivity_update_interval', 100),
                sensitivity_alpha=self.cbp_params.get('sensitivity_alpha', 0.1)
            )
            self.cbp_layer2.set_model_info(self, 'layer2')
            self.cbp_layers.append(self.cbp_layer2)

        # CBP after layer3 - monitor layer3 output and layer4 input
        if hasattr(self, 'layer3') and hasattr(self, 'layer4'):
            layer4_first_block = self.layer4[0]
            last_block_layer3 = self.layer3[-1]
            if hasattr(last_block_layer3, 'conv3'):
                in_layer = last_block_layer3.conv3
            else:
                in_layer = last_block_layer3.conv2
            out_layer = layer4_first_block.conv1
            self._register_cbp_module_params(in_layer)
            self._register_cbp_module_params(out_layer)
            self.cbp_layer3 = CBPConv(
                in_layer=in_layer,
                out_layer=out_layer,
                replacement_rate=replacement_rate,
                maturity_threshold=maturity_threshold,
                decay_rate=decay_rate,
                util_type=util_type,
                # 新增频域敏感度参数
                frequency_sensitivity_enabled=self.cbp_params.get('frequency_sensitivity_enabled', True),
                lambda_freq=self.cbp_params.get('lambda_freq', 0.1),
                frequency_cutoff=self.cbp_params.get('frequency_cutoff', 15),
                sensitivity_update_interval=self.cbp_params.get('sensitivity_update_interval', 100),
                sensitivity_alpha=self.cbp_params.get('sensitivity_alpha', 0.1)
            )
            self.cbp_layer3.set_model_info(self, 'layer3')
            self.cbp_layers.append(self.cbp_layer3)

    def _register_cbp_module_params(self, module):
        if module is None:
            return
        for _, param in module.named_parameters(recurse=False):
            self._cbp_monitored_parameters_ids.add(id(param))

    def get_cbp_monitored_param_names(self, prefix=""):
        names = []
        for name, param in self.named_parameters():
            if id(param) in self._cbp_monitored_parameters_ids:
                names.append(f"{prefix}{name}" if prefix else name)
        return names


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 存储原始输入供CBP层使用（30通道HPF输出）
        if self.use_cbp:
            self._original_input = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        
        # Pass through layer2 and apply CBP
        x = self.layer2(x)
        if self.use_cbp and hasattr(self, 'cbp_layer2'):
            x = self.cbp_layer2(x)
        
        # Pass through layer3 and apply CBP
        x = self.layer3(x)
        if self.use_cbp and hasattr(self, 'cbp_layer3'):
            x = self.cbp_layer3(x)
        
        # Pass through layer4
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class AIDE_Model(nn.Module):

    def __init__(self, resnet_path, convnext_path, use_cbp=False, cbp_params=None):
        super(AIDE_Model, self).__init__()
        self.hpf = HPF()
        self.model_min = ResNet(Bottleneck, [3, 4, 6, 3], use_cbp=use_cbp, cbp_params=cbp_params)
        self.model_max = ResNet(Bottleneck, [3, 4, 6, 3], use_cbp=use_cbp, cbp_params=cbp_params)
       
        if resnet_path is not None:
            pretrained_dict = torch.load(resnet_path, map_location='cpu',weights_only=False)
        
            model_min_dict = self.model_min.state_dict()
            model_max_dict = self.model_max.state_dict()
    
            for k in pretrained_dict.keys():
                if k in model_min_dict and pretrained_dict[k].size() == model_min_dict[k].size():
                    model_min_dict[k] = pretrained_dict[k]
                    model_max_dict[k] = pretrained_dict[k]
                else:
                    print(f"Skipping layer {k} because of size mismatch")
        
        self.fc = Mlp(2048 + 256 , 1024, 2)

        print("build model with convnext_xxl")
        self.openclip_convnext_xxl, _, _ = open_clip.create_model_and_transforms(
            "convnext_xxlarge", pretrained=convnext_path
        )

        self.openclip_convnext_xxl = self.openclip_convnext_xxl.visual.trunk
        self.openclip_convnext_xxl.head.global_pool = nn.Identity()
        self.openclip_convnext_xxl.head.flatten = nn.Identity()

        self.openclip_convnext_xxl.eval()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.convnext_proj = nn.Sequential(
            nn.Linear(3072, 256),

        )
        for param in self.openclip_convnext_xxl.parameters():
            param.requires_grad = False


#1108
    def get_cbp_monitored_param_names(self):
        monitored = set()
        if hasattr(self.model_min, 'get_cbp_monitored_param_names'):
            monitored.update(self.model_min.get_cbp_monitored_param_names(prefix="model_min."))
        if hasattr(self.model_max, 'get_cbp_monitored_param_names'):
            monitored.update(self.model_max.get_cbp_monitored_param_names(prefix="model_max."))
        return sorted(monitored)


    def forward(self, x):

        b, t, c, h, w = x.shape

        x_minmin = x[:, 0] #[b, c, h, w]
        x_maxmax = x[:, 1]
        x_minmin1 = x[:, 2]
        x_maxmax1 = x[:, 3]
        tokens = x[:, 4]

        x_minmin = self.hpf(x_minmin)
        x_maxmax = self.hpf(x_maxmax)
        x_minmin1 = self.hpf(x_minmin1)
        x_maxmax1 = self.hpf(x_maxmax1)

        with torch.no_grad():
            
            clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
            clip_mean = clip_mean.to(tokens, non_blocking=True).view(3, 1, 1)
            clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
            clip_std = clip_std.to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_mean = torch.Tensor([0.485, 0.456, 0.406]).to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_std = torch.Tensor([0.229, 0.224, 0.225]).to(tokens, non_blocking=True).view(3, 1, 1)

            local_convnext_image_feats = self.openclip_convnext_xxl(
                tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
            ) #[b, 3072, 8, 8]
            assert local_convnext_image_feats.size()[1:] == (3072, 8, 8)
            local_convnext_image_feats = self.avgpool(local_convnext_image_feats).view(tokens.size(0), -1)
            x_0 = self.convnext_proj(local_convnext_image_feats)

        x_min = self.model_min(x_minmin)
        x_max = self.model_max(x_maxmax)
        x_min1 = self.model_min(x_minmin1)
        x_max1 = self.model_max(x_maxmax1)

        x_1 = (x_min + x_max + x_min1 + x_max1) / 4

        x = torch.cat([x_0, x_1], dim=1)

        x = self.fc(x)

        return x

def AIDE(resnet_path, convnext_path, use_cbp=False, cbp_params=None):
    model = AIDE_Model(resnet_path, convnext_path, use_cbp=use_cbp, cbp_params=cbp_params)
    return model

