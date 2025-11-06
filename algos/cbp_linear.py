# -*- coding: utf-8 -*-

import torch
from torch import nn
from math import sqrt
from .frequency_sensitivity import SensitivityAnalyzer


def call_reinit(m, i, o):
    """Hook function called during backward pass to trigger reinitialization."""
    # 更新频域敏感度
    m.update_frequency_sensitivity(i[0])
    # 执行重初始化
    m.reinit()


def log_features(m, i, o):
    """Hook function called during forward pass to log feature activations."""
    with torch.no_grad():
        current_features = i[0]
        
        if m.decay_rate == 0:
            m.features = current_features
        else:
            if m.features is None:
                # 第一次记录特征
                m.features = (1 - m.decay_rate) * current_features
            else:
                # 检查尺寸是否匹配
                if m.features.shape == current_features.shape:
                    # 尺寸匹配，正常更新
                    m.features = m.features * m.decay_rate + (1 - m.decay_rate) * current_features
                else:
                    # 尺寸不匹配，重新初始化（可能是批次大小变化）
                    m.features = (1 - m.decay_rate) * current_features


def get_layer_bound(layer, init, gain):
    """Calculate initialization bounds for different layer types."""
    if isinstance(layer, nn.Conv2d):
        return sqrt(1 / (layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]))
    elif isinstance(layer, nn.Linear):
        if init == 'default':
            bound = sqrt(1 / layer.in_features)
        elif init == 'xavier':
            bound = gain * sqrt(6 / (layer.in_features + layer.out_features))
        elif init == 'lecun':
            bound = sqrt(3 / layer.in_features)
        else:
            bound = gain * sqrt(3 / layer.in_features)
        return bound


class CBPLinear(nn.Module):
    def __init__(
            self,
            in_layer: nn.Linear,
            out_layer: nn.Linear,
            ln_layer: nn.LayerNorm = None,
            bn_layer: nn.BatchNorm1d = None,
            replacement_rate=1e-4,
            maturity_threshold=100,
            init='kaiming',
            act_type='relu',
            util_type='contribution',
            decay_rate=0,
            # 新增频域敏感度参数
            frequency_sensitivity_enabled=True,
            lambda_freq=0.1,
            frequency_cutoff=15,
            sensitivity_update_interval=100,
            sensitivity_alpha=0.1,
    ):
        super().__init__()
        if type(in_layer) is not nn.Linear:
            raise Warning("Make sure in_layer is a weight layer")
        if type(out_layer) is not nn.Linear:
            raise Warning("Make sure out_layer is a weight layer")
        """
        Define the hyper-parameters of the algorithm
        """
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type
        self.decay_rate = decay_rate
        self.features = None
        
        # 新增频域敏感度参数
        self.frequency_sensitivity_enabled = frequency_sensitivity_enabled
        self.lambda_freq = lambda_freq
        self.frequency_cutoff = frequency_cutoff
        self.sensitivity_update_interval = sensitivity_update_interval
        self.sensitivity_alpha = sensitivity_alpha
        
        # 频域敏感度相关状态
        self.hf_sensitivity = None
        self.sensitivity_analyzer = None
        self.update_counter = 0
        """
        Register hooks
        """
        if self.replacement_rate > 0:
            self.register_full_backward_hook(call_reinit)
            self.register_forward_hook(log_features)

        self.in_layer = in_layer
        self.out_layer = out_layer
        self.ln_layer = ln_layer
        self.bn_layer = bn_layer
        """
        Utility of all features/neurons
        """
        self.util = nn.Parameter(torch.zeros(self.in_layer.out_features), requires_grad=False)
        self.ages = nn.Parameter(torch.zeros(self.in_layer.out_features), requires_grad=False)
        self.accumulated_num_features_to_replace = nn.Parameter(torch.zeros(1), requires_grad=False)
        """
        Calculate uniform distribution's bound for random feature initialization
        """
        self.bound = get_layer_bound(layer=self.in_layer, init=init, gain=nn.init.calculate_gain(nonlinearity=act_type))

    def forward(self, _input):
        return _input
    
    def update_frequency_sensitivity(self, input_data):
        """更新频域敏感度"""
        if not self.frequency_sensitivity_enabled:
            return
        
        self.update_counter += 1
        
        # 按间隔更新敏感度
        if self.update_counter % self.sensitivity_update_interval == 0:
            self._calculate_and_update_sensitivity(input_data)
    
    def _calculate_and_update_sensitivity(self, input_data):
        """计算并更新频域敏感度"""
        # 1. 初始化敏感度分析器（如果未初始化）
        if self.sensitivity_analyzer is None:
            self.sensitivity_analyzer = SensitivityAnalyzer(device=self.util.device)
        
        # 2. 获取当前激活值
        if self.features is None:
            return
        
        # 3. 计算当前批次的高频敏感度
        current_sensitivity = self.sensitivity_analyzer.calculate_high_frequency_sensitivity(
            self.features, 
            input_data,
            cutoff_freq=self.frequency_cutoff
        )
        
        # 4. 更新敏感度值（使用指数移动平均）
        if self.hf_sensitivity is None:
            self.hf_sensitivity = current_sensitivity
        else:
            # 确保尺寸匹配
            min_size = min(self.hf_sensitivity.size(0), current_sensitivity.size(0))
            self.hf_sensitivity = (1 - self.sensitivity_alpha) * self.hf_sensitivity[:min_size] + \
                                self.sensitivity_alpha * current_sensitivity[:min_size]

    def get_features_to_reinit(self):
        """
        Returns: Features to replace
        """
        features_to_replace = torch.empty(0, dtype=torch.long, device=self.util.device)
        self.ages += 1
        """
        Calculate number of features to replace
        """
        eligible_feature_indices = torch.where(self.ages > self.maturity_threshold)[0]
        if eligible_feature_indices.shape[0] == 0:  return features_to_replace

        num_new_features_to_replace = self.replacement_rate*eligible_feature_indices.shape[0]
        self.accumulated_num_features_to_replace += num_new_features_to_replace
        if self.accumulated_num_features_to_replace < 1:    return features_to_replace

        num_new_features_to_replace = int(self.accumulated_num_features_to_replace)
        self.accumulated_num_features_to_replace -= num_new_features_to_replace
        """
        Calculate feature utility with frequency sensitivity enhancement
        """
        # 1. 计算基础效用值
        output_weight_mag = self.out_layer.weight.data.abs().mean(dim=0)
        base_utility = output_weight_mag * self.features.abs().mean(dim=[i for i in range(self.features.ndim - 1)])
        
        # 2. 应用频域敏感度增强
        if self.frequency_sensitivity_enabled and self.hf_sensitivity is not None:
            # 确保尺寸匹配
            min_size = min(base_utility.size(0), self.hf_sensitivity.size(0))
            frequency_enhancement = 1 + self.lambda_freq * self.hf_sensitivity[:min_size]
            enhanced_utility = base_utility[:min_size] * frequency_enhancement
        else:
            enhanced_utility = base_utility
        
        # 3. 更新效用值（保持现有的衰减机制）
        if self.decay_rate == 0:
            self.util.data = enhanced_utility
        else:
            self.util.data = self.util.data * self.decay_rate + (1 - self.decay_rate) * enhanced_utility
        """
        Find features with smallest utility
        """
        new_features_to_replace = torch.topk(-self.util[eligible_feature_indices], num_new_features_to_replace)[1]
        new_features_to_replace = eligible_feature_indices[new_features_to_replace]
        features_to_replace = new_features_to_replace
        return features_to_replace

    def reinit_features(self, features_to_replace):
        """
        Reset input and output weights for low utility features
        """
        with torch.no_grad():
            num_features_to_replace = features_to_replace.shape[0]
            if num_features_to_replace == 0: return
            self.in_layer.weight.data[features_to_replace, :] *= 0.0
            self.in_layer.weight.data[features_to_replace, :] += \
                torch.empty(num_features_to_replace, self.in_layer.in_features, device=self.util.device).uniform_(-self.bound, self.bound)
            
            # 只有当bias存在时才重置bias
            if self.in_layer.bias is not None:
                self.in_layer.bias.data[features_to_replace] *= 0

            self.out_layer.weight.data[:, features_to_replace] = 0
            self.ages[features_to_replace] = 0

            """
            Reset the corresponding batchnorm/layernorm layers
            """
            if self.bn_layer is not None:
                self.bn_layer.bias.data[features_to_replace] = 0.0
                self.bn_layer.weight.data[features_to_replace] = 1.0
                self.bn_layer.running_mean.data[features_to_replace] = 0.0
                self.bn_layer.running_var.data[features_to_replace] = 1.0
            if self.ln_layer is not None:
                self.ln_layer.bias.data[features_to_replace] = 0.0
                self.ln_layer.weight.data[features_to_replace] = 1.0

    def reinit(self):
        """
        Perform selective reinitialization
        """
        features_to_replace = self.get_features_to_reinit()
        self.reinit_features(features_to_replace)

