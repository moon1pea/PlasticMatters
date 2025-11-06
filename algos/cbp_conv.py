# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn.init import calculate_gain
from math import sqrt
from .frequency_sensitivity import SensitivityAnalyzer


def call_reinit(m, i, o):
    """Hook function called during backward pass to trigger reinitialization.
    backward hook，在反向传播时触发。"""
    # 记录全局反向步数，用于控制前若干步不进行重置
    if not hasattr(m, 'global_step'):
        m.global_step = 0
    m.global_step += 1

    # 更新频域敏感度
    # 注意：i[0] 是当前层的输入，不是原始图像输入
    # 我们需要从模型的最开始获取原始输入
    m.update_frequency_sensitivity(i[0])
    # 在达到最小重置步数之前，不执行重初始化
    min_reset_step = getattr(m, 'min_reset_step', 0)
    if m.global_step < min_reset_step:
        return

    # 执行重初始化
    m.reinit()


def log_features(m, i, o):
    """Hook function called during forward pass to log feature activations.
    forward hook，在前向传播时记录激活值"""
    with torch.no_grad():
        current_features = i[0]
        
        # 存储原始输入（如果是第一层或输入是3通道图像或30通道HPF输出）
        if m._original_input is None or (current_features.dim() == 4 and (current_features.size(1) == 3 or current_features.size(1) == 30)):
            m._original_input = current_features.clone()
        
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
    """Calculate initialization bounds for different layer types.
    根据层类型（卷积/全连接）和初始化方式（default/xavier/lecun）计算 权重初始化范围。"""
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


class CBPConv(nn.Module):
    def __init__(
            self,
            in_layer: nn.Conv2d,
            out_layer: [nn.Conv2d, nn.Linear],
            ln_layer: nn.LayerNorm = None,
            bn_layer: nn.BatchNorm2d = None,
            num_last_filter_outputs=1,
            replacement_rate=1e-5,
            maturity_threshold=1000,#神经元"成熟"多少步后允许被重置
            init='kaiming',
            act_type='relu',
            util_type='contribution',
            decay_rate=0,
            frequency_sensitivity_enabled=True,#是否启用频域敏感度
            lambda_freq=0.1,
            frequency_cutoff=15,#频域敏感度分析的截止频率
            sensitivity_update_interval=100,#频域敏感度更新的间隔
            sensitivity_alpha=0.1,#频域敏感度更新的权重λ
    ):
        super().__init__()
        if type(in_layer) is not nn.Conv2d:
            raise Warning("Make sure in_layer is a convolutional layer")
        if type(out_layer) not in [nn.Linear, nn.Conv2d]:
            raise Warning("Make sure out_layer is a convolutional or linear layer")

        """
        Define the hyper-parameters of the algorithm
        CBP原超参数
        """
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type
        self.decay_rate = decay_rate
        self.features = None
        self.num_last_filter_outputs = num_last_filter_outputs
        
        # 新增频域敏感度参数
        self.frequency_sensitivity_enabled = frequency_sensitivity_enabled
        self.lambda_freq = lambda_freq
        self.frequency_cutoff = frequency_cutoff
        self.sensitivity_update_interval = sensitivity_update_interval
        self.sensitivity_alpha = sensitivity_alpha
        
        # 频域敏感度相关状态
        self.hf_sensitivity = None#当前层的高频敏感度值（与每个通道对应）
        self.sensitivity_analyzer = None#敏感度分析器，用于计算高频敏感度
        self.update_counter = 0
        self._model_ref = None  # 用于重新前向传播的模型引用（弱引用）
        self.layer_name = None  # 当前层的名称（用于获取激活值）
        self._original_input = None  # 存储原始输入（用于频域敏感度计算）

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
        self.util = nn.Parameter(torch.zeros(self.in_layer.out_channels), requires_grad=False)
        self.ages = nn.Parameter(torch.zeros(self.in_layer.out_channels), requires_grad=False)
        self.accumulated_num_features_to_replace = nn.Parameter(torch.zeros(1), requires_grad=False)
        """
        Calculate uniform distribution's bound for random feature initialization
        """
        self.bound = get_layer_bound(layer=self.in_layer, init=init, gain=calculate_gain(nonlinearity=act_type))

        # 控制重置的最小步数与全局步计数（用于"前N步不替换"）
        self.global_step = 0
        # 默认20000步前不进行替换，可根据需要在外部修改该属性
        self.min_reset_step = 20000

    def forward(self, _input):
        return _input
    
    def set_model_info(self, model, layer_name):
        """设置模型和层信息，用于重新前向传播（用于计算频域敏感度）"""
        # 使用弱引用避免循环引用
        import weakref
        self._model_ref = weakref.ref(model)
        self.layer_name = layer_name
    
    def update_frequency_sensitivity(self, input_data):
        """更新频域敏感度"""
        if not self.frequency_sensitivity_enabled:
            return#如果未启用频域敏感度，直接返回
        
        self.update_counter += 1
        
        # 按间隔更新敏感度
        if self.update_counter % self.sensitivity_update_interval == 0:
            self._calculate_and_update_sensitivity(input_data)
    
    def _calculate_and_update_sensitivity(self, input_data):
        
        #print("进入计算频域敏感度更新函数")
        
        """计算并更新频域敏感度"""
        # 1. 初始化敏感度分析器（如果未初始化）
        if self.sensitivity_analyzer is None:
            # 获取模型引用（如果存在）
            model_ref = self._model_ref() if self._model_ref is not None else None
            #print(f"初始化敏感度分析器，模型引用: {model_ref is not None}, 目标层: {self.layer_name}")
            self.sensitivity_analyzer = SensitivityAnalyzer(
                device=self.util.device, 
                model=model_ref, 
                target_layer_name=self.layer_name
            )
        
        # 2. 获取当前激活值
        if self.features is None:
            print("Warning: features is None, skipping frequency sensitivity calculation")
            return
        
        #print(f"当前激活值形状: {self.features.shape}")
        
        # 3. 从模型中获取原始输入（3通道或30通道）
        model_ref = self._model_ref() if self._model_ref is not None else None
        if model_ref is not None and hasattr(model_ref, '_original_input'):
            original_input = model_ref._original_input
            #print(f"从模型引用获取原始输入，形状: {original_input.shape}")
        elif self._original_input is not None and self._original_input.dim() == 4 and (self._original_input.size(1) == 3 or self._original_input.size(1) == 30):
            # 回退到本地存储的原始输入
            original_input = self._original_input
            #print(f"从本地存储获取原始输入，形状: {original_input.shape}")
        else:
            # 如果没有原始输入，跳过频域敏感度计算
            print("Warning: No original input available for frequency sensitivity calculation")
            print(f"  model_ref: {model_ref is not None}")
            if model_ref is not None:
                print(f"  hasattr(model_ref, '_original_input'): {hasattr(model_ref, '_original_input')}")
            print(f"  self._original_input: {self._original_input is not None}")
            if self._original_input is not None:
                print(f"  self._original_input.shape: {self._original_input.shape}")
            return
        
        # 4. 计算当前批次的高频敏感度
        try:
            #print(f"开始计算高频敏感度，截止频率: {self.frequency_cutoff}")
            current_sensitivity = self.sensitivity_analyzer.calculate_high_frequency_sensitivity(
                self.features, 
                original_input,
                cutoff_freq=self.frequency_cutoff
            )
            #print(f"高频敏感度计算成功，形状: {current_sensitivity.shape}")
            #print(f"敏感度统计 - 均值: {current_sensitivity.mean().item():.6f}, 标准差: {current_sensitivity.std().item():.6f}")
            #print(f"敏感度范围: [{current_sensitivity.min().item():.6f}, {current_sensitivity.max().item():.6f}]")
        except Exception as e:
            print(f"Warning: Failed to calculate frequency sensitivity: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 5. 更新敏感度值（使用指数移动平均）
        if self.hf_sensitivity is None:
            self.hf_sensitivity = current_sensitivity
            #print(f"初始化敏感度值，形状: {self.hf_sensitivity.shape}")
        else:
            # 确保尺寸匹配
            min_size = min(self.hf_sensitivity.size(0), current_sensitivity.size(0))
            old_sensitivity = self.hf_sensitivity.clone()
            self.hf_sensitivity = (1 - self.sensitivity_alpha) * self.hf_sensitivity[:min_size] + \
                                self.sensitivity_alpha * current_sensitivity[:min_size]
           # print(f"更新敏感度值，更新前均值: {old_sensitivity.mean().item():.6f}, 更新后均值: {self.hf_sensitivity.mean().item():.6f}")
        
       # print(f"频域敏感度更新完成，最终形状: {self.hf_sensitivity.shape}")

    def get_features_to_reinit(self):
        """
        Returns: Features to replace输入层和输出层 上待重置的神经元。初始化空张量，年龄累加
        """
        features_to_replace_input_indices = torch.empty(0, dtype=torch.long, device=self.util.device)
        features_to_replace_output_indices = torch.empty(0, dtype=torch.long, device=self.util.device)
        self.ages += 1
        """
        Calculate number of features to replace
        """
        eligible_feature_indices = torch.where(self.ages > self.maturity_threshold)[0]
        if eligible_feature_indices.shape[0] == 0:  return features_to_replace_input_indices, features_to_replace_output_indices

        num_new_features_to_replace = self.replacement_rate*eligible_feature_indices.shape[0]
        self.accumulated_num_features_to_replace += num_new_features_to_replace
        if self.accumulated_num_features_to_replace < 1:    return features_to_replace_input_indices, features_to_replace_output_indices

        num_new_features_to_replace = int(self.accumulated_num_features_to_replace)
        self.accumulated_num_features_to_replace -= num_new_features_to_replace
        """
        Calculate feature utility with frequency sensitivity enhancement
        """
        # 1. 计算基础效用值（不使用频域敏感度增强）
        if isinstance(self.out_layer, torch.nn.Linear):
            # 处理卷积层到全连接层的转换
            output_weight_mag = self.out_layer.weight.data.abs().mean(dim=0)
            
            # 计算特征的平均值，确保尺寸匹配
            if self.features.dim() == 4:  # [batch, channels, height, width]
                features_mean = self.features.abs().mean(dim=(0, 2, 3))  # [channels]
            elif self.features.dim() == 2:  # [batch, features]
                features_mean = self.features.abs().mean(dim=0)  # [features]
            else:
                features_mean = self.features.abs().mean()
                features_mean = features_mean.expand(output_weight_mag.size(0))
            
            # 确保尺寸匹配
            if output_weight_mag.size(0) == features_mean.size(0):
                base_utility = output_weight_mag * features_mean
            else:
                min_size = min(output_weight_mag.size(0), features_mean.size(0))
                base_utility = output_weight_mag[:min_size] * features_mean[:min_size]
            
        elif isinstance(self.out_layer, torch.nn.Conv2d):
            # 处理卷积层到卷积层的转换
            output_weight_mag = self.out_layer.weight.data.abs().mean(dim=(0, 2, 3))
            features_mean = self.features.abs().mean(dim=(0, 2, 3))
            
            min_size = min(output_weight_mag.size(0), features_mean.size(0))
            base_utility = output_weight_mag[:min_size] * features_mean[:min_size]
        
        # 2. 应用频域敏感度增强
        if self.frequency_sensitivity_enabled and self.hf_sensitivity is not None:
            # 确保尺寸匹配
            min_size = min(base_utility.size(0), self.hf_sensitivity.size(0))
            frequency_enhancement = 1 + self.lambda_freq * self.hf_sensitivity[:min_size]
            enhanced_utility = base_utility[:min_size] * frequency_enhancement
        else:
            enhanced_utility = base_utility
            
        self.util.data = enhanced_utility
        
        """
        Find features with smallest utility
        """
        new_features_to_replace = torch.topk(-self.util[eligible_feature_indices], num_new_features_to_replace)[1]
        new_features_to_replace = eligible_feature_indices[new_features_to_replace]
        features_to_replace_input_indices, features_to_replace_output_indices = new_features_to_replace, new_features_to_replace

        if isinstance(self.in_layer, torch.nn.Conv2d) and isinstance(self.out_layer, torch.nn.Linear):
            # 确保所有张量都在同一个设备上
            device = new_features_to_replace.device
            features_to_replace_output_indices = (
                    (new_features_to_replace * self.num_last_filter_outputs).repeat_interleave(self.num_last_filter_outputs) +
                    torch.tensor([i for i in range(self.num_last_filter_outputs)], device=device).repeat(new_features_to_replace.size()[0]))
        return features_to_replace_input_indices, features_to_replace_output_indices

    def reinit_features(self, features_to_replace_input_indices, features_to_replace_output_indices):
        """
        Reset input and output weights for low utility features
        实际重置指定神经元的权重和归一化层参数
        """
        with torch.no_grad():
            num_features_to_replace = features_to_replace_input_indices.shape[0]
            if num_features_to_replace == 0: return
            #重置输入层权重:先置零，再用均匀随机值重新初始化。
            self.in_layer.weight.data[features_to_replace_input_indices, :] *= 0.0
            
            #print("重置神经元发生")
            
            # noinspection PyArgumentList
            self.in_layer.weight.data[features_to_replace_input_indices, :] += \
                torch.empty([num_features_to_replace] + list(self.in_layer.weight.shape[1:]), device=self.util.device).uniform_(-self.bound, self.bound)
            
            #重置输入层偏置:如果存在则置零
            # 只有当bias存在时才重置bias
            if self.in_layer.bias is not None:
                self.in_layer.bias.data[features_to_replace_input_indices] *= 0

            #重置输出层权重:先置零，再用均匀随机值重新初始化。
            self.out_layer.weight.data[:, features_to_replace_output_indices] = 0
            
            #重置年龄:重置为0
            self.ages[features_to_replace_input_indices] = 0

            """
            Reset the corresponding batchnorm/layernorm layers
            """
            if self.bn_layer is not None:
                self.bn_layer.bias.data[features_to_replace_input_indices] = 0.0
                self.bn_layer.weight.data[features_to_replace_input_indices] = 1.0
                self.bn_layer.running_mean.data[features_to_replace_input_indices] = 0.0
                self.bn_layer.running_var.data[features_to_replace_input_indices] = 1.0
            if self.ln_layer is not None:
                self.ln_layer.bias.data[features_to_replace_input_indices] = 0.0
                self.ln_layer.weight.data[features_to_replace_input_indices] = 1.0

    def reinit(self):
        """
        Perform selective reinitialization
        """
        features_to_replace_input_indices, features_to_replace_output_indices = self.get_features_to_reinit()
        self.reinit_features(features_to_replace_input_indices, features_to_replace_output_indices)

