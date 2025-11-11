# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from .frequency_filter import FrequencyFilter
from .nac_calculator import NACCalculator


class SensitivityAnalyzer:
    """
    频域敏感度分析器
    实现完整的NAC分析流程
    """
    
    def __init__(self, device: str = 'cuda', model=None, target_layer_name=None):
        """
        初始化分析器
        
        Args:
            device: 设备类型
            model: 模型实例（用于重新前向传播）
            target_layer_name: 目标层名称（用于获取激活值）
        """
        self.device = device
        self.frequency_filter = FrequencyFilter(device=device)
        self.nac_calculator = NACCalculator()
        self.model = model
        self.target_layer_name = target_layer_name
        self.activation_hook = None
        self.captured_activation = None
    
    def _activation_hook(self, module, input, output):
        """钩子函数，用于捕获目标层的激活值"""
        self.captured_activation = output.detach().clone()
    
    def _register_hook(self):
        """注册钩子函数"""
        self.captured_activation = None
        if self.model is not None and self.target_layer_name is not None:
            # 查找目标层
            target_module = None
            for name, module in self.model.named_modules():
                if name == self.target_layer_name:
                    target_module = module
                    break
            
            if target_module is not None:
                self.activation_hook = target_module.register_forward_hook(self._activation_hook)
            else:
                print(f"Warning: Target layer '{self.target_layer_name}' not found in model")
    
    def _remove_hook(self):
        """移除钩子函数"""
        if self.activation_hook is not None:
            self.activation_hook.remove()
            self.activation_hook = None
    
    def calculate_high_frequency_sensitivity(self, 
                                           activations: torch.Tensor, 
                                           input_data: torch.Tensor,
                                           cutoff_freq: float = 15,
                                           aggregation: str = 'mean') -> torch.Tensor:
        """
        计算高频敏感度NAC值
        
        Args:
            activations: 当前层的激活值，形状为 (B, C, H, W) 或 (B, C)
            input_data: 输入数据，形状为 (B, C, H, W)
            cutoff_freq: 低通滤波截止频率，默认15
            aggregation: 聚合方式
        
        Returns:
            sensitivity: 高频敏感度NAC值，形状为 (C,)
        """
       # print(f"敏感度分析器 - 输入数据形状: {input_data.shape}, 激活值形状: {activations.shape}")
        
        with torch.no_grad():
            # 1. 对输入数据应用低通滤波
            #print(f"开始应用低通滤波，截止频率: {cutoff_freq}")
            filtered_input = self.frequency_filter.apply_lowpass_filter_batch(
                input_data, cutoff_freq
            )
           # print(f"低通滤波完成，滤波后形状: {filtered_input.shape}")
            
            # 2. 重新进行前向传播获取滤波后的激活值
            if self.model is not None and self.target_layer_name is not None:
                #print(f"使用模型重新前向传播，目标层: {self.target_layer_name}")
                # 保存当前模型状态
                was_training = self.model.training
                
                # 注册钩子函数
                self._register_hook()
                
                try:
                    # 使用滤波后的输入重新进行前向传播
                    self.model.eval()
                    _ = self.model(filtered_input)
                    
                    # 获取滤波后的激活值
                    if self.captured_activation is not None:
                        filtered_activations = self.captured_activation
                        #print(f"成功捕获滤波后激活值，形状: {filtered_activations.shape}")
                    else:
                        # 如果钩子函数失败，回退到估算方法
                        print("Warning: Failed to capture activation, using estimation method")
                        filtered_activations = self._estimate_filtered_activations(activations, input_data, filtered_input)
                        print(f"使用估算方法，滤波后激活值形状: {filtered_activations.shape}")
                finally:
                    # 移除钩子函数
                    self._remove_hook()
                    # 恢复模型状态
                    if was_training:
                        self.model.train()
            else:
                # 如果没有模型信息，使用估算方法
                print("没有模型信息，使用估算方法")
                filtered_activations = self._estimate_filtered_activations(activations, input_data, filtered_input)
                print(f"估算方法结果，滤波后激活值形状: {filtered_activations.shape}")
            
            # 3. 计算NAC值
            #print("开始计算NAC值")
            nac_values = self.nac_calculator.calculate_nac_for_layer(
                activations, filtered_activations, aggregation
            )
            #print(f"NAC值计算完成，形状: {nac_values.shape}, 均值: {nac_values.mean().item():.6f}")
            
            return nac_values
    
    def _estimate_filtered_activations(self, activations, input_data, filtered_input):
        """估算滤波后的激活值（备用方法）"""
        # 统一计算输入差异比例（按 batch 的均值缩放成标量，避免通道数不匹配）
        input_diff = torch.abs(input_data - filtered_input).mean(dim=(2, 3))  # (B, C_in)
        input_base = input_data.abs().mean(dim=(2, 3)) + 1e-6  # (B, C_in)
        diff_ratio = (input_diff / input_base).mean(dim=1, keepdim=True)  # (B, 1)

        if activations.dim() == 4:  # 卷积层
            diff_ratio = diff_ratio.unsqueeze(-1).unsqueeze(-1)  # (B,1,1,1)
            filtered_activations = activations - activations * diff_ratio
        else:  # 全连接层
            filtered_activations = activations - activations * diff_ratio
        
        return filtered_activations
    
    def calculate_sensitivity_batch(self, 
                                  activations_dict: Dict[str, torch.Tensor],
                                  input_data: torch.Tensor,
                                  cutoff_freq: float = 15,
                                  aggregation: str = 'mean') -> Dict[str, torch.Tensor]:
        """
        批量计算敏感度
        
        Args:
            activations_dict: 激活值字典 {layer_name: tensor}
            input_data: 输入数据
            cutoff_freq: 低通滤波截止频率
            aggregation: 聚合方式
        
        Returns:
            sensitivity_dict: 敏感度字典
        """
        sensitivity_dict = {}
        
        for layer_name, activations in activations_dict.items():
            sensitivity = self.calculate_high_frequency_sensitivity(
                activations, input_data, cutoff_freq, aggregation
            )
            sensitivity_dict[layer_name] = sensitivity
        
        return sensitivity_dict
    
    def get_sensitivity_statistics(self, sensitivity: torch.Tensor) -> Dict[str, float]:
        """
        获取敏感度统计信息
        
        Args:
            sensitivity: 敏感度值
        
        Returns:
            statistics: 统计信息
        """
        return self.nac_calculator.calculate_nac_statistics(sensitivity)

