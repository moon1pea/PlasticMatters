# -*- coding: utf-8 -*-


import torch
from typing import Dict, List, Tuple, Optional


class NACCalculator:
    """NAC计算器类"""
    
    def __init__(self, epsilon: float = 1e-6):
        """
        初始化NAC计算器
        
        Args:
            epsilon: 防止除零的小常数
        """
        self.epsilon = epsilon
    
    def calculate_nac_per_image(self, 
                              original_activation: torch.Tensor, 
                              perturbed_activation: torch.Tensor, 
                              aggregation: str = 'mean') -> torch.Tensor:
        """
        计算单张图像的NAC值：计算这张图像的一个通道的 NAC 值
        
        Args:
            original_activation: 原始激活值，形状为 (H, W)
            perturbed_activation: 扰动后激活值，形状为 (H, W)
            aggregation: 聚合方式，'mean' 或 'median'
        
        Returns:
            nac_value: 归一化绝对差值
        """
        # 1. 空间汇聚成标量 (mean 或 median)
        if aggregation == 'mean':
            a_bar = torch.mean(original_activation, dim=(-2, -1))
            a_B_bar = torch.mean(perturbed_activation, dim=(-2, -1))
        elif aggregation == 'median':
            a_bar = torch.median(original_activation.view(-1), dim=0)[0]
            a_B_bar = torch.median(perturbed_activation.view(-1), dim=0)[0]
        else:
            raise ValueError("aggregation must be 'mean' or 'median'")
        
        # 2. 归一化绝对差
        abs_diff = torch.abs(a_bar - a_B_bar)
        orig_magnitude = torch.abs(a_bar)
        nac_value = abs_diff / (orig_magnitude + self.epsilon)
        
        return nac_value
    
    def calculate_nac_for_channel(self, 
                                 original_activations: torch.Tensor, 
                                 perturbed_activations: torch.Tensor, 
                                 aggregation: str = 'mean') -> torch.Tensor:
        """
        计算单个通道的NAC值：计算这个通道（神经元族）在整个 batch 上的 NAC
        
        Args:
            original_activations: 原始激活值，形状为 (B, H, W)
            perturbed_activations: 扰动后激活值，形状为 (B, H, W)
            aggregation: 聚合方式
        
        Returns:
            channel_nac: 通道NAC值
        """
        batch_size = original_activations.size(0)
        
        # 对批次中的每张图像计算NAC
        image_nac_values = []
        for n in range(batch_size):
            nac_per_image = self.calculate_nac_per_image(
                original_activations[n], 
                perturbed_activations[n], 
                aggregation
            )
            image_nac_values.append(nac_per_image)
        
        # 对批次取均值
        channel_nac = torch.mean(torch.stack(image_nac_values))
        
        return channel_nac
    
    def calculate_nac_for_layer(self, 
                            original_activations: torch.Tensor, 
                            perturbed_activations: torch.Tensor, 
                            aggregation: str = 'mean') -> torch.Tensor:
        """
        计算单个层的所有通道NAC值：计算该层每个通道的 NAC，返回 [C] 维度结果
        
        Args:
            original_activations: 原始激活值，形状为 (B, C, H, W) 或 (B, C)
            perturbed_activations: 扰动后激活值，形状为 (B, C, H, W) 或 (B, C)
            aggregation: 聚合方式
        
        Returns:
            layer_nac_values: 该层所有通道的NAC值
        """
        # 处理不同维度的激活值
        if original_activations.dim() == 2:  # 全连接层 (B, C)
            num_channels = original_activations.size(1)
            layer_nac_values = []
            
            for c in range(num_channels):
                # 提取第c个通道的激活值
                orig_channel = original_activations[:, c]  # 形状: (B,)
                pert_channel = perturbed_activations[:, c]
                
                # 计算该通道的NAC值
                channel_nac = self.calculate_nac_for_channel(
                    orig_channel.unsqueeze(-1).unsqueeze(-1),  # 扩展为 (B, 1, 1)
                    pert_channel.unsqueeze(-1).unsqueeze(-1),
                    aggregation
                )
                layer_nac_values.append(channel_nac)
            
            return torch.stack(layer_nac_values)
            
        elif original_activations.dim() == 4:  # 卷积层 (B, C, H, W)
            num_channels = original_activations.size(1)
            layer_nac_values = []
            
            for c in range(num_channels):
                # 提取第c个通道的激活值
                orig_channel = original_activations[:, c, :, :]  # 形状: (B, H, W)
                pert_channel = perturbed_activations[:, c, :, :]
                
                # 计算该通道的NAC值
                channel_nac = self.calculate_nac_for_channel(
                    orig_channel, pert_channel, aggregation
                )
                layer_nac_values.append(channel_nac)
            
            return torch.stack(layer_nac_values)
        else:
            raise ValueError(f"Unsupported activation tensor dimension: {original_activations.dim()}")
    
    def calculate_nac_statistics(self, nac_values: torch.Tensor) -> Dict[str, float]:
        """
        计算NAC值的统计信息
        
        Args:
            nac_values: NAC值张量
        
        Returns:
            statistics: 统计信息字典
        """
        if nac_values.numel() == 0:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
        
        statistics = {
            'mean': torch.mean(nac_values).item(),
            'std': torch.std(nac_values).item(),
            'min': torch.min(nac_values).item(),
            'max': torch.max(nac_values).item(),
            'median': torch.median(nac_values).item(),
            'num_channels': nac_values.size(0)
        }
        
        return statistics

