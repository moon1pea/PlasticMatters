# -*- coding: utf-8 -*-


import torch
import numpy as np
from typing import Tuple


class FrequencyFilter:
    """频域滤波器类"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def create_lowpass_mask(self, image_shape: Tuple[int, int], cutoff_freq: float) -> torch.Tensor:
        """
        创建低通滤波器掩码
        
        Args:
            image_shape: 图像形状 (H, W)
            cutoff_freq: 截止频率
            
        Returns:
            mask: 低通滤波器掩码
        """
        h, w = image_shape
        center_h, center_w = h // 2, w // 2
        
        # 创建坐标网格
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        distance = torch.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        # 低通滤波器：保留距离中心小于等于cutoff_freq的频率
        mask = distance <= cutoff_freq
        
        return mask.to(self.device)
    
    def apply_lowpass_filter(self, image: torch.Tensor, cutoff_freq: float = 15) -> torch.Tensor:
        """
        对图像应用低通滤波器
        
        Args:
            image: 输入图像，形状为 (C, H, W)
            cutoff_freq: 截止频率，默认15
            
        Returns:
            filtered_image: 滤波后的图像
        """
        filtered_channels = []
        
        for c in range(image.size(0)):
            channel = image[c]  # 形状: (H, W)
            
            # FFT变换
            fft = torch.fft.fft2(channel, dim=(-2, -1))
            fft_shifted = torch.fft.fftshift(fft)
            
            # 创建低通滤波器掩码
            mask = self.create_lowpass_mask(channel.shape, cutoff_freq)
            
            # 应用掩码
            filtered_fft = fft_shifted * mask
            
            # 逆FFT变换
            filtered_fft_shifted = torch.fft.ifftshift(filtered_fft)
            filtered_channel = torch.fft.ifft2(filtered_fft_shifted, dim=(-2, -1))
            
            # 取实部
            filtered_channel = filtered_channel.real
            filtered_channels.append(filtered_channel)
        
        return torch.stack(filtered_channels, dim=0)
    
    def apply_lowpass_filter_batch(self, images: torch.Tensor, cutoff_freq: float = 15) -> torch.Tensor:
        """
        对批次图像应用低通滤波器
        
        Args:
            images: 输入图像批次，形状为 (B, C, H, W)
            cutoff_freq: 截止频率，默认15
            
        Returns:
            filtered_images: 滤波后的图像批次
        """
        filtered_images = []
        
        for i in range(images.size(0)):
            image = images[i]  # 形状: (C, H, W)
            filtered_image = self.apply_lowpass_filter(image, cutoff_freq)
            filtered_images.append(filtered_image)
        
        return torch.stack(filtered_images, dim=0)

