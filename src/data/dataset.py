"""
光伏CAD数据集类
支持手稿图片 + 文字描述的数据加载
"""

import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import json
import numpy as np

class PVCAIDataset(Dataset):
    """
    Photovoltaic CAD AI Dataset
    支持手稿图片 + 文字描述的数据集
    """
    
    def __init__(self, data_dir, transform=None, tokenizer=None, max_length=128):
        self.data_dir = data_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据索引
        self.data_list = self._load_data_index()
        
    def _load_data_index(self):
        """加载数据索引"""
        data_list = []
        
        # 如果有示例数据，加载示例数据
        examples_dir = os.path.join(self.data_dir, 'examples')
        if os.path.exists(examples_dir):
            for filename in os.listdir(examples_dir):
                if filename.endswith('.json'):
                    data_list.append(os.path.join(examples_dir, filename))
        
        # 如果没有示例数据，创建虚拟数据用于测试
        if not data_list:
            # 创建虚拟数据条目
            data_list = ['dummy_data_1', 'dummy_data_2']
            
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """获取单个数据项"""
        if isinstance(self.data_list[idx], str) and self.data_list[idx].startswith('dummy_data'):
            # 返回虚拟数据（用于无数据情况下的测试）
            return self._get_dummy_data()
        else:
            # 返回真实数据
            return self._load_real_data(idx)
    
    def _get_dummy_data(self):
        """生成虚拟数据用于测试"""
        # 虚拟手稿图片 (224x224 RGB)
        sketch_image = torch.randn(3, 224, 224)
        
        # 虚拟文字描述
        text_description = "Photovoltaic system design for residential roof"
        
        # 虚拟CAD输出
        cad_output = {
            'geometry': torch.randn(64),
            'components': torch.randn(32),
            'layout': torch.randn(16)
        }
        
        # 如果有tokenizer，对文本进行编码
        if self.tokenizer:
            text_encoded = self.tokenizer(
                text_description,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            text_input_ids = text_encoded['input_ids'].squeeze()
            text_attention_mask = text_encoded['attention_mask'].squeeze()
        else:
            text_input_ids = torch.zeros(self.max_length, dtype=torch.long)
            text_attention_mask = torch.ones(self.max_length, dtype=torch.long)
        
        return {
            'image': sketch_image,
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'cad_output': cad_output
        }
    
    def _load_real_data(self, idx):
        """加载真实数据"""
        # 这里可以实现真实数据的加载逻辑
        # 为简化，先返回虚拟数据
        return self._get_dummy_data()

# 确保类被正确导出
__all__ = ['PVCAIDataset']