"""
光伏CAD生成模型架构 - 完整可运行版本
多模态输入（图片 + 文字） → 标准CAD输出
"""

import torch
import torch.nn as nn
from transformers import ViTModel, BertModel, ViTConfig, BertConfig
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PVCAIModel(nn.Module):
    """
    Photovoltaic CAD AI Model
    多模态输入处理 + CAD生成
    支持无数据情况下的基本运行
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 图像编码器 (ViT)
        try:
            self.image_encoder = ViTModel.from_pretrained(
                config.get('vit_model', 'google/vit-base-patch16-224')
            )
            logger.info("Loaded pre-trained ViT model")
        except Exception as e:
            logger.warning(f"Failed to load pre-trained ViT, using random initialization: {e}")
            vit_config = ViTConfig()
            self.image_encoder = ViTModel(vit_config)
        
        # 文本编码器 (BERT)
        try:
            self.text_encoder = BertModel.from_pretrained(
                config.get('bert_model', 'bert-base-uncased')
            )
            logger.info("Loaded pre-trained BERT model")
        except Exception as e:
            logger.warning(f"Failed to load pre-trained BERT, using random initialization: {e}")
            bert_config = BertConfig()
            self.text_encoder = BertModel(bert_config)
        
        # 特征融合层
        fusion_input_size = (
            self.image_encoder.config.hidden_size + 
            self.text_encoder.config.hidden_size
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_size, config.get('fusion_hidden_size', 768)),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(config.get('fusion_hidden_size', 768), 512),
            nn.ReLU()
        )
        
        # CAD生成器
        self.cad_generator = CADGenerator(config)
        
        # 输出格式配置
        self.output_format = config.get('output_format', 'dxf')
        
    def forward(self, image_input: torch.Tensor, text_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            image_input: [batch_size, 3, 224, 224] - 手稿图片
            text_input: 包含 input_ids, attention_mask 的字典
            
        Returns:
            CAD生成结果字典
        """
        # 图像特征提取
        image_outputs = self.image_encoder(image_input)
        image_features = image_outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # 文本特征提取
        text_outputs = self.text_encoder(
            input_ids=text_input['input_ids'],
            attention_mask=text_input.get('attention_mask', None)
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # 特征融合
        combined_features = torch.cat([image_features, text_features], dim=-1)
        fused_features = self.fusion_layer(combined_features)
        
        # CAD生成
        cad_output = self.cad_generator(fused_features)
        
        return cad_output


class CADGenerator(nn.Module):
    """
    CAD生成器模块
    将融合特征转换为CAD格式输出
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 几何数据生成 (vertices, edges, faces)
        self.geometry_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # 输出64维几何特征
        )
        
        # 组件信息生成 (panels, inverters, mounting, wiring)
        self.components_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32)  # 输出32维组件特征
        )
        
        # 布局参数生成 (roof_area, panel_count, orientation, tilt_angle)
        self.layout_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)  # 输出16维布局参数
        )
        
        # 激活函数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        生成CAD数据
        
        Args:
            features: [batch_size, 512] - 融合特征
            
        Returns:
            CAD数据字典
        """
        geometry_data = self.geometry_head(features)
        components_data = self.components_head(features)
        layout_data = self.layout_head(features)
        
        # 应用适当的激活函数
        # 几何数据使用tanh (-1, 1)
        # 组件数据使用sigmoid (0, 1)
        # 布局参数混合使用
        layout_params = torch.zeros_like(layout_data)
        layout_params[:, 0] = self.tanh(layout_data[:, 0]) * 1000  # roof_area (0-1000 m²)
        layout_params[:, 1] = self.sigmoid(layout_data[:, 1]) * 100  # panel_count (0-100)
        layout_params[:, 2] = self.tanh(layout_data[:, 2])  # orientation (-1, 1) -> N/S
        layout_params[:, 3] = self.sigmoid(layout_data[:, 3]) * 90  # tilt_angle (0-90°)
        
        return {
            'geometry': geometry_data,
            'components': components_data,
            'layout': layout_params,
            'raw_geometry': geometry_data,
            'raw_components': components_data,
            'raw_layout': layout_data
        }


# 模型配置
DEFAULT_CONFIG = {
    'vit_model': 'google/vit-base-patch16-224',
    'bert_model': 'bert-base-uncased',
    'fusion_hidden_size': 768,
    'dropout': 0.1,
    'output_format': 'dxf',
    'image_size': 224,
    'max_text_length': 128
}


def create_model(config: Optional[Dict[str, Any]] = None) -> PVCAIModel:
    """创建模型实例"""
    if config is None:
        config = DEFAULT_CONFIG.copy()
    return PVCAIModel(config)


def test_model():
    """测试模型是否能正常运行"""
    print("Testing model initialization...")
    
    # 创建模型
    model = create_model()
    print("✓ Model created successfully")
    
    # 创建测试输入
    batch_size = 2
    image_input = torch.randn(batch_size, 3, 224, 224)
    text_input = {
        'input_ids': torch.randint(0, 30522, (batch_size, 128)),
        'attention_mask': torch.ones(batch_size, 128)
    }
    
    # 前向传播
    with torch.no_grad():
        output = model(image_input, text_input)
    
    print(f"✓ Forward pass successful")
    print(f"  Geometry shape: {output['geometry'].shape}")
    print(f"  Components shape: {output['components'].shape}")
    print(f"  Layout shape: {output['layout'].shape}")
    
    return model, output


if __name__ == "__main__":
    # 运行测试
    test_model()
    print("✅ Model architecture test completed!")