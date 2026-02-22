"""
光伏CAD推理模块
支持手稿图片 + 文字描述输入，生成标准CAD输出
"""

import torch
from transformers import ViTImageProcessor, BertTokenizer
from src.model.architecture import create_model
from src.data.dataset import PVCAIDataset
import os
import json
from typing import Dict, Any, Optional

class PVCAIInference:
    """
    Photovoltaic CAD AI Inference Class
    """
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型配置
        if config is None:
            config = {
                'vit_model': 'google/vit-base-patch16-224',
                'bert_model': 'bert-base-uncased',
                'fusion_hidden_size': 768,
                'dropout': 0.1,
                'output_format': 'dxf'
            }
        
        # 创建模型
        self.model = create_model(config)
        self.model.to(self.device)
        self.model.eval()
        
        # 加载预训练权重（如果提供）
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded model from {model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load model weights: {e}")
                print("Using randomly initialized model")
        
        # 初始化处理器
        self.image_processor = ViTImageProcessor.from_pretrained(
            config.get('vit_model', 'google/vit-base-patch16-224')
        )
        self.tokenizer = BertTokenizer.from_pretrained(
            config.get('bert_model', 'bert-base-uncased')
        )
        
        print("✅ PVCAI Inference initialized successfully")
    
    def preprocess_inputs(self, sketch_image_path: str, text_description: str) -> Dict[str, torch.Tensor]:
        """
        预处理输入数据
        
        Args:
            sketch_image_path: 手稿图片路径
            text_description: 文字描述
            
        Returns:
            处理后的输入字典
        """
        from PIL import Image
        import torchvision.transforms as transforms
        
        # 处理图片
        image = Image.open(sketch_image_path).convert('RGB')
        # 使用 ViTImageProcessor 处理图片
        image_tensor = self.image_processor(image, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.squeeze(0)  # 移除 batch 维度
        
        # 处理文本
        text_encoded = self.tokenizer(
            text_description,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        input_ids = text_encoded['input_ids'].squeeze(0)
        attention_mask = text_encoded['attention_mask'].squeeze(0)
        
        return {
            'image': image_tensor,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def generate_cad(self, sketch_image_path: str, text_description: str) -> Dict[str, Any]:
        """
        生成CAD设计
        
        Args:
            sketch_image_path: 手稿图片路径
            text_description: 文字描述
            
        Returns:
            CAD生成结果
        """
        # 预处理输入
        inputs = self.preprocess_inputs(sketch_image_path, text_description)
        
        # 准备批处理输入
        image_batch = inputs['image'].unsqueeze(0).to(self.device)
        text_batch = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(self.device),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(self.device)
        }
        
        # 推理
        with torch.no_grad():
            output = self.model(image_batch, text_batch)
        
        # 转换为CPU并移除batch维度
        result = {}
        for key, value in output.items():
            result[key] = value.cpu().squeeze(0).numpy().tolist()
        
        return result
    
    def save_cad_output(self, cad_data: Dict[str, Any], output_path: str, format: str = 'json'):
        """
        保存CAD输出
        
        Args:
            cad_data: CAD数据
            output_path: 输出路径
            format: 输出格式 ('json', 'dxf', 'svg')
        """
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(cad_data, f, indent=2)
        elif format == 'dxf':
            # 这里可以实现DXF格式生成
            # 简化版本：保存为JSON
            dxf_path = output_path.replace('.dxf', '.json')
            with open(dxf_path, 'w') as f:
                json.dump(cad_data, f, indent=2)
            print(f"⚠️ DXF generation not implemented yet. Saved as JSON: {dxf_path}")
        elif format == 'svg':
            # 这里可以实现SVG格式生成
            svg_path = output_path.replace('.svg', '.json')
            with open(svg_path, 'w') as f:
                json.dump(cad_data, f, indent=2)
            print(f"⚠️ SVG generation not implemented yet. Saved as JSON: {svg_path}")
        
        print(f"✅ CAD output saved to {output_path}")


# 测试函数
def test_inference():
    """测试推理功能"""
    print("Testing PVCAI Inference...")
    
    # 创建推理实例
    inference = PVCAIInference()
    
    # 创建虚拟手稿图片用于测试
    from PIL import Image
    import numpy as np
    
    # 创建虚拟图片
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    dummy_image_path = "dummy_sketch.png"
    dummy_image.save(dummy_image_path)
    
    # 测试推理
    try:
        result = inference.generate_cad(dummy_image_path, "Photovoltaic system for residential roof")
        print("✓ Inference successful")
        print(f"  Geometry length: {len(result['geometry'])}")
        print(f"  Components length: {len(result['components'])}")
        print(f"  Layout length: {len(result['layout'])}")
        
        # 保存结果
        inference.save_cad_output(result, "test_output.json")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
    
    # 清理虚拟文件
    if os.path.exists(dummy_image_path):
        os.remove(dummy_image_path)
    
    print("✅ Inference test completed!")


if __name__ == "__main__":
    test_inference()