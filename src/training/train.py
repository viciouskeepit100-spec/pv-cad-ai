"""
光伏CAD训练脚本
支持完整的训练pipeline，包括数据加载、模型训练、验证和保存
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, BertTokenizer
from src.model.architecture import create_model
from src.data.dataset import PVCAIDataset
import os
import json
import argparse
from typing import Dict, Any, Optional

class PVCAITrainer:
    """
    Photovoltaic CAD AI Trainer
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = create_model(config)
        self.model.to(self.device)
        
        # 创建优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 2e-5),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # 创建学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 100)
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 初始化处理器
        self.image_processor = ViTImageProcessor.from_pretrained(
            config.get('vit_model', 'google/vit-base-patch16-224')
        )
        self.tokenizer = BertTokenizer.from_pretrained(
            config.get('bert_model', 'bert-base-uncased')
        )
        
        print(f"✅ PVCAI Trainer initialized on {self.device}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # 准备数据
            images = batch['image'].to(self.device)
            input_ids = batch['text_input_ids'].to(self.device)
            attention_mask = batch['text_attention_mask'].to(self.device)
            targets = {
                'geometry': batch['cad_output']['geometry'].to(self.device),
                'components': batch['cad_output']['components'].to(self.device),
                'layout': batch['cad_output']['layout'].to(self.device)
            }
            
            # 前向传播
            text_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            outputs = self.model(images, text_inputs)
            
            # 计算损失
            loss_geometry = self.criterion(outputs['geometry'], targets['geometry'])
            loss_components = self.criterion(outputs['components'], targets['components'])
            loss_layout = self.criterion(outputs['layout'], targets['layout'])
            
            total_loss_batch = loss_geometry + loss_components + loss_layout
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {total_loss_batch.item():.4f}")
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                input_ids = batch['text_input_ids'].to(self.device)
                attention_mask = batch['text_attention_mask'].to(self.device)
                targets = {
                    'geometry': batch['cad_output']['geometry'].to(self.device),
                    'components': batch['cad_output']['components'].to(self.device),
                    'layout': batch['cad_output']['layout'].to(self.device)
                }
                
                text_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
                outputs = self.model(images, text_inputs)
                
                loss_geometry = self.criterion(outputs['geometry'], targets['geometry'])
                loss_components = self.criterion(outputs['components'], targets['components'])
                loss_layout = self.criterion(outputs['layout'], targets['layout'])
                
                total_loss_batch = loss_geometry + loss_components + loss_layout
                total_loss += total_loss_batch.item()
        
        return total_loss / len(val_loader)
    
    def save_checkpoint(self, epoch: int, loss: float, path: str):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"✅ Checkpoint saved to {path}")
    
    def train(self, train_dataset, val_dataset=None):
        """完整训练流程"""
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=True,
            num_workers=self.config.get('num_workers', 0)
        )
        
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.get('batch_size', 8),
                shuffle=False,
                num_workers=self.config.get('num_workers', 0)
            )
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.get('epochs', 100)):
            print(f"\nEpoch {epoch+1}/{self.config.get('epochs', 100)}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            print(f"Train Loss: {train_loss:.4f}")
            
            # 验证
            if val_dataset:
                val_loss = self.validate(val_loader)
                print(f"Val Loss: {val_loss:.4f}")
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(
                        epoch, 
                        val_loss, 
                        os.path.join(self.config.get('output_dir', '.'), 'best_model.pth')
                    )
            
            # 保存最新模型
            self.save_checkpoint(
                epoch, 
                train_loss, 
                os.path.join(self.config.get('output_dir', '.'), 'latest_model.pth')
            )
            
            # 更新学习率
            self.scheduler.step()
        
        print("✅ Training completed!")


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='Train PV CAD AI Model')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='models', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 配置
    config = {
        'vit_model': 'google/vit-base-patch16-224',
        'bert_model': 'bert-base-uncased',
        'fusion_hidden_size': 768,
        'dropout': 0.1,
        'output_format': 'dxf',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': 0.01,
        'output_dir': args.output_dir,
        'num_workers': 0 if args.no_cuda else 4
    }
    
    # 创建数据集
    train_dataset = PVCAIDataset(args.data_dir)
    
    # 创建训练器
    trainer = PVCAITrainer(config)
    
    # 开始训练
    trainer.train(train_dataset)


if __name__ == "__main__":
    main()