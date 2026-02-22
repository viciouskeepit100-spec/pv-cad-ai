# 光伏CAD智能设计系统 v3.0

基于大模型的光伏CAD设计系统，能够从用户输入（手稿图片 + 文字描述）生成标准CAD图纸。

## 项目结构

```
pv_cad_ai_complete_v3/
├── src/
│   ├── model/              # 完整模型架构
│   ├── data/               # 数据处理模块  
│   ├── training/           # 训练脚本
│   ├── inference/          # 推理脚本
│   └── web/                # Web界面
├── requirements.txt        # 依赖列表
└── README.md               # 项目说明
```

## 核心功能

- **多模态输入**: 支持手稿图片 + 文字描述
- **标准CAD输出**: 生成DXF/SVG格式文件
- **完整训练pipeline**: 包含训练、验证、保存
- **Web界面**: 简单易用的交互页面
- **无数据模式**: 即使没有真实数据也能运行

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动Web服务
python src/web/app.py

# 训练模型
python src/training/train.py --data_dir data --output_dir models
```