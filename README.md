# Individual Spatiotemporal Activity Generation using MCP-Enhanced Chain-of-Thought Large Language Models

## 概述：本项目研究使用MCP（Memory Consolidation Process）增强的链式思考大语言模型来生成个体时空活动模式的方法。

spatiotemporal-activity-generation-mcp-cot/
├── src/                          # 源代码
│   ├── models/                   # 模型实现
│   │   ├── mcp_enhanced_cot/     # MCP增强CoT模型
│   │   ├── base_models/          # 基础模型
│   │   └── fine_tuning/          # 微调相关
│   ├── data_processing/          # 数据处理
│   ├── activity_generation/      # 活动生成
│   ├── evaluation/               # 评估指标
│   ├── utils/                    # 工具函数
│   └── visualization/            # 可视化
├── data/                         # 数据目录
├── models/                       # 模型存储
├── experiments/                  # 实验配置和结果
├── configs/                      # 配置文件
├── scripts/                      # 脚本文件
├── tests/                        # 测试文件
├── notebooks/                    # Jupyter笔记本
├── docs/                         # 文档
└── deployment/                   # 部署相关

## 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/yourusername/spatiotemporal-activity-generation-mcp-cot.git
cd spatiotemporal-activity-generation-mcp-cot

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```



## 运行示例

```bash
# 数据预处理
python scripts/data_processing/preprocess_data.py

# 模型训练
python scripts/training/train_mcp_cot.py

# 活动生成
python scripts/evaluation/evaluate_model.py
```



## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。



## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@article{zhang2024study,
  title={A Study on Individual Spatiotemporal Activity Generation Method Using MCP-Enhanced Chain-of-Thought Large Language Models},
  author={Zhang, Yu and Hu, Yang and Wang, De},
  journal={Preprint},
  year={2024},
  abstract={Human spatiotemporal behavior simulation is crucial for research topics in urban planning and related fields, yet traditional rule-based and statistical approaches suffer from high costs, limited generalizability, and poor scalability. While large language models (LLMs) show promise as "world simulators," they face critical challenges in spatiotemporal reasoning including limited spatial cognition, lack of physical constraint understanding, and group homogenization tendencies. This paper introduces a novel framework that integrates chain-of-thought (CoT) reasoning with Model Context Protocol (MCP) to enable LLMs to generate realistic individual spatiotemporal activity chains...},
  keywords={Large Language Models, Chain-of-Thought Reasoning, Model Context Protocol, Spatiotemporal Behavior, Urban Computing, Human Mobility},
  institution={Tongji University},
  address={Shanghai, China}
}
```



## 联系方式

