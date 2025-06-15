import logging
from pathlib import Path
from typing import Dict, Optional
from models.config import ConfigManager, ModelConfig
from models.loader import ModelLoader
from data.data_processor import DataProcessor
from training.enhanced_lora_trainer import EnhancedLoRATrainer


class FoundationLayerManager:
    """基础模型层管理器"""

    def __init__(self, config_path: str = "config/models.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.model_loader = ModelLoader()
        self.data_processor = DataProcessor()
        self.logger = logging.getLogger(self.__class__.__name__)

    def setup_model(self, model_name: str):
        """设置模型"""
        try:
            self.logger.info(f"开始设置模型: {model_name}")

            config = self.config_manager.get_model_config(model_name)
            self.logger.info(f"模型配置: {config.model_path}")
            self.logger.info(f"模型类型: {config.model_type}")
            self.logger.info(f"LoRA配置: r={config.lora_r}, alpha={config.lora_alpha}")

            self.logger.info("正在加载模型和分词器...")
            model, tokenizer = self.model_loader.load_model_and_tokenizer(config)

            self.logger.info(f"模型设置完成: {model_name}")
            return model, tokenizer, config

        except Exception as e:
            self.logger.error(f"设置模型失败: {e}")
            raise

    def train_lora(self, model_name: str, data_config: Optional[Dict[str, str]] = None, output_suffix: str = "") -> str:
        """执行LoRA训练"""
        try:
            # 1. 设置模型
            self.logger.info("=== 第1步: 模型设置 ===")
            model, tokenizer, config = self.setup_model(model_name)

            # 2. 准备数据配置
            self.logger.info("=== 第2步: 数据准备 ===")
            if data_config is None:
                data_config = {
                    "spatiotemporal_knowledge": self.config_manager.get_data_path("spatiotemporal_knowledge"),
                    "local_behavior_data": self.config_manager.get_data_path("local_behavior_data"),
                    "cot_demonstrations": self.config_manager.get_data_path("cot_demonstrations"),
                    "mcp_training_data": self.config_manager.get_data_path("mcp_training_data")
                }

            self.logger.info("数据文件配置:")
            for task, path in data_config.items():
                self.logger.info(f"  {task}: {path}")

            # 3. 加载和处理数据
            self.logger.info("=== 第3步: 数据加载与处理 ===")
            dataset = self.data_processor.load_multi_task_data(data_config)
            balanced_dataset = self.data_processor.create_balanced_dataset(dataset)
            train_dataset, eval_dataset = self.data_processor.split_dataset(balanced_dataset)

            # 4. 创建训练器并开始训练
            self.logger.info("=== 第4步: 模型训练 ===")
            trainer = EnhancedLoRATrainer(model, tokenizer, config)

            output_dir = f"{self.config_manager.get_output_path('lora_adapters')}/{model_name}{output_suffix}"
            self.logger.info(f"输出目录: {output_dir}")

            result_path = trainer.train(train_dataset, eval_dataset, output_dir)

            self.logger.info(f"LoRA训练完成: {result_path}")
            return result_path

        except Exception as e:
            self.logger.error(f"LoRA训练失败: {e}")
            raise