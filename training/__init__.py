"""
训练模块

提供模型训练、微调和管理功能：
- LoRA训练器 (EnhancedLoRATrainer)
- 训练管理器 (FoundationLayerManager)
- 训练流程控制
"""

from .enhanced_lora_trainer import EnhancedLoRATrainer
from .foundation_layer_manager import FoundationLayerManager

__all__ = [
    "EnhancedLoRATrainer",
    "FoundationLayerManager"
]

TRAINING_STRATEGIES = [
    "standard",          # 标准训练
    "curriculum",        # 课程学习
    "multi_task",        # 多任务训练
    "balanced"           # 平衡训练
]

DEFAULT_TRAINING_CONFIG = {
    "learning_rate": 5e-5,
    "batch_size": 6,
    "num_epochs": 5,
    "warmup_steps": 100,
    "save_steps": 100,
    "gradient_accumulation_steps": 1,
    "fp16": True,
    "dataloader_pin_memory": False
}

def create_trainer(model, tokenizer, config):
    """创建训练器的便捷函数"""
    return EnhancedLoRATrainer(model, tokenizer, config)

def create_foundation_manager(config_path="config/models.yaml"):
    """创建训练管理器的便捷函数"""
    return FoundationLayerManager(config_path)

def quick_train(model_name, data_config=None, output_suffix=""):
    """快速训练的便捷函数"""
    manager = FoundationLayerManager()
    return manager.train_lora(model_name, data_config, output_suffix)