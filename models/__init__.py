"""
模型管理模块

提供模型配置、加载和管理功能：
- 模型配置管理 (ConfigManager)
- 模型加载器 (ModelLoader)  
- 模型配置类 (ModelConfig)
"""

from .config import ConfigManager, ModelConfig
from .loader import ModelLoader

__all__ = [
    "ConfigManager",
    "ModelConfig",
    "ModelLoader"
]

SUPPORTED_MODEL_TYPES = [
    "glm",
    "deepseek", 
    "llama",
    "bloom"
]

DEFAULT_LORA_CONFIG = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.1,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
}

def create_model_config(model_name, model_path, model_type, **kwargs):
    """创建模型配置的便捷函数"""
    return ModelConfig(
        model_name=model_name,
        model_path=model_path,
        model_type=model_type,
        **kwargs
    )

def load_model_with_config(config_path, model_name):
    """根据配置文件加载模型的便捷函数"""
    config_manager = ConfigManager(config_path)
    model_config = config_manager.get_model_config(model_name)
    loader = ModelLoader()
    return loader.load_model_and_tokenizer(model_config)