#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : config.py
# @Time    : 2025/6/13 18:21
# @Desc    : 模型配置管理

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path
import yaml
import json


@dataclass
class ModelConfig:
    """模型配置类"""
    model_name: str
    model_path: str
    model_type: str  # "glm", "deepseek", "llama", "bloom"

    # 基础参数
    max_length: int = 9216
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    # 硬件配置
    device: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: str = "float16"
    trust_remote_code: bool = True

    # LoRA配置
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    # 训练配置
    learning_rate: float = 5e-5
    batch_size: int = 6
    gradient_accumulation_steps: int = 1
    num_epochs: int = 5
    warmup_steps: int = 100
    save_steps: int = 100


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: str = "config/models.yaml"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """加载配置文件"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            return self._create_default_config()

    def _create_default_config(self) -> Dict:
        """创建默认配置"""
        default_config = {
            "models": {
                "glm-4-9b": {
                    "model_path": "THUDM/glm-4-9b-chat",
                    "model_type": "glm",
                    "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
                },
                "deepseek-r1-7b": {
                    "model_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                    "model_type": "deepseek",
                    "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
                },
                "llama2-13b": {
                    "model_path": "meta-llama/Llama-2-13b-chat-hf",
                    "model_type": "llama",
                    "lora_target_modules": ["q_proj", "v_proj"]
                },
                "bloom-7b": {
                    "model_path": "bigscience/bloom-7b1",
                    "model_type": "bloom",
                    "lora_target_modules": ["query_key_value"]
                }
            },
            "data_paths": {
                "spatiotemporal_knowledge": "data/knowledge/spatiotemporal_behavior_alpaca.json",
                "local_behavior_data": "data/local/lujiazui_behavior_alpaca.json",
                "cot_demonstrations": "data/cot/reasoning_examples_alpaca.json",
                "mcp_training_data": "data/mcp/tool_usage_alpaca.json"
            },
            "output_paths": {
                "fine_tuned_models": "models/fine_tuned",
                "lora_adapters": "models/lora_adapters",
                "logs": "logs",
                "checkpoints": "checkpoints"
            }
        }

        self._save_config(default_config)
        return default_config

    def _save_config(self, config: Dict):
        """保存配置文件"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, ensure_ascii=False)

    def get_model_config(self, model_name: str) -> ModelConfig:
        """获取模型配置"""
        models = self.config.get("models", {})
        if model_name not in models:
            raise ValueError(f"Model {model_name} not found in configuration")

        model_info = models[model_name]
        return ModelConfig(
            model_name=model_name,
            model_path=model_info["model_path"],
            model_type=model_info["model_type"],
            lora_target_modules=model_info.get("lora_target_modules", ["q_proj", "v_proj"])
        )

    def get_data_path(self, data_type: str) -> str:
        """获取数据路径"""
        data_paths = self.config.get("data_paths", {})
        if data_type not in data_paths:
            raise ValueError(f"Data type {data_type} not found in configuration")
        return data_paths[data_type]

    def get_output_path(self, output_type: str) -> str:
        """获取输出路径"""
        output_paths = self.config.get("output_paths", {})
        if output_type not in output_paths:
            raise ValueError(f"Output type {output_type} not found in configuration")
        return output_paths[output_type]
