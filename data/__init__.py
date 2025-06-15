
"""
数据处理模块

提供多任务数据加载、处理和管理功能：
- 数据处理器 (DataProcessor)
- 多任务数据集管理
- 数据平衡和分割
"""

from .data_processor import DataProcessor

__all__ = [
    "DataProcessor"
]

SUPPORTED_DATA_FORMATS = [
    "alpaca",  # Alpaca格式：instruction, input, output
    "json",    # 标准JSON格式
]

SUPPORTED_TASK_TYPES = [
    "spatiotemporal_knowledge",  # 时空行为专业知识
    "local_behavior_data",       # 本地化行为数据
    "cot_demonstrations",        # CoT推理示例
    "mcp_training_data"          # MCP工具使用数据
]

DEFAULT_DATA_PARAMS = {
    "max_samples_per_task": 1000,
    "train_ratio": 0.8,
    "min_length": 50,
    "max_length": 9216,
    "shuffle_seed": 42
}

def create_data_processor():
    """创建数据处理器的便捷函数"""
    return DataProcessor()

def load_multi_task_data(data_config):
    """加载多任务数据的便捷函数"""
    processor = DataProcessor()
    return processor.load_multi_task_data(data_config)