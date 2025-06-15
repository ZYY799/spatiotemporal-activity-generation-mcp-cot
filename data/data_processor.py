#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : data_processor.py
# @Time    : 2025/6/13 18:52
# @Desc    : 预训练数据处理


import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datasets import Dataset, concatenate_datasets


class DataProcessor:
    """数据处理器"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_multi_task_data(self, data_paths: Dict[str, str]) -> Dataset:
        """加载多任务数据"""
        try:
            all_datasets = []

            for task_name, data_path in data_paths.items():
                if not Path(data_path).exists():
                    self.logger.warning(f"数据文件不存在: {data_path}")
                    continue

                self.logger.info(f"加载 {task_name} 数据: {data_path}")

                with open(data_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)

                formatted_data = []
                for item in raw_data:
                    if item.get('input'):
                        text = f"### Task: {task_name}\n### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}<|endoftext|>"
                    else:
                        text = f"### Task: {task_name}\n### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}<|endoftext|>"

                    formatted_data.append({
                        "text": text,
                        "task": task_name,
                        "length": len(text)
                    })

                dataset = Dataset.from_list(formatted_data)
                all_datasets.append(dataset)

                self.logger.info(f"{task_name} 数据集加载完成，共 {len(dataset)} 条样本")

            if not all_datasets:
                raise ValueError("没有找到有效的数据文件")

            combined_dataset = concatenate_datasets(all_datasets)

            task_counts = {}
            for item in combined_dataset:
                task = item['task']
                task_counts[task] = task_counts.get(task, 0) + 1

            self.logger.info("多任务数据集统计:")
            for task, count in task_counts.items():
                self.logger.info(f"  {task}: {count} 条样本")

            return combined_dataset

        except Exception as e:
            self.logger.error(f"加载多任务数据失败: {e}")
            raise

    def create_balanced_dataset(self, dataset: Dataset, max_samples_per_task: int = 1000) -> Dataset:
        """创建任务平衡的数据集"""
        try:
            task_datasets = {}
            for item in dataset:
                task = item['task']
                if task not in task_datasets:
                    task_datasets[task] = []
                task_datasets[task].append(item)

            balanced_data = []
            for task, items in task_datasets.items():
                if len(items) > max_samples_per_task:
                    import random
                    random.seed(42)
                    items = random.sample(items, max_samples_per_task)

                balanced_data.extend(items)
                self.logger.info(f"任务 {task}: 使用 {len(items)} 条样本")

            return Dataset.from_list(balanced_data)

        except Exception as e:
            self.logger.error(f"创建平衡数据集失败: {e}")
            raise

    def split_dataset(self, dataset: Dataset, train_ratio: float = 0.8) -> tuple[Dataset, Dataset]:
        """分割数据集为训练集和验证集"""
        try:
            total_size = len(dataset)
            train_size = int(total_size * train_ratio)

            dataset = dataset.shuffle(seed=42)
            train_dataset = dataset.select(range(train_size))
            eval_dataset = dataset.select(range(train_size, total_size))

            self.logger.info(f"数据集分割完成: 训练集 {len(train_dataset)} 条, 验证集 {len(eval_dataset)} 条")

            return train_dataset, eval_dataset

        except Exception as e:
            self.logger.error(f"分割数据集失败: {e}")
            raise
