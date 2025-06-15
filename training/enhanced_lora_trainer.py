#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : enhanced_lora_trainer.py
# @Time    : 2025/6/13 18:53
# @Desc    : LoRA微调


import os
import torch
from datasets import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class EnhancedLoRATrainer:
    """LoRA微调训练器"""

    def __init__(self, model, tokenizer, config: ModelConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """对数据集进行tokenize"""

        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
                return_overflowing_tokens=False
            )

            tokenized["input_length"] = [len(ids) for ids in tokenized["input_ids"]]
            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in dataset.column_names if col not in ["task"]],
            desc="Tokenizing dataset"
        )

        min_length = 50
        max_length = self.config.max_length

        original_size = len(tokenized_dataset)
        tokenized_dataset = tokenized_dataset.filter(
            lambda x: min_length <= x["input_length"] <= max_length
        )
        filtered_size = len(tokenized_dataset)

        self.logger.info(f"长度过滤: {original_size} -> {filtered_size} 样本")

        return tokenized_dataset

    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset], output_dir: str) -> str:
        """执行LoRA训练"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            train_dataset = self.tokenize_dataset(train_dataset)
            if eval_dataset is not None:
                eval_dataset = self.tokenize_dataset(eval_dataset)

            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                logging_steps=50,
                save_steps=self.config.save_steps,
                eval_steps=self.config.save_steps if eval_dataset else None,
                evaluation_strategy="steps" if eval_dataset else "no",
                save_strategy="steps",
                load_best_model_at_end=True if eval_dataset else False,
                metric_for_best_model="eval_loss" if eval_dataset else None,
                greater_is_better=False,
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                fp16=True,
                report_to="none"
            )

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer
            )

            self.logger.info("开始LoRA训练...")
            trainer.train()

            final_model_path = output_dir / "final_model"
            trainer.save_model(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)

            lora_path = output_dir / "lora_weights"
            self.model.save_pretrained(lora_path)

            self.logger.info(f"训练完成，模型保存至: {final_model_path}")
            self.logger.info(f"LoRA权重保存至: {lora_path}")

            return str(final_model_path)

        except Exception as e:
            self.logger.error(f"LoRA训练失败: {e}")
            raise

    def evaluate_model(self, test_dataset: Dataset) -> Dict:
        """评估模型性能"""
        try:
            test_dataset = self.tokenize_dataset(test_dataset)

            self.model.eval()
            total_loss = 0
            num_batches = 0

            with torch.no_grad():
                for i in range(0, len(test_dataset), self.config.batch_size):
                    batch = test_dataset[i:i + self.config.batch_size]

                    input_ids = torch.tensor([item['input_ids'] for item in batch])
                    attention_mask = torch.tensor([item['attention_mask'] for item in batch])

                    if torch.cuda.is_available():
                        input_ids = input_ids.cuda()
                        attention_mask = attention_mask.cuda()

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )

                    total_loss += outputs.loss.item()
                    num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            perplexity = torch.exp(torch.tensor(avg_loss)).item()

            metrics = {
                "eval_loss": avg_loss,
                "eval_perplexity": perplexity,
                "num_samples": len(test_dataset)
            }

            self.logger.info(f"评估完成: Loss={avg_loss:.4f}, Perplexity={perplexity:.2f}")

            return metrics

        except Exception as e:
            self.logger.error(f"模型评估失败: {e}")
            return {}
