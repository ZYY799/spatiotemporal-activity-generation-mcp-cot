#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : loader.py
# @Time    : 2025/6/13 18:50
# @Desc    : 模型加载器



import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import logging
from typing import Tuple, Optional


class ModelLoader:
    """模型加载器"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_model_and_tokenizer(self, config: ModelConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """加载模型和分词器"""
        try:
            tokenizer = self._load_tokenizer(config)
            model = self._load_model(config)

            if config.use_lora:
                model = self._apply_lora(model, config)

            self.logger.info(f"模型和分词器加载完成: {config.model_name}")
            return model, tokenizer

        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            raise

    def _load_tokenizer(self, config: ModelConfig) -> AutoTokenizer:
        """加载分词器"""
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=config.trust_remote_code,
            use_fast=True
        )

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        return tokenizer

    def _load_model(self, config: ModelConfig) -> AutoModelForCausalLM:
        """加载模型"""
        quantization_config = None
        if config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        elif config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        torch_dtype = torch.float16 if config.torch_dtype == "float16" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch_dtype,
            device_map=config.device if config.device != "auto" else "auto",
            trust_remote_code=config.trust_remote_code,
            quantization_config=quantization_config
        )

        return model

    def _apply_lora(self, model: AutoModelForCausalLM, config: ModelConfig) -> AutoModelForCausalLM:
        """应用LoRA"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none"
        )

        model = get_peft_model(model, lora_config)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        self.logger.info(f"LoRA配置完成")
        self.logger.info(f"可训练参数: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

        return model