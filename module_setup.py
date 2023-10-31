# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/16 16:03

from deep_training.utils.hf import register_transformer_model,register_transformer_config,register_transformer_tokenizer
from transformers import AutoModelForCausalLM
from aigc_zoo.model_zoo.skywork.llm_model import MySkyworkForCausalLM,SkyworkConfig,SkyworkTokenizer
__all__ = [
    "module_setup"
]

def module_setup():
    # 导入模型
    register_transformer_config(SkyworkConfig)
    register_transformer_model(MySkyworkForCausalLM, AutoModelForCausalLM)
    register_transformer_tokenizer(SkyworkConfig,SkyworkTokenizer,SkyworkTokenizer)