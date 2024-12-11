from model import ModelWithLoRA

from typing import Tuple

from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, prepare_model_for_kbit_training


def default_generation_config(**kwargs):
    return GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        use_cache=False,
        do_sample=True,
        max_length=384,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    )


def load_llama_tokenizer(base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # Fix the decapoda-research tokenizer bug
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"  # Allow batched inference
    return tokenizer


def load_model(base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", lora_r: int = 8, 
               lora_alpha: int = 16,lora_dropout: float = 0.05, 
               lora_target_modules: Tuple = ("q_proj", "k_proj", "v_proj", "o_proj"),
               resume_from_checkpoint: str = "pretrained_model/", load_in_8bit: bool = True):
    
    llama_model = LlamaForCausalLM.from_pretrained(
        base_model,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16,
        device_map="cuda:1",
    )
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none"
    )

    llama_model = prepare_model_for_kbit_training(llama_model)
    model = ModelWithLoRA(llama_model, lora_config)
    model.generation_config = default_generation_config()
    model.model.model.generation_config = model.generation_config
    model.model.model.config.pad_token_id = 0
    model.model.model.config.bos_token_id = 1
    model.model.model.config.eos_token_id = 2

    return model
