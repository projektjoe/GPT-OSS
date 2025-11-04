"""
GPT-OSS: Educational Implementation from Scratch

An educational implementation of the GPT-OSS transformer model,
demonstrating how large language models work under the hood.
"""

__version__ = "0.1.0"
__author__ = "projektjoe"
__email__ = "projektjoetom@gmail.com"

from .main import (
    GPTOSS,
    ModelConfig,
    load_config,
    generate,
    RMSNorm,
    linear,
    softmax,
    ROPE,
    swiglu,
    attention_head,
)

from .load import (
    load_state_dict,
    load_safetensors,
    Checkpoint,
)

__all__ = [
    # Main model
    "GPTOSS",
    "ModelConfig",
    "load_config",
    "generate",
    
    # Neural network operations
    "RMSNorm",
    "linear",
    "softmax",
    "ROPE",
    "swiglu",
    "attention_head",
    
    # Loading utilities
    "load_state_dict",
    "load_safetensors",
    "Checkpoint",
]

