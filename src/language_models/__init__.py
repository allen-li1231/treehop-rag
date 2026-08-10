from .aoai import AOAI
from .base import LanguageModel
from .deepseek import DeepSeek
from .llama import LlamaServer
from .utils import (ask_model, ask_model_in_parallel,
                    model_generate, batch_inference)

MODEL_DICT = {
    # generative models
    "gpt35": "gpt-35-turbo-1106",
    "gpt4": "gpt-4-0125-preview",
    "llama3-70B": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama3-8B": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14B": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-72B": "Qwen/Qwen2.5-72B-Instruct",
    "qwen3.5-27B": "Qwen/Qwen3.5-27B",
    "qwen3.6-27B": "Qwen/Qwen3.6-27B",
    "deepseek-V2": "deepseek-ai/DeepSeek-V2-Chat-0628",
    # embedding models
    "bge-m3": "BAAI/bge-m3",
    "qwen3-4b": "Qwen/Qwen3-Embedding-4B",
}
