"""
Models package for LLM API calls and sentiment classification.
"""

from .model_interface import (
    ModelInterface, 
    TransformersModelInterface, 
    APIModelInterface,
    sentiment_to_score
)
from .gpt4o_mini import GPT4oMiniInterface, call_gpt4o_mini
from .claude_haiku import ClaudeHaikuInterface, call_claude_haiku
from .gpt41_batch import GPT41BatchInterface, call_gpt41
from .phi4_vllm import Phi4VllmInterface, call_phi4_vllm
from .claude_batch import ClaudeBatchInterface, call_claude

__all__ = [
    'ModelInterface',
    'TransformersModelInterface',
    'APIModelInterface',
    'sentiment_to_score',
    'GPT4oMiniInterface',
    'call_gpt4o_mini',
    'ClaudeHaikuInterface',
    'call_claude_haiku',
    'GPT41BatchInterface',
    'call_gpt41',
    'Phi4VllmInterface',
    'call_phi4_vllm',
    'ClaudeBatchInterface',
    'call_claude'
]