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
from .phi3_medium import Phi3MediumInterface, call_phi3_medium

__all__ = [
    'ModelInterface',
    'TransformersModelInterface',
    'APIModelInterface',
    'sentiment_to_score',
    'GPT4oMiniInterface',
    'call_gpt4o_mini',
    'ClaudeHaikuInterface',
    'call_claude_haiku',
    'Phi3MediumInterface',
    'call_phi3_medium'
]