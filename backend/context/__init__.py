"""Conversation memory and context-aware interpretation.

Neither module retrains or fine-tunes anything. They sit *around* the frozen
NLU model and decide when a short follow-up should be read against earlier turns.
"""

from backend.context.context_interpreter import ContextInterpreter, InterpreterConfig
from backend.context.context_manager import ConversationContext, Turn

__all__ = ["ConversationContext", "Turn", "ContextInterpreter", "InterpreterConfig"]
