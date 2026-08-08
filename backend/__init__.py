"""Backend inference service for the Hinglish joint NLU model.

Layers:
    model/    - the trained XLM-RoBERTa Large joint model (intent + sentiment + slots)
    context/  - conversation memory and context-aware interpretation
    priority/ - rule-based priority / routing layer
    pipeline  - wires the three together
    api       - FastAPI HTTP surface
"""

__version__ = "0.1.0"
