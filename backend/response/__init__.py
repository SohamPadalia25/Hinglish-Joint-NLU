"""Template-based response generation over the pipeline's structured output."""

from backend.response.response_generator import (
    RESPONSE_TYPES,
    IntentStrategy,
    ResponseGenerator,
)

__all__ = ["ResponseGenerator", "IntentStrategy", "RESPONSE_TYPES"]
