"""The joint NLU architecture, moved out of the training notebook unchanged.

Shared XLM-RoBERTa Large encoder with three linear heads:

    XLMRobertaModel("xlm-roberta-large")            hidden = 1024
            |
            +-- last_hidden_state[:, 0, :] -> Dropout(0.3) -> Linear(1024 ->  15)  intent
            +-- last_hidden_state[:, 0, :] -> Dropout(0.3) -> Linear(1024 ->   3)  sentiment
            +-- last_hidden_state          -> Dropout(0.3) -> Linear(1024 -> 165)  slots

Two details that must not drift from the notebook:

* Sentence-level heads read ``last_hidden_state[:, 0, :]`` - the raw first-token
  vector - NOT ``pooler_output``. The pooler exists in the checkpoint (it came
  along with the pretrained encoder and was fine-tuned) but never feeds a head.
* Slot filling is plain per-token softmax. There is no CRF.

Submodule attribute names (``encoder``, ``dropout``, ``intent_head``,
``sentiment_head``, ``slot_head``) are what produce the state_dict keys, so they
are fixed by the checkpoint and cannot be renamed.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PretrainedConfig


class JointHinglishModel(nn.Module):
    """Shared encoder + intent / sentiment / slot heads."""

    def __init__(
        self,
        encoder_config: PretrainedConfig,
        num_intents: int,
        num_sentiments: int,
        num_slots: int,
        dropout: float = 0.3,
        skip_encoder_init: bool = True,
    ):
        super().__init__()

        # The notebook called AutoModel.from_pretrained(model_name) here, which
        # downloads ~2.2 GB of pretrained weights that load_state_dict then
        # overwrites in full. Building from the config alone is equivalent and
        # costs nothing, since every encoder tensor comes from the checkpoint.
        with _no_init_weights() if skip_encoder_init else nullcontext():
            self.encoder = AutoModel.from_config(encoder_config)

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden_size, num_intents)
        self.sentiment_head = nn.Linear(hidden_size, num_sentiments)
        self.slot_head = nn.Linear(hidden_size, num_slots)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # [CLS] token for sentence-level tasks
        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])   # (B, H)
        # All tokens for slot filling
        token_output = self.dropout(outputs.last_hidden_state)          # (B, S, H)

        intent_logits = self.intent_head(cls_output)         # (B, num_intents)
        sentiment_logits = self.sentiment_head(cls_output)   # (B, num_sentiments)
        slot_logits = self.slot_head(token_output)           # (B, S, num_slots)

        return intent_logits, sentiment_logits, slot_logits

    @classmethod
    def from_encoder_config_path(
        cls,
        config_path_or_id: str,
        num_intents: int,
        num_sentiments: int,
        num_slots: int,
        dropout: float = 0.3,
        **kwargs,
    ) -> "JointHinglishModel":
        """Build the architecture from a local config dir or a Hub model id.

        Only ``config.json`` is read - no weight file is fetched.
        """
        encoder_config = AutoConfig.from_pretrained(str(config_path_or_id))
        return cls(
            encoder_config=encoder_config,
            num_intents=num_intents,
            num_sentiments=num_sentiments,
            num_slots=num_slots,
            dropout=dropout,
            **kwargs,
        )


def _no_init_weights():
    """Skip the encoder's random weight init - 560M values we immediately discard.

    Falls back to normal (slower, still correct) initialisation if the
    transformers internal moves.
    """
    try:
        from transformers.modeling_utils import no_init_weights

        return no_init_weights()
    except Exception:  # pragma: no cover - depends on transformers internals
        return nullcontext()
