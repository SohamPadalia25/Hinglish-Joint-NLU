"""HinglishNLU - loads the fine-tuned checkpoint once and serves predictions.

The model is ~2.2 GB on disk and ~2.2 GB in RAM, so a process holds exactly one
instance and reuses it for every request.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from backend import config as cfg
from backend.model.joint_model import JointHinglishModel
from backend.model.metadata import ModelMetadata
from backend.model.preprocessing import align_to_original, normalize_hinglish

logger = logging.getLogger(__name__)

UNKNOWN_LABEL = "UNKNOWN"


@dataclass
class _Span:
    """A slot span under construction while walking the BIO tag sequence."""

    slot_type: str
    word_indices: List[int]
    confidences: List[float]


class HinglishNLU:
    """Intent + sentiment + slot prediction from the trained joint model.

    Example
    -------
    >>> nlu = HinglishNLU()
    >>> nlu.predict("Payment 799 rupaye cut ho gayi")["intent"]
    'PAYMENT_ISSUE'
    """

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        metadata_path: Path | str | None = None,
        tokenizer_dir: Path | str | None = None,
        device: str | None = None,
        pad_to_max_length: bool | None = None,
        dropout: float = 0.3,
    ):
        self.checkpoint_path = Path(checkpoint_path or cfg.CHECKPOINT_PATH)
        self.metadata_path = Path(metadata_path or cfg.METADATA_PATH)
        self.tokenizer_dir = Path(tokenizer_dir or cfg.TOKENIZER_DIR)
        self.pad_to_max_length = (
            cfg.PAD_TO_MAX_LENGTH if pad_to_max_length is None else pad_to_max_length
        )

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        resolved_device = device or cfg.DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)

        # torch modules are not safe under concurrent forward passes on the same
        # instance; uvicorn runs sync endpoints in a threadpool, so serialise.
        self._lock = threading.Lock()

        started = time.perf_counter()

        self.metadata = ModelMetadata.load(self.metadata_path)
        logger.info("Metadata loaded: %s", self.metadata.describe())

        self.tokenizer = self._load_tokenizer()
        if not self.tokenizer.is_fast:
            raise RuntimeError(
                "A fast tokenizer is required: slot alignment depends on "
                "encoding.word_ids(), which the slow tokenizer does not provide."
            )

        self.model = self._build_and_load()
        self.load_seconds = time.perf_counter() - started
        logger.info(
            "Model ready on %s in %.1fs (%s)",
            self.device, self.load_seconds, self.checkpoint_path.name,
        )

    # ------------------------------------------------------------------ setup

    def _load_tokenizer(self):
        """Prefer the vendored tokenizer; fall back to the Hub on first run."""
        if (self.tokenizer_dir / "tokenizer.json").is_file():
            logger.info("Loading tokenizer from %s", self.tokenizer_dir)
            return AutoTokenizer.from_pretrained(str(self.tokenizer_dir))

        logger.warning(
            "No tokenizer at %s - downloading '%s' from the Hugging Face Hub "
            "(tokenizer + config only, no model weights).",
            self.tokenizer_dir, cfg.HF_MODEL_ID,
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.HF_MODEL_ID)
        try:
            self.tokenizer_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(str(self.tokenizer_dir))
            logger.info("Tokenizer cached to %s for offline reuse", self.tokenizer_dir)
        except OSError as exc:
            logger.warning("Could not cache tokenizer locally: %s", exc)
        return tokenizer

    def _encoder_config_source(self) -> str:
        """Local config.json if vendored, else the Hub id (config only)."""
        if (self.tokenizer_dir / "config.json").is_file():
            return str(self.tokenizer_dir)
        return cfg.HF_MODEL_ID

    def _build_and_load(self) -> JointHinglishModel:
        model = JointHinglishModel.from_encoder_config_path(
            self._encoder_config_source(),
            num_intents=self.metadata.num_intents,
            num_sentiments=self.metadata.num_sentiments,
            num_slots=self.metadata.num_slots,
            dropout=0.3,
        )

        state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(state_dict, dict):
            raise TypeError(
                f"{self.checkpoint_path} did not deserialise to a state_dict "
                f"(got {type(state_dict).__name__})"
            )
        if "model_state_dict" in state_dict:  # tolerate a wrapped checkpoint
            state_dict = state_dict["model_state_dict"]

        # strict=True is the point: it proves the architecture rebuilt here is
        # exactly the one that was trained. Any silent rename would raise.
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:  # pragma: no cover - strict=True raises first
            raise RuntimeError(f"State dict mismatch: missing={missing} unexpected={unexpected}")

        model.to(self.device)
        model.eval()
        return model

    # -------------------------------------------------------------- inference

    def predict(self, text: str) -> Dict[str, Any]:
        """Run the joint model over one sentence.

        Returns intent, sentiment, their softmax confidences, and slot spans as
        a list (a list, not a dict, so repeated entity types survive).
        """
        normalized = normalize_hinglish(text)
        tokens = normalized.split()

        if not tokens:
            return {
                "intent": UNKNOWN_LABEL,
                "intent_confidence": 0.0,
                "sentiment": UNKNOWN_LABEL,
                "sentiment_confidence": 0.0,
                "slots": [],
            }

        surface_tokens, aligned = align_to_original(text, tokens)

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.metadata.max_len,
            padding="max_length" if self.pad_to_max_length else True,
            truncation=True,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with self._lock, torch.no_grad():
            intent_logits, sentiment_logits, slot_logits = self.model(input_ids, attention_mask)

        intent_probs = F.softmax(intent_logits[0], dim=-1).cpu()
        sentiment_probs = F.softmax(sentiment_logits[0], dim=-1).cpu()
        slot_probs = F.softmax(slot_logits[0], dim=-1).cpu()

        intent_idx = int(intent_probs.argmax())
        sentiment_idx = int(sentiment_probs.argmax())

        slot_conf, slot_ids = slot_probs.max(dim=-1)
        spans = self._decode_slot_spans(
            word_ids=word_ids,
            slot_ids=slot_ids.tolist(),
            slot_conf=slot_conf.tolist(),
            surface_tokens=surface_tokens,
            normalized_tokens=tokens,
            aligned=aligned,
        )

        return {
            "intent": self.metadata.intent_labels[intent_idx],
            "intent_confidence": round(float(intent_probs[intent_idx]), 4),
            "sentiment": self.metadata.sentiment_labels[sentiment_idx],
            "sentiment_confidence": round(float(sentiment_probs[sentiment_idx]), 4),
            "slots": spans,
        }

    def predict_verbose(self, text: str, top_k: int = 3) -> Dict[str, Any]:
        """`predict` plus the top-k intent and full sentiment distributions."""
        result = self.predict(text)
        if result["intent"] == UNKNOWN_LABEL:
            return {**result, "intent_top_k": [], "sentiment_distribution": {}}

        normalized = normalize_hinglish(text)
        encoding = self.tokenizer(
            normalized.split(),
            is_split_into_words=True,
            max_length=self.metadata.max_len,
            padding="max_length" if self.pad_to_max_length else True,
            truncation=True,
            return_tensors="pt",
        )
        with self._lock, torch.no_grad():
            intent_logits, sentiment_logits, _ = self.model(
                encoding["input_ids"].to(self.device),
                encoding["attention_mask"].to(self.device),
            )

        intent_probs = F.softmax(intent_logits[0], dim=-1).cpu()
        sentiment_probs = F.softmax(sentiment_logits[0], dim=-1).cpu()

        k = min(top_k, self.metadata.num_intents)
        values, indices = intent_probs.topk(k)
        result["intent_top_k"] = [
            {"intent": self.metadata.intent_labels[int(i)], "confidence": round(float(v), 4)}
            for v, i in zip(values, indices)
        ]
        result["sentiment_distribution"] = {
            label: round(float(sentiment_probs[i]), 4)
            for i, label in enumerate(self.metadata.sentiment_labels)
        }
        return result

    # ---------------------------------------------------------- slot decoding

    def _decode_slot_spans(
        self,
        word_ids: Sequence[Optional[int]],
        slot_ids: Sequence[int],
        slot_conf: Sequence[float],
        surface_tokens: Sequence[str],
        normalized_tokens: Sequence[str],
        aligned: bool,
    ) -> List[Dict[str, Any]]:
        """Walk the BIO tags and emit entity spans, keeping duplicates.

        Only the first sub-token of each word carries a label - that is how
        training aligned them (continuations were set to -100 and ignored by the
        loss), so reading any other position would read an untrained output.

        A stray ``I-X`` with no matching open span opens a new span instead of
        being dropped; this is standard lenient BIO recovery for a model without
        a CRF to enforce valid transitions.
        """
        spans: List[Dict[str, Any]] = []
        current: Optional[_Span] = None
        prev_word_id: Optional[int] = None

        def close() -> None:
            nonlocal current
            if current is not None:
                spans.append(self._finalize(current, surface_tokens, normalized_tokens, aligned))
                current = None

        for position, word_id in enumerate(word_ids):
            if word_id is None or word_id == prev_word_id:
                continue  # special token, or a continuation sub-token
            prev_word_id = word_id

            tag = self.metadata.id2slot[slot_ids[position]]
            confidence = float(slot_conf[position])

            if tag == "O":
                close()
            elif tag.startswith("B-"):
                close()
                current = _Span(tag[2:], [word_id], [confidence])
            elif tag.startswith("I-"):
                if current is not None and current.slot_type == tag[2:]:
                    current.word_indices.append(word_id)
                    current.confidences.append(confidence)
                else:
                    close()
                    current = _Span(tag[2:], [word_id], [confidence])
            else:  # pragma: no cover - metadata guarantees O / B- / I-
                close()

        close()
        return spans

    @staticmethod
    def _finalize(
        span: _Span,
        surface_tokens: Sequence[str],
        normalized_tokens: Sequence[str],
        aligned: bool,
    ) -> Dict[str, Any]:
        def join(source: Sequence[str]) -> str:
            return " ".join(source[i] for i in span.word_indices if i < len(source))

        normalized_value = join(normalized_tokens)
        value = join(surface_tokens) if aligned else normalized_value

        return {
            "type": span.slot_type,
            "value": value,
            "confidence": round(sum(span.confidences) / len(span.confidences), 4),
            "value_normalized": normalized_value,
        }

    # ------------------------------------------------------------------ misc

    def info(self) -> Dict[str, Any]:
        return {
            "model_name": self.metadata.model_name,
            "checkpoint": str(self.checkpoint_path),
            "device": str(self.device),
            "max_len": self.metadata.max_len,
            "num_intents": self.metadata.num_intents,
            "num_sentiments": self.metadata.num_sentiments,
            "num_slots": self.metadata.num_slots,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "load_seconds": round(self.load_seconds, 2),
        }
