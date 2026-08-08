"""Label mappings produced by training, loaded from model_metadata2.json.

The label ordering is load-bearing: it is the order the classifier heads were
trained against (alphabetical, from sklearn's LabelEncoder for intent and
sentiment; "O" pinned to index 0 followed by sorted BIO tags for slots).
Reordering these silently scrambles every prediction, so the loader validates
the file instead of trusting it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ModelMetadata:
    intent_labels: List[str]
    sentiment_labels: List[str]
    slot_labels: List[str]
    slot2id: Dict[str, int]
    id2slot: Dict[int, str] = field(repr=False)
    model_name: str = "xlm-roberta-large"
    max_len: int = 128

    @property
    def num_intents(self) -> int:
        return len(self.intent_labels)

    @property
    def num_sentiments(self) -> int:
        return len(self.sentiment_labels)

    @property
    def num_slots(self) -> int:
        return len(self.slot_labels)

    @classmethod
    def load(cls, path: Path | str) -> "ModelMetadata":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Model metadata not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        missing = {
            "intent_labels", "sentiment_labels", "slot_labels",
            "slot2id", "id2slot", "model_name", "max_len",
        } - raw.keys()
        if missing:
            raise ValueError(f"{path} is missing required keys: {sorted(missing)}")

        meta = cls(
            intent_labels=list(raw["intent_labels"]),
            sentiment_labels=list(raw["sentiment_labels"]),
            slot_labels=list(raw["slot_labels"]),
            slot2id=dict(raw["slot2id"]),
            # JSON forces string keys; training wrote {str(id): tag}.
            id2slot={int(k): v for k, v in raw["id2slot"].items()},
            model_name=raw["model_name"],
            max_len=int(raw["max_len"]),
        )
        meta.validate()
        return meta

    def validate(self) -> None:
        """Fail loudly on an inconsistent metadata file rather than mispredicting."""
        if self.slot_labels[0] != "O":
            raise ValueError("slot_labels[0] must be 'O' - index 0 is the outside tag")

        for index, tag in enumerate(self.slot_labels):
            if self.slot2id.get(tag) != index:
                raise ValueError(f"slot2id['{tag}'] != {index}; metadata is inconsistent")
            if self.id2slot.get(index) != tag:
                raise ValueError(f"id2slot[{index}] != '{tag}'; metadata is inconsistent")

        if len(self.id2slot) != len(self.slot_labels):
            raise ValueError("id2slot and slot_labels disagree on the number of slot classes")

        for name, labels in (
            ("intent_labels", self.intent_labels),
            ("sentiment_labels", self.sentiment_labels),
            ("slot_labels", self.slot_labels),
        ):
            if len(set(labels)) != len(labels):
                raise ValueError(f"{name} contains duplicates")

        if self.max_len <= 0:
            raise ValueError("max_len must be positive")

    def describe(self) -> str:
        return (
            f"{self.model_name} | max_len={self.max_len} | "
            f"intents={self.num_intents} sentiments={self.num_sentiments} slots={self.num_slots}"
        )
