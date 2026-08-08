"""The checkpoint loads strictly into the rebuilt architecture."""

from __future__ import annotations

import torch

from backend import config as cfg
from backend.model.metadata import ModelMetadata


def test_metadata_is_self_consistent():
    meta = ModelMetadata.load(cfg.METADATA_PATH)

    assert meta.model_name == "xlm-roberta-large"
    assert meta.max_len == 128
    assert (meta.num_intents, meta.num_sentiments, meta.num_slots) == (15, 3, 165)
    assert meta.slot_labels[0] == "O"
    meta.validate()  # raises on any inconsistency


def test_checkpoint_is_a_plain_state_dict():
    state = torch.load(cfg.CHECKPOINT_PATH, map_location="meta", weights_only=True, mmap=True)

    assert isinstance(state, dict)
    assert len(state) == 397
    assert all(torch.is_tensor(v) for v in state.values())
    # No optimizer / scheduler / epoch bookkeeping, and no LoRA or adapters.
    assert not any(
        key in state for key in ("model_state_dict", "optimizer", "scheduler", "epoch")
    )
    assert not any("lora" in k.lower() or "adapter" in k.lower() for k in state)


def test_head_shapes_match_the_label_counts():
    state = torch.load(cfg.CHECKPOINT_PATH, map_location="meta", weights_only=True, mmap=True)
    meta = ModelMetadata.load(cfg.METADATA_PATH)

    assert tuple(state["intent_head.weight"].shape) == (meta.num_intents, 1024)
    assert tuple(state["sentiment_head.weight"].shape) == (meta.num_sentiments, 1024)
    assert tuple(state["slot_head.weight"].shape) == (meta.num_slots, 1024)
    assert tuple(state["encoder.embeddings.word_embeddings.weight"].shape) == (250002, 1024)


def test_model_loaded_strictly(nlu):
    """`_build_and_load` uses strict=True; reaching here proves it succeeded."""
    info = nlu.info()

    assert info["parameters"] == 560_078_007
    assert info["num_intents"] == 15
    assert info["num_slots"] == 165
    assert nlu.model.training is False  # eval mode


def test_encoder_is_xlm_roberta_large(nlu):
    config = nlu.model.encoder.config

    assert config.hidden_size == 1024
    assert config.num_hidden_layers == 24
    assert config.vocab_size == 250002


def test_sentence_heads_read_cls_not_pooler(nlu):
    """Intent/sentiment must come from last_hidden_state[:, 0, :].

    Perturbing only the pooler must leave predictions untouched; if a head had
    been wired to pooler_output instead, this would change the logits.
    """
    encoding = nlu.tokenizer(
        ["mera", "order", "nahi", "aaya"],
        is_split_into_words=True, max_length=128,
        padding="max_length", truncation=True, return_tensors="pt",
    )
    ids = encoding["input_ids"].to(nlu.device)
    mask = encoding["attention_mask"].to(nlu.device)

    with torch.no_grad():
        before = nlu.model(ids, mask)[0].clone()
        original = nlu.model.encoder.pooler.dense.weight.clone()
        nlu.model.encoder.pooler.dense.weight.add_(100.0)
        try:
            after = nlu.model(ids, mask)[0]
            assert torch.equal(before, after)
        finally:
            nlu.model.encoder.pooler.dense.weight.copy_(original)


def test_fast_tokenizer_with_word_ids(nlu):
    assert nlu.tokenizer.is_fast
    encoding = nlu.tokenizer(
        ["mera", "order"], is_split_into_words=True, return_tensors="pt",
    )
    assert encoding.word_ids(batch_index=0) is not None
