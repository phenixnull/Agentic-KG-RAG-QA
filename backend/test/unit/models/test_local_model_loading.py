from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def reload_yuxi_module(module_name: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("FUCHEERS_API_KEY", "test-fucheers-key")

    for name in list(sys.modules):
        if name == "yuxi" or name.startswith("yuxi."):
            sys.modules.pop(name, None)

    return importlib.import_module(module_name)


def test_static_models_include_fucheers_and_local_variants(monkeypatch: pytest.MonkeyPatch):
    models = reload_yuxi_module("yuxi.config.static.models", monkeypatch)

    assert "fucheers" in models.DEFAULT_CHAT_MODEL_PROVIDERS
    assert "local/bge-m3" in models.DEFAULT_EMBED_MODELS
    assert "local/bge-reranker-v2-m3" in models.DEFAULT_RERANKERS


def test_select_embedding_model_returns_local_embedding(monkeypatch: pytest.MonkeyPatch):
    embed = reload_yuxi_module("yuxi.models.embed", monkeypatch)

    model = embed.select_embedding_model("local/bge-m3")

    assert isinstance(model, embed.LocalEmbedding)
    assert model.dimension == 1024
    assert Path(model.model_path).name == "bge-m3"
    assert Path(model.model_path).parent.name == "embedding"


def test_get_reranker_returns_local_reranker(monkeypatch: pytest.MonkeyPatch):
    rerank = reload_yuxi_module("yuxi.models.rerank", monkeypatch)

    model = rerank.get_reranker("local/bge-reranker-v2-m3")

    assert isinstance(model, rerank.LocalReranker)
    assert Path(model.model_path).name == "bge-reranker-v2-m3"
    assert Path(model.model_path).parent.name == "embedding"


def test_local_models_use_cuda_when_requested(monkeypatch: pytest.MonkeyPatch):
    embed = reload_yuxi_module("yuxi.models.embed", monkeypatch)
    rerank = reload_yuxi_module("yuxi.models.rerank", monkeypatch)

    monkeypatch.setenv("YUXI_LOCAL_MODEL_DEVICE", "cuda")
    monkeypatch.setattr(embed.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rerank.torch.cuda, "is_available", lambda: True)

    embed_model = embed.LocalEmbedding(
        model_id="local/bge-m3",
        name="bge-m3 (local)",
        dimension=1024,
        base_url="/tmp/bge-m3",
        api_key="no_api_key",
    )
    rerank_model = rerank.LocalReranker(
        model_name="BAAI/bge-reranker-v2-m3 (local)",
        api_key="no_api_key",
        base_url="/tmp/bge-reranker-v2-m3",
    )

    assert embed_model.device.type == "cuda"
    assert rerank_model.device.type == "cuda"


def test_local_embedding_cls_pooling_returns_normalized_vectors(monkeypatch: pytest.MonkeyPatch):
    embed = reload_yuxi_module("yuxi.models.embed", monkeypatch)

    class DummyTokenizer:
        def __call__(self, texts, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
            }

    class DummyModel:
        def eval(self):
            return self

        def __call__(self, **kwargs):
            last_hidden_state = torch.tensor(
                [
                    [[3.0, 4.0], [0.0, 1.0]],
                    [[5.0, 12.0], [0.0, 1.0]],
                ],
                dtype=torch.float32,
            )
            return type("Output", (), {"last_hidden_state": last_hidden_state})()

    model = embed.LocalEmbedding(
        model_id="local/bge-m3",
        name="bge-m3 (local)",
        dimension=2,
        base_url="/tmp/bge-m3",
        api_key="no_api_key",
    )
    model.tokenizer = DummyTokenizer()
    model.model = DummyModel()
    model._loaded = True

    vectors = model.encode(["alpha", "beta"])

    assert vectors[0] == pytest.approx([0.6, 0.8])
    assert vectors[1] == pytest.approx([5 / 13, 12 / 13])


@pytest.mark.asyncio
async def test_local_reranker_uses_sequence_classification_logits(monkeypatch: pytest.MonkeyPatch):
    rerank = reload_yuxi_module("yuxi.models.rerank", monkeypatch)

    class DummyTokenizer:
        def __call__(self, pairs, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long),
            }

    class DummyModel:
        def eval(self):
            return self

        def __call__(self, **kwargs):
            return type("Output", (), {"logits": torch.tensor([[1.0], [2.0]], dtype=torch.float32)})()

    model = rerank.LocalReranker(
        model_name="BAAI/bge-reranker-v2-m3 (local)",
        api_key="no_api_key",
        base_url="/tmp/bge-reranker-v2-m3",
    )
    model.tokenizer = DummyTokenizer()
    model.model = DummyModel()
    model._loaded = True

    scores = await model.acompute_score(["query", ["doc-1", "doc-2"]], normalize=False)

    assert scores == pytest.approx([1.0, 2.0])


@pytest.mark.asyncio
async def test_lightrag_uses_local_embedding_abatch_encode(monkeypatch: pytest.MonkeyPatch):
    embed = reload_yuxi_module("yuxi.models.embed", monkeypatch)
    lightrag = reload_yuxi_module("yuxi.knowledge.implementations.lightrag", monkeypatch)

    captured = {}

    class DummyLocalEmbedding:
        dimension = 1024
        batch_size = 7

        async def abatch_encode(self, texts, batch_size=None):
            captured["texts"] = list(texts)
            captured["batch_size"] = batch_size
            return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(embed, "select_embedding_model", lambda model_id: DummyLocalEmbedding())
    monkeypatch.setattr(lightrag, "select_embedding_model", lambda model_id: DummyLocalEmbedding())

    kb = lightrag.LightRagKB.__new__(lightrag.LightRagKB)
    embedding_func = kb._get_embedding_func(
        {
            "model_id": "local/bge-m3",
            "name": "bge-m3 (local)",
            "dimension": 1024,
            "base_url": "/tmp/bge-m3",
            "api_key": "no_api_key",
        }
    )

    result = await embedding_func.func(["alpha", "beta"])

    assert isinstance(result, np.ndarray)
    assert np.allclose(result, [[0.1, 0.2], [0.1, 0.2]])
    assert captured == {"texts": ["alpha", "beta"], "batch_size": 7}
