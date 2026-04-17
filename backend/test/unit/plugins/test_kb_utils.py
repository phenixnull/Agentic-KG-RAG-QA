from __future__ import annotations

import importlib
import sys

import pytest


def reload_yuxi_module(module_name: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("FUCHEERS_API_KEY", "test-fucheers-key")
    monkeypatch.setenv("YUXI_SKIP_APP_INIT", "1")

    for name in list(sys.modules):
        if name == "yuxi" or name.startswith("yuxi."):
            sys.modules.pop(name, None)

    return importlib.import_module(module_name)


@pytest.mark.asyncio
async def test_prepare_item_metadata_computes_hash_for_minio_file(monkeypatch: pytest.MonkeyPatch):
    kb_utils = reload_yuxi_module("yuxi.knowledge.utils.kb_utils", monkeypatch)
    minio_client = importlib.import_module("yuxi.storage.minio.client")

    file_bytes = b"smoke markdown content"
    expected_hash = await kb_utils.calculate_content_hash(file_bytes)

    class DummyClient:
        async def adownload_file(self, bucket_name: str, object_name: str) -> bytes:
            assert bucket_name == "knowledgebases"
            assert object_name == "kb_test/upload/smoke.md"
            return file_bytes

    monkeypatch.setattr(minio_client, "get_minio_client", lambda: DummyClient())

    metadata = await kb_utils.prepare_item_metadata(
        item="http://localhost:9000/knowledgebases/kb_test/upload/smoke.md",
        content_type="file",
        db_id="kb_test",
        params={},
    )

    assert metadata["content_hash"] == expected_hash
    assert metadata["path"] == "http://localhost:9000/knowledgebases/kb_test/upload/smoke.md"
    assert metadata["file_type"] == "md"
