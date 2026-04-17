# Standalone Graph + RAG Environment

This repository contains a standalone YUXI graph + RAG setup that can run next to an existing local stack.

## What is included

- CPU and GPU compatible local embedding/reranker support
- Fucheers as the chat LLM provider
- A standalone Docker Compose layout with isolated host ports
- A smoke-test flow for `lightrag` knowledge-base creation, upload, indexing, and Q&A

## What is intentionally not committed

- Real `.env` secrets
- Local model weight files
- Dependency directories such as virtual environments and `node_modules`
- Runtime data under `saves/` except the reproducible defaults listed below

## Reproducible defaults that are committed

- `.env.template`
- `saves/config/base.toml`
- `saves/smoke/smoke_graph_test.md`
- `docker-compose.yml`
- `docker-compose.gpu.yml`
- `scripts/up_standalone_graph_rag.ps1`
- `scripts/up_standalone_graph_rag_gpu.ps1`
- `scripts/check_local_model_gpu.ps1`
- `scripts/smoke_test_graph_rag.ps1`

## CPU startup

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\up_standalone_graph_rag.ps1
```

## GPU startup

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\up_standalone_graph_rag_gpu.ps1
```

## GPU verification

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_local_model_gpu.ps1
```

Expected output should show:

- `device_env cuda`
- `cuda_available True`
- a CUDA-enabled `torch` build

## Smoke test

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_test_graph_rag.ps1
```

This smoke test covers:

- API health
- admin login/bootstrap
- `lightrag` knowledge-base creation
- markdown upload
- parse/index completion
- retrieval and graph generation

## Local model notes

The repository assumes local embedding and reranker model directories are mounted into the containers at runtime. Model weight files are intentionally excluded from version control.
