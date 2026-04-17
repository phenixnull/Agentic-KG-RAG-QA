import asyncio
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock

import httpx
import requests
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from yuxi import config
from yuxi.utils import get_docker_safe_url, hashstr, logger


_LOCAL_EMBEDDING_CACHE: dict[str, tuple[object, object]] = {}
_LOCAL_EMBEDDING_CACHE_LOCK = Lock()


def _resolve_local_model_device() -> torch.device:
    requested = (os.getenv("YUXI_LOCAL_MODEL_DEVICE") or "cpu").strip().lower()

    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested)
        logger.warning(
            "YUXI_LOCAL_MODEL_DEVICE is set to CUDA, but CUDA is unavailable in the current container. Falling back to CPU."
        )

    return torch.device("cpu")


class BaseEmbeddingModel(ABC):
    def __init__(
        self,
        model=None,
        name=None,
        dimension=None,
        url=None,
        base_url=None,
        api_key=None,
        model_id=None,
        batch_size=40,
    ):
        """
        Args:
            model: 模型名称，冗余设计，同name
            name: 模型名称，冗余设计，同model
            dimension: 维度
            url: 请求URL，冗余设计，同base_url
            base_url: 基础URL，请求URL，冗余设计，同url
            api_key: 请求API密钥
            batch_size: 模型推荐的批量向量化大小
        """
        base_url = base_url or url
        self.model = model or name
        self.dimension = dimension
        self.base_url = get_docker_safe_url(base_url)
        self.api_key = os.getenv(api_key, api_key)
        self.batch_size = int(batch_size or 40)
        self.embed_state = {}

    @abstractmethod
    def encode(self, message: list[str] | str) -> list[list[float]]:
        """同步编码"""
        raise NotImplementedError("Subclasses must implement this method")

    def encode_queries(self, queries: list[str] | str) -> list[list[float]]:
        """等同于encode"""
        return self.encode(queries)

    @abstractmethod
    async def aencode(self, message: list[str] | str) -> list[list[float]]:
        """异步编码"""
        raise NotImplementedError("Subclasses must implement this method")

    async def aencode_queries(self, queries: list[str] | str) -> list[list[float]]:
        """等同于aencode"""
        return await self.aencode(queries)

    def batch_encode(self, messages: list[str], batch_size: int | None = None) -> list[list[float]]:
        # logger.info(f"Batch encoding {len(messages)} messages")
        batch_size = batch_size or self.batch_size
        data = []
        task_id = None
        if len(messages) > batch_size:
            task_id = hashstr(messages)
            self.embed_state[task_id] = {"status": "in-progress", "total": len(messages), "progress": 0}

        for i in range(0, len(messages), batch_size):
            group_msg = messages[i : i + batch_size]
            logger.info(f"Encoding [{i}/{len(messages)}] messages (bsz={batch_size})")
            response = self.encode(group_msg)
            data.extend(response)
            if task_id:
                self.embed_state[task_id]["progress"] = i + len(group_msg)

        if task_id:
            self.embed_state[task_id]["status"] = "completed"

        return data

    async def abatch_encode(self, messages: list[str], batch_size: int | None = None) -> list[list[float]]:
        batch_size = batch_size or self.batch_size
        data = []
        task_id = None
        if len(messages) > batch_size:
            task_id = hashstr(messages)
            self.embed_state[task_id] = {"status": "in-progress", "total": len(messages), "progress": 0}

        # 保留原有逻辑：
        # 使用 asyncio.gather 并发执行所有 embedding 批次请求：
        # tasks = []
        # for i in range(0, len(messages), batch_size):
        #     group_msg = messages[i : i + batch_size]
        #     tasks.append(self.aencode(group_msg))

        # results = await asyncio.gather(*tasks)
        # for res in results:
        #     data.extend(res)

        # if task_id:
        #     self.embed_state[task_id]["progress"] = len(messages)
        #     self.embed_state[task_id]["status"] = "completed"

        # return data

        for i in range(0, len(messages), batch_size):
            group_msg = messages[i : i + batch_size]
            logger.info(f"Async encoding [{i}/{len(messages)}] messages (bsz={batch_size})")
            res = await self.aencode(group_msg)
            data.extend(res)
            if task_id:
                self.embed_state[task_id]["progress"] = i + len(group_msg)

        if task_id:
            self.embed_state[task_id]["status"] = "completed"

        return data

    async def test_connection(self) -> tuple[bool, str]:
        """
        测试embedding模型的连接性

        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            # 使用简单的测试文本
            test_text = ["Hello world"]
            await self.aencode(test_text)
            return True, "连接正常"
        except Exception as e:
            error_msg = str(e)
            error_msg += f", maybe you can check the `{self.base_url}` end with /embeddings as examples."
            logger.error(error_msg)
            return False, error_msg


class OllamaEmbedding(BaseEmbeddingModel):
    """
    Ollama Embedding Model
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.base_url = self.base_url or get_docker_safe_url("http://localhost:11434/api/embed")

    def encode(self, message: list[str] | str) -> list[list[float]]:
        if isinstance(message, str):
            message = [message]

        payload = {"model": self.model, "input": message}
        try:
            response = requests.post(self.base_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            if "embeddings" not in result:
                raise ValueError(f"Ollama Embedding failed: Invalid response format {result}")
            return result["embeddings"]
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.error(f"Ollama Embedding request failed: {e}, {payload}")
            raise ValueError(f"Ollama Embedding request failed: {e}")

    async def aencode(self, message: list[str] | str) -> list[list[float]]:
        if isinstance(message, str):
            message = [message]

        payload = {"model": self.model, "input": message}
        async with httpx.AsyncClient() as client:
            try:
                print(f"\n\n\nOllama Embedding request: {payload}\n\n\n")
                response = await client.post(self.base_url, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                if "embeddings" not in result:
                    raise ValueError(f"Ollama Embedding failed: Invalid response format {result}")
                return result["embeddings"]
            except (httpx.RequestError, json.JSONDecodeError) as e:
                raise ValueError(f"Ollama Embedding async request failed: {e}, {payload}, {self.base_url=}")


class OtherEmbedding(BaseEmbeddingModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def build_payload(self, message: list[str] | str) -> dict:
        return {"model": self.model, "input": message}

    def encode(self, message: list[str] | str) -> list[list[float]]:
        payload = self.build_payload(message)
        try:
            response = requests.post(self.base_url, json=payload, headers=self.headers, timeout=60)
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict) or "data" not in result:
                raise ValueError(f"Other Embedding failed: Invalid response format {result}")
            return [item["embedding"] for item in result["data"]]
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.error(f"Other Embedding request failed: {e}, {payload}")
            raise ValueError(f"Other Embedding request failed: {e}")

    async def aencode(self, message: list[str] | str) -> list[list[float]]:
        payload = self.build_payload(message)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.base_url, json=payload, headers=self.headers, timeout=60)
                response.raise_for_status()
                result = response.json()
                if not isinstance(result, dict) or "data" not in result:
                    raise ValueError(f"Other Embedding failed: Invalid response format {result}")
                return [item["embedding"] for item in result["data"]]
            except (httpx.RequestError, json.JSONDecodeError) as e:
                raise ValueError(f"Other Embedding async request failed: {e}, {payload}, {self.base_url=}")


class LocalEmbedding(BaseEmbeddingModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_id = kwargs.get("model_id") or self.model
        self.model_path = Path(self.base_url)
        self.device = _resolve_local_model_device()
        self.tokenizer = None
        self.model = None
        self._loaded = False
        self._pooling_mode = "cls"

    def _resolve_pooling_mode(self) -> str:
        pooling_config_path = self.model_path / "1_Pooling" / "config.json"
        if not pooling_config_path.exists():
            return "cls"

        try:
            config_data = json.loads(pooling_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Failed to read pooling config from {pooling_config_path}: {exc}")
            return "cls"

        if config_data.get("pooling_mode_mean_tokens"):
            return "mean"
        return "cls"

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        model_key = f"{self.model_path.resolve()}::{self.device}"

        with _LOCAL_EMBEDDING_CACHE_LOCK:
            cached = _LOCAL_EMBEDDING_CACHE.get(model_key)
            if cached is None:
                if not self.model_path.exists():
                    raise FileNotFoundError(f"Local embedding model path not found: {self.model_path}")

                logger.info(f"Loading local embedding model from {self.model_path}")
                tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
                model = AutoModel.from_pretrained(str(self.model_path), local_files_only=True)
                model.to(self.device)
                model.eval()
                cached = (tokenizer, model)
                _LOCAL_EMBEDDING_CACHE[model_key] = cached

            self.tokenizer, self.model = cached
            self._pooling_mode = self._resolve_pooling_mode()
            self._loaded = True

    def _pool_hidden_state(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self._pooling_mode == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
            summed = (hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            return summed / counts

        return hidden_state[:, 0]

    def encode(self, message: list[str] | str) -> list[list[float]]:
        texts = [message] if isinstance(message, str) else list(message)
        if not texts:
            return []

        self._ensure_loaded()
        assert self.tokenizer is not None
        assert self.model is not None

        max_length = getattr(self.tokenizer, "model_max_length", 8192)
        if not isinstance(max_length, int) or max_length <= 0 or max_length > 8192:
            max_length = 8192

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            pooled = self._pool_hidden_state(outputs.last_hidden_state, inputs["attention_mask"])
            normalized = F.normalize(pooled, p=2, dim=1)

        return normalized.cpu().tolist()

    async def aencode(self, message: list[str] | str) -> list[list[float]]:
        return await asyncio.to_thread(self.encode, message)


async def test_embedding_model_status(model_id: str) -> dict:
    """
    测试指定embedding模型的状态

    Args:
        model_id: 模型ID，格式为 "provider/model_name"

    Returns:
        dict: 包含状态信息的字典
    """
    try:
        support_embed_models = config.embed_model_names.keys()
        if model_id not in support_embed_models:
            return {"model_id": model_id, "status": "unsupported", "message": f"不支持的模型: {model_id}"}

        # 选择并创建模型实例
        model = select_embedding_model(model_id)

        # 测试连接
        success, message = await model.test_connection()

        return {
            "model_id": model_id,
            "status": "available" if success else "unavailable",
            "message": message if not success else "连接正常",
            "dimension": model.dimension,
        }

    except Exception as e:
        logger.warning(f"测试embedding模型状态失败 {model_id}: {e}")
        return {"model_id": model_id, "status": "error", "message": str(e)}


async def test_all_embedding_models_status() -> dict:
    """
    测试所有支持的embedding模型状态

    Returns:
        dict: 包含所有模型状态的字典
    """
    support_embed_models = list(config.embed_model_names.keys())
    results = {}

    # 并发测试所有模型
    tasks = [test_embedding_model_status(model_id) for model_id in support_embed_models]
    model_statuses = await asyncio.gather(*tasks, return_exceptions=True)

    for i, status in enumerate(model_statuses):
        if isinstance(status, Exception):
            model_id = support_embed_models[i]
            results[model_id] = {"model_id": model_id, "status": "error", "message": str(status)}
        else:
            results[status["model_id"]] = status

    return {
        "models": results,
        "total": len(support_embed_models),
        "available": len([m for m in results.values() if m["status"] == "available"]),
    }


def select_embedding_model(model_id):
    provider, model_name = model_id.split("/", 1) if model_id else ("", "")
    support_embed_models = config.embed_model_names.keys()
    assert model_id in support_embed_models, f"Unsupported embed model: {model_id}, only support {support_embed_models}"
    logger.info(f"Loading embedding model {model_id}")

    # 获取嵌入模型配置并转换为字典
    embed_config = config.embed_model_names[model_id].model_dump()

    if provider == "local":
        model = LocalEmbedding(**embed_config)
    elif provider == "ollama":
        model = OllamaEmbedding(**embed_config)
    else:
        model = OtherEmbedding(**embed_config)

    return model
