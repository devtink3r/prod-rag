"""Typed config: config.yaml for tunables, .env for secrets."""

from functools import lru_cache
from pathlib import Path

import os

import yaml
from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PathsConfig(BaseModel):
    source_dir: Path = Path("data/docs")
    cache_dir: Path = Path(".cache")


class IngestionConfig(BaseModel):
    include_patterns: list[str] = ["**/*.pdf", "**/*.md"]
    workers: int = 2
    pipeline_version: int = 1


class CleaningConfig(BaseModel):
    boilerplate_page_frequency: float = 0.6
    min_doc_words: int = 50
    near_duplicate_threshold: float = 0.9


class ChunkingConfig(BaseModel):
    target_tokens: int = 512
    max_tokens: int = 1024
    min_tokens: int = 100
    forced_split_overlap: int = 50
    parent_tokens: int = 2048
    contextual_summaries: bool = True


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    device: str = "auto"
    batch_size: int = 8
    max_length: int = 8192


class VectorStoreConfig(BaseModel):
    collection: str = "rag_chunks"
    url: str = "http://localhost:6333"


class RegistryConfig(BaseModel):
    schema_name: str = "rag"

    model_config = {"populate_by_name": True}


class RetrievalConfig(BaseModel):
    fused_top_k: int = 30
    rerank_top_n: int = 8
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_score_floor: float = 0.15
    context_token_budget: int = 10000


class LLMConfig(BaseModel):
    answer_model: str = "deepseek/deepseek-v4-flash-0731"
    utility_model: str = "deepseek/deepseek-v4-flash"
    temperature: float = 0.1
    max_tokens: int = 2048


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class Secrets(BaseSettings):
    """Loaded from environment / .env — never from config.yaml."""

    openrouter_api_key: str = ""
    postgres_dsn: str = "postgresql://rag:rag@localhost:5433/rag"
    api_key: str = ""
    # accepts either RAG_HF_TOKEN or plain HF_TOKEN in .env / environment
    hf_token: str = Field("", validation_alias=AliasChoices("RAG_HF_TOKEN", "HF_TOKEN"))

    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=PROJECT_ROOT / ".env", extra="ignore"
    )


class Config(BaseModel):
    paths: PathsConfig = PathsConfig()
    ingestion: IngestionConfig = IngestionConfig()
    cleaning: CleaningConfig = CleaningConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    registry: RegistryConfig = RegistryConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig = LLMConfig()
    api: APIConfig = APIConfig()

    def resolve_path(self, p: Path) -> Path:
        return p if p.is_absolute() else PROJECT_ROOT / p


@lru_cache
def load_config(path: Path | None = None) -> Config:
    cfg_file = path or PROJECT_ROOT / "config.yaml"
    data = yaml.safe_load(cfg_file.read_text()) or {}
    if "registry" in data and "schema" in data["registry"]:
        data["registry"]["schema_name"] = data["registry"].pop("schema")
    return Config.model_validate(data)


@lru_cache
def load_secrets() -> Secrets:
    return Secrets()


def export_hf_token() -> None:
    """Expose the HF token to huggingface_hub for authenticated downloads."""
    token = load_secrets().hf_token
    if token:
        os.environ.setdefault("HF_TOKEN", token)
