from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    delta_dir: Path = ROOT / "data/delta"
    qdrant_path: Path = ROOT / "qdrant_local"
    model_cache: Path = ROOT / ".cache/fastembed"
    collection: str = "rbi_chunks"

    dense_model: str = "BAAI/bge-small-en-v1.5"  # 384-d, runs on CPU via ONNX
    sparse_model: str = "Qdrant/bm25"  # lexical (BM25) as sparse vectors, IDF applied by Qdrant
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"  # cross-encoder, ~80 MB

    anthropic_api_key: str = ""
    gen_model: str = "claude-haiku-5-5"  # cheap generator; a weaker model also stresses grounding
    judge_model: str = "claude-haiku-5-5"  # cheapest model for judging too (budget is $5)
    max_spend_usd: float = 3.0  # hard stop for cumulative API spend (budget is $5)
    gate_min_dense_score: float = 0.67  # below this the question is out of domain: skip the LLM
    api_keys: str = "dev-key-1"  # comma-separated; set real keys via the API_KEYS env var
    rate_limit_per_min: int = 10  # per API key


settings = Settings()
