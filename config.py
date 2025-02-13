from dataclasses import dataclass, field
from typing import List


@dataclass
class VectorStoreConfig:
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "chat_with_docs"


@dataclass
class DocumentConfig:
    input_dir: str = "./docs"
    file_extensions: List[str] = None

    def __post_init__(self):
        if self.file_extensions is None:
            self.file_extensions = [".pdf"]


@dataclass
class ModelConfig:
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    llm_model: str = "llama3.2:1b"
    llm_timeout: int = 120
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 3
    similarity_top_k: int = 10


@dataclass
class Config:
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    document: DocumentConfig = field(default_factory=DocumentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
