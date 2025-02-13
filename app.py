import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import nest_asyncio
import qdrant_client
import streamlit as st
from typing import List

import torch

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    PromptTemplate,
    Settings,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SentenceTransformerRerank

from config import Config

torch.classes.__path__ = []


class DocumentLoader:
    def __init__(self, config: Config):
        self.config = config

    def load(self) -> List:
        loader = SimpleDirectoryReader(
            input_dir=self.config.document.input_dir,
            required_exts=self.config.document.file_extensions,
            recursive=True,
        )
        return loader.load_data()


class VectorStoreManager:
    def __init__(self, config: Config):
        self.config = config
        self.client = qdrant_client.QdrantClient(
            host=config.vector_store.host, port=config.vector_store.port
        )

    def create_store(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.config.vector_store.collection_name,
        )


class RAGSystem:
    def __init__(self, config: Config):
        self.config = config
        self.setup_embedding_model()
        self.setup_llm()

    def setup_embedding_model(self):
        embed_model = HuggingFaceEmbedding(
            model_name=self.config.model.embedding_model, trust_remote_code=True
        )
        Settings.embed_model = embed_model

    def setup_llm(self):
        llm = Ollama(
            model=self.config.model.llm_model, request_timeout=self.config.model.llm_timeout
        )
        Settings.llm = llm

    def create_index(self, documents: List) -> VectorStoreIndex:
        vector_store = VectorStoreManager(self.config).create_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, show_progress=True
        )

    def setup_query_engine(self, index: VectorStoreIndex, qa_template: PromptTemplate):
        reranker = SentenceTransformerRerank(
            model=self.config.model.reranker_model, top_n=self.config.model.reranker_top_n
        )

        query_engine = index.as_query_engine(
            similarity_top_k=self.config.model.similarity_top_k,
            node_postprocessors=[reranker],
        )

        query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_template})

        return query_engine


def get_qa_template() -> PromptTemplate:
    template = """Context information is below:
                  ---------------------
                  {context_str}
                  ---------------------
                  Given the context information above I want you to think
                  step by step to answer the query in a crisp manner,
                  incase you don't know the answer say 'I don't know!'
                
                  Query: {query_str}
            
                  Answer:"""
    return PromptTemplate(template)


def initialize_rag():
    if "rag_system" not in st.session_state:
        config = Config()

        with st.spinner("Loading RAG system..."):
            # Initialize components
            doc_loader = DocumentLoader(config)
            rag_system = RAGSystem(config)

            # Load and index documents
            documents = doc_loader.load()
            if not documents:
                st.error("❌ No documents found in the specified directory")
                st.stop()

            # Success message
            st.success(f"✅ Documents loaded successfully")

            # Info message
            st.info(f"ℹ️ Loaded {len(documents)} documents")

            # Create index and query engine
            index = rag_system.create_index(documents)
            query_engine = rag_system.setup_query_engine(index, get_qa_template())

            # Store in session state
            st.session_state.rag_system = rag_system
            st.session_state.query_engine = query_engine
            st.session_state.chat_history = []


def main():
    try:
        # Apply nest_asyncio at startup
        nest_asyncio.apply()

        st.set_page_config(page_title="Document RAG System", page_icon="📚", layout="wide")

        st.title("📚 Document RAG System")

        # Initialize RAG system
        initialize_rag()

        # Chat interface
        st.write("### Chat with your documents")

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents"):
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)

            # Add to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.query_engine.query(prompt)
                    st.write(response.response)

                    # Add sources if available
                    if hasattr(response, "source_nodes") and response.source_nodes:
                        with st.expander("View Sources"):
                            for idx, source in enumerate(response.source_nodes):
                                st.markdown(f"**Source {idx + 1}:**")
                                st.markdown(source.node.text)
                                st.markdown("---")

            # Add to chat history
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response.response}
            )

        # Sidebar
        with st.sidebar:
            st.header("About")
            st.write(
                """
            This is a RAG (Retrieval-Augmented Generation) system that allows you to chat with your documents.
            Upload your documents to the `docs` directory and ask questions about them.
            """
            )

            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

    except Exception as e:
        print(f"🔴 An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
