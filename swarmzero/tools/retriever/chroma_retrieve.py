import os

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from swarmzero.tools.retriever.base_retrieve import RetrieverBase


class ChromaRetriever(RetrieverBase):
    def __init__(self, base_dir="swarmzero-data/index/chromadb"):
        super().__init__(
            name="ChromaRetriever",
            description="This tool creates chroma retriever index",
        )
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def create_index(self, file_path=None, folder_path=None, collection_name="swarmzero_chroma"):
        documents, file_names = self._load_documents(file_path, folder_path)

        chroma_client = chromadb.PersistentClient(path=self.base_dir)
        chroma_collection = chroma_client.get_or_create_collection(collection_name)

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        return index, file_names

    def delete_collection(self, collection_name="swarmzero_chroma"):
        chroma_client = chromadb.PersistentClient(path=self.base_dir)
        chroma_client.delete_collection(collection_name)
