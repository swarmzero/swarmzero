import os

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

from swarmzero.sdk_context import SDKContext
from swarmzero.server.routes import files

supported_exts = [".md", ".mdx", ".txt", ".csv", ".docx", ".pdf"]

class RetrieverBase:
    def __init__(
        self,
        required_exts=supported_exts,
        retrieve_data_path=files.BASE_DIR,
        name="BaseRetriever",
        description="This tool creates a base retriever index",
        sdk_context: SDKContext = None,
    ):
        self.retrieve_data_path = retrieve_data_path
        self.required_exts = required_exts
        self.name = name
        self.description = description
        self.sdk_context = sdk_context

    def _load_documents(self, file_path=None, folder_path=None):
        if file_path is None:
            if folder_path is None:
                folder_path = self.retrieve_data_path

        reader = SimpleDirectoryReader(
            input_files=file_path,
            input_dir=folder_path,
            required_exts=self.required_exts,
            recursive=True,
            filename_as_id=True,
        )
        documents = reader.load_data()

        file_names = []
        if file_path:
            file_names = [os.path.basename(f) for f in file_path]
        elif folder_path:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if any(file.endswith(ext) for ext in self.required_exts):
                        file_names.append(file)

        return documents, file_names

    def create_basic_index(self, file_path=None, folder_path=None):
        documents, file_names = self._load_documents(file_path, folder_path)
        index = VectorStoreIndex.from_documents(
            documents, callback_manager=self.sdk_context.get_utility("callback_manager")
        )
        return index, file_names

    def insert_documents(self, index, file_path=None, folder_path=None):
        documents, file_names = self._load_documents(file_path, folder_path)
        if not documents:
            raise KeyError("No documents found to insert.")

        for document in documents:
            index.insert(document)

        return f"{len(documents)} documents inserted successfully."

    def update_documents(self, index, file_path=None, folder_path=None):
        documents, file_names = self._load_documents(file_path, folder_path)
        if not documents:
            raise KeyError("No documents found to update.")

        index.refresh(documents)
        return f"{len(documents)} documents updated successfully."

    def delete_documents(self, index, file_path=None, folder_path=None):
        documents, file_names = self._load_documents(file_path, folder_path)
        if not documents:
            raise KeyError("No documents found to delete.")

        document_ids = [doc.doc_id for doc in documents]
        for doc_id in document_ids:
            index.delete(doc_id)

        return f"{len(document_ids)} documents deleted successfully."
