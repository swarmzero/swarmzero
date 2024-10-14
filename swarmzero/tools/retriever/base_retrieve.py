import os
import pickle

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

from swarmzero.server.routes import files

supported_exts = [".md", ".mdx", ".txt", ".csv", ".docx", ".pdf"]
index_base_dir = "swarmzero-data/index/store/"

os.makedirs(index_base_dir, exist_ok=True)


class IndexStore:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.indexes = {}
        self.index_files = {}  # New dictionary to store files for each index

    def save_to_file(self, file_path='indexes.pkl'):
        """Saves the current indexes and index_files to a file."""
        with open(index_base_dir + file_path, 'wb') as file:
            pickle.dump((self.indexes, self.index_files), file)
        return f"Indexes and file lists saved to {file_path}."

    @classmethod
    def load_from_file(cls, file_path='indexes.pkl'):
        """Loads indexes and index_files from a file."""
        with open(index_base_dir + file_path, 'rb') as file:
            loaded_indexes, loaded_index_files = pickle.load(file)
        instance = cls.get_instance()
        instance.indexes = loaded_indexes
        instance.index_files = loaded_index_files
        return instance

    def add_index(self, index_name, index, file_list):
        if index_name in self.indexes:
            raise ValueError("An index with this name already exists.")
        self.indexes[index_name] = index
        self.index_files[index_name] = file_list
        return f"Index '{index_name}' added successfully with {len(file_list)} files."

    def get_index(self, index_name):
        if index_name not in self.indexes:
            raise KeyError("No index found with this name.")
        return self.indexes[index_name]

    def update_index(self, index_name, new_index):
        if index_name not in self.indexes:
            raise KeyError("No index found with this name to update.")
        self.indexes[index_name] = new_index
        return f"Index '{index_name}' updated successfully."

    def delete_index(self, index_name):
        if index_name not in self.indexes:
            raise KeyError("No index found with this name to delete.")
        del self.indexes[index_name]
        del self.index_files[index_name]
        return f"Index '{index_name}' and its file list deleted successfully."

    def list_indexes(self):
        return list(self.indexes.keys())

    def get_all_indexes(self):
        """Returns a list of all index objects stored in the index store."""
        return list(self.indexes.values())

    def get_all_index_names(self):
        """Returns a list of all index objects stored in the index store."""
        return list(self.indexes.keys())

    def get_index_files(self, index_name):
        if index_name not in self.index_files:
            raise KeyError("No file list found for this index name.")
        return self.index_files[index_name]

    def update_index_files(self, index_name, new_file_list):
        if index_name not in self.index_files:
            raise KeyError("No file list found for this index name to update.")
        self.index_files[index_name] = new_file_list
        return f"File list for index '{index_name}' updated successfully."

    def insert_index_files(self, index_name, new_files):
        if index_name not in self.index_files:
            raise KeyError("No index found with this name.")
        self.index_files[index_name].extend(new_files)
        return f"{len(new_files)} files inserted into index '{index_name}' successfully."


class RetrieverBase:
    def __init__(
        self,
        required_exts=supported_exts,
        retrieve_data_path=files.BASE_DIR,
        name="BaseRetriever",
        description="This tool creates a base retriever index",
    ):
        self.retrieve_data_path = retrieve_data_path
        self.required_exts = required_exts
        self.name = name
        self.description = description

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
        index = VectorStoreIndex.from_documents(documents)
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
