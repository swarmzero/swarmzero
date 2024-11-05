import os
import pickle

index_base_dir = "swarmzero-data/index/store/"

os.makedirs(index_base_dir, exist_ok=True)

class IndexStore:

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
