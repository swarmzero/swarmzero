from unittest.mock import MagicMock, patch

import pytest

from swarmzero.tools.retriever.base_retrieve import IndexStore, RetrieverBase


@pytest.fixture
def retriever_base():
    return RetrieverBase()


@pytest.fixture
def index_store():
    IndexStore._instance = None  # Reset singleton instance before each test
    return IndexStore.get_instance()


def test_retriever_base_initialization(retriever_base):
    assert retriever_base.name == "BaseRetriever"
    assert retriever_base.description == "This tool creates a base retriever index"
    assert retriever_base.required_exts == [".md", ".mdx", ".txt", ".csv", ".docx", ".pdf"]
    assert retriever_base.retrieve_data_path == "swarmzero-data/files/user"


@patch('swarmzero.tools.retriever.base_retrieve.RetrieverBase._load_documents')
@patch('llama_index.core.VectorStoreIndex.from_documents', return_value="index")
def test_create_basic_index(mock_from_documents, mock_load_documents, retriever_base):
    mock_documents = [MagicMock(doc_id="doc1"), MagicMock(doc_id="doc2")]
    mock_load_documents.return_value = (mock_documents, ["doc1", "doc2"])
    index, file_names = retriever_base.create_basic_index()
    mock_load_documents.assert_called_once()
    mock_from_documents.assert_called_once_with(mock_documents)
    assert index == "index"
    assert file_names == ["doc1", "doc2"]


@patch('swarmzero.tools.retriever.base_retrieve.RetrieverBase._load_documents')
def test_insert_documents(mock_load_documents, retriever_base):
    mock_documents = [MagicMock(doc_id="doc1"), MagicMock(doc_id="doc2")]
    mock_load_documents.return_value = (mock_documents, ["doc1", "doc2"])
    mock_index = MagicMock()
    result = retriever_base.insert_documents(mock_index)
    mock_load_documents.assert_called_once()
    assert mock_index.insert.call_count == 2
    assert result == "2 documents inserted successfully."


@patch('swarmzero.tools.retriever.base_retrieve.RetrieverBase._load_documents')
def test_update_documents(mock_load_documents, retriever_base):
    mock_documents = [MagicMock(doc_id="doc1"), MagicMock(doc_id="doc2")]
    mock_load_documents.return_value = (mock_documents, ["doc1", "doc2"])
    mock_index = MagicMock()
    result = retriever_base.update_documents(mock_index)
    mock_load_documents.assert_called_once()
    mock_index.refresh.assert_called_once_with(mock_documents)
    assert result == "2 documents updated successfully."


@patch('swarmzero.tools.retriever.base_retrieve.RetrieverBase._load_documents')
def test_delete_documents(mock_load_documents, retriever_base):
    mock_documents = [MagicMock(doc_id="doc1"), MagicMock(doc_id="doc2")]
    mock_load_documents.return_value = (mock_documents, ["doc1", "doc2"])
    mock_index = MagicMock()
    result = retriever_base.delete_documents(mock_index)
    mock_load_documents.assert_called_once()
    mock_index.delete.assert_any_call("doc1")
    mock_index.delete.assert_any_call("doc2")
    assert result == "2 documents deleted successfully."


def test_delete_index_nonexistent(index_store):
    with pytest.raises(KeyError, match="No index found with this name to delete."):
        index_store.delete_index("nonexistent_index")


def test_list_indexes(index_store):
    mock_index1 = MagicMock()
    mock_index2 = MagicMock()
    index_store.add_index("index1", mock_index1, ["file1.txt"])
    index_store.add_index("index2", mock_index2, ["file2.txt"])
    assert index_store.list_indexes() == ["index1", "index2"]


def test_get_all_indexes(index_store):
    mock_index1 = MagicMock()
    mock_index2 = MagicMock()
    index_store.add_index("index1", mock_index1, ["file1.txt"])
    index_store.add_index("index2", mock_index2, ["file2.txt"])
    assert index_store.get_all_indexes() == [mock_index1, mock_index2]


def test_get_index_files(index_store):
    mock_index = MagicMock()
    file_list = ["file1.txt", "file2.txt"]
    index_store.add_index("test_index", mock_index, file_list)
    assert index_store.get_index_files("test_index") == file_list


def test_get_index_files_nonexistent(index_store):
    with pytest.raises(KeyError, match="No file list found for this index name."):
        index_store.get_index_files("nonexistent_index")


def test_update_index_files(index_store):
    mock_index = MagicMock()
    file_list = ["file1.txt"]
    index_store.add_index("test_index", mock_index, file_list)
    new_file_list = ["file2.txt", "file3.txt"]
    result = index_store.update_index_files("test_index", new_file_list)
    assert result == "File list for index 'test_index' updated successfully."
    assert index_store.get_index_files("test_index") == new_file_list


def test_update_index_files_nonexistent(index_store):
    with pytest.raises(KeyError, match="No file list found for this index name to update."):
        index_store.update_index_files("nonexistent_index", ["file1.txt"])


def test_insert_index_files(index_store):
    mock_index = MagicMock()
    file_list = ["file1.txt"]
    index_store.add_index("test_index", mock_index, file_list)
    new_files = ["file2.txt", "file3.txt"]
    result = index_store.insert_index_files("test_index", new_files)
    assert result == "2 files inserted into index 'test_index' successfully."
    assert index_store.get_index_files("test_index") == ["file1.txt", "file2.txt", "file3.txt"]


def test_insert_index_files_nonexistent(index_store):
    with pytest.raises(KeyError, match="No index found with this name."):
        index_store.insert_index_files("nonexistent_index", ["file1.txt"])
