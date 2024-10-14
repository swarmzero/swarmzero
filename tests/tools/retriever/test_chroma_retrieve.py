from unittest import mock

import pytest

from swarmzero.tools.retriever.chroma_retrieve import ChromaRetriever


@pytest.fixture
def chroma_retriever():
    return ChromaRetriever()


def test_create_index(chroma_retriever):
    with (
        mock.patch('swarmzero.tools.retriever.chroma_retrieve.chromadb.PersistentClient') as MockClient,
        mock.patch('swarmzero.tools.retriever.chroma_retrieve.ChromaVectorStore') as MockVectorStore,
        mock.patch('swarmzero.tools.retriever.chroma_retrieve.StorageContext') as MockStorageContext,
        mock.patch('swarmzero.tools.retriever.chroma_retrieve.VectorStoreIndex') as MockVectorStoreIndex,
        mock.patch.object(
            chroma_retriever, '_load_documents', return_value=(['doc1', 'doc2'], ['file1.txt', 'file2.txt'])
        ),
    ):

        mock_client_instance = MockClient.return_value
        mock_collection = mock_client_instance.get_or_create_collection.return_value
        mock_vector_store_instance = MockVectorStore.return_value
        mock_storage_context_instance = MockStorageContext.from_defaults.return_value
        mock_index_instance = MockVectorStoreIndex.from_documents.return_value

        index, file_names = chroma_retriever.create_index(file_path='dummy_path')

        chroma_retriever._load_documents.assert_called_once_with('dummy_path', None)
        MockClient.assert_called_once_with(path=chroma_retriever.base_dir)
        mock_client_instance.get_or_create_collection.assert_called_once_with('swarmzero_chroma')
        MockVectorStore.assert_called_once_with(chroma_collection=mock_collection)
        MockStorageContext.from_defaults.assert_called_once_with(vector_store=mock_vector_store_instance)
        MockVectorStoreIndex.from_documents.assert_called_once_with(
            ['doc1', 'doc2'], storage_context=mock_storage_context_instance
        )
        assert index == mock_index_instance
        assert file_names == ['file1.txt', 'file2.txt']


def test_delete_collection(chroma_retriever):
    with mock.patch('swarmzero.tools.retriever.chroma_retrieve.chromadb.PersistentClient') as MockClient:
        mock_client_instance = MockClient.return_value

        chroma_retriever.delete_collection(collection_name='test_collection')

        MockClient.assert_called_once_with(path=chroma_retriever.base_dir)
        mock_client_instance.delete_collection.assert_called_once_with('test_collection')


def test_create_index_with_single_document(chroma_retriever):
    with (
        mock.patch('swarmzero.tools.retriever.chroma_retrieve.chromadb.PersistentClient') as MockClient,
        mock.patch('swarmzero.tools.retriever.chroma_retrieve.ChromaVectorStore') as MockVectorStore,
        mock.patch('swarmzero.tools.retriever.chroma_retrieve.StorageContext') as MockStorageContext,
        mock.patch('swarmzero.tools.retriever.chroma_retrieve.VectorStoreIndex') as MockVectorStoreIndex,
        mock.patch.object(chroma_retriever, '_load_documents', return_value=(['single_doc'], ['single_file.txt'])),
    ):

        mock_storage_context_instance = MockStorageContext.from_defaults.return_value
        mock_index_instance = MockVectorStoreIndex.from_documents.return_value

        index, file_names = chroma_retriever.create_index(file_path='single_doc_path')

        MockVectorStoreIndex.from_documents.assert_called_once_with(
            ['single_doc'], storage_context=mock_storage_context_instance
        )
        assert index == mock_index_instance
        assert file_names == ['single_file.txt']
