import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import APIRouter, FastAPI
from httpx import AsyncClient

from swarmzero.api.routes.vectorindex import setup_vectorindex_routes


@pytest.fixture(scope="module")
def app():
    fastapi_app = FastAPI()
    router = APIRouter()
    setup_vectorindex_routes(router)
    fastapi_app.include_router(router)
    return fastapi_app


@pytest.fixture
async def client(app):
    async with AsyncClient(app=app, base_url="http://test") as test_client:
        yield test_client


@pytest.mark.asyncio
async def test_create_index_basic(client):
    with patch('swarmzero.tools.retriever.base_retrieve.RetrieverBase.create_basic_index') as mock_create_index:
        mock_create_index.return_value = (MagicMock(), ["test.txt"])
        params = {"index_name": "test_basic", "index_type": "basic", "file_path": ["test.txt"]}

        response = await client.post("/create_index/", params=params)

        assert response.status_code == 200
        assert response.json()["message"] == "Index test_basic created successfully."


@pytest.mark.asyncio
async def test_create_index_chroma(client):
    with patch('swarmzero.tools.retriever.chroma_retrieve.ChromaRetriever.create_index') as mock_create_index:
        mock_create_index.return_value = (MagicMock(), ["test.txt"])
        params = {"index_name": "test_chroma", "index_type": "chroma", "file_path": ["test.txt"]}

        response = await client.post("/create_index/", params=params)

        assert response.status_code == 200
        assert response.json()["message"] == "Index test_chroma created successfully."


@pytest.mark.asyncio
async def test_create_index_pinecone_serverless(client):
    with (
        patch(
            'swarmzero.tools.retriever.pinecone_retrieve.PineconeRetriever.create_serverless_index'
        ) as mock_create_index,
        patch.dict(os.environ, {'PINECONE_API_KEY': 'fake_api_key'}),
    ):
        mock_create_index.return_value = (MagicMock(), ["test.txt"])
        params = {
            "index_name": "test_pinecone_serverless",
            "index_type": "pinecone-serverless",
            "file_path": ["test.txt"],
        }

        response = await client.post("/create_index/", params=params)

        assert response.status_code == 200
        assert response.json()["message"] == "Index test_pinecone_serverless created successfully."


@pytest.mark.asyncio
async def test_create_index_pinecone_pod(client):
    with (
        patch('swarmzero.tools.retriever.pinecone_retrieve.PineconeRetriever.create_pod_index') as mock_create_index,
        patch.dict(os.environ, {'PINECONE_API_KEY': 'fake_api_key'}),
    ):
        mock_create_index.return_value = (MagicMock(), ["test.txt"])
        params = {"index_name": "test_pinecone_pod", "index_type": "pinecone-pod", "file_path": ["test.txt"]}

        response = await client.post("/create_index/", params=params)

        assert response.status_code == 200
        assert response.json()["message"] == "Index test_pinecone_pod created successfully."


@pytest.mark.asyncio
async def test_insert_documents(client):
    with (
        patch('swarmzero.server.routes.vectorindex.index_store') as mock_index_store,
        patch('swarmzero.tools.retriever.base_retrieve.RetrieverBase.insert_documents') as mock_insert,
    ):
        mock_index = MagicMock()
        mock_index_store.get_index.return_value = mock_index
        mock_insert.return_value = ("Documents inserted successfully.", ["test.txt"])
        params = {"index_name": "test_insert", "file_path": ["test.txt"]}

        response = await client.post("/insert_documents/", params=params)

        assert response.status_code == 200, f"Unexpected status code: {response.status_code}, response: {response.text}"
        assert response.json()["message"] == "Documents inserted successfully."


@pytest.mark.asyncio
async def test_update_documents(client):
    with (
        patch('swarmzero.server.routes.vectorindex.index_store') as mock_index_store,
        patch('swarmzero.tools.retriever.base_retrieve.RetrieverBase.update_documents') as mock_update,
    ):
        mock_index = MagicMock()
        mock_index_store.get_index.return_value = mock_index
        mock_update.return_value = ("Documents updated successfully.", ["test.txt"])
        params = {"index_name": "test_update", "file_path": ["test.txt"]}

        response = await client.put("/update_documents/", params=params)

        assert response.status_code == 200, f"Unexpected status code: {response.status_code}, response: {response.text}"
        assert response.json()["message"] == "Documents updated successfully."


@pytest.mark.asyncio
async def test_delete_documents(client):
    with (
        patch('swarmzero.server.routes.vectorindex.index_store') as mock_index_store,
        patch('swarmzero.tools.retriever.base_retrieve.RetrieverBase.delete_documents') as mock_delete,
    ):
        mock_index = MagicMock()
        mock_index_store.get_index.return_value = mock_index
        mock_delete.return_value = ("Documents deleted successfully.", ["test.txt"])
        params = {"index_name": "test_delete", "file_path": ["test.txt"]}

        response = await client.delete("/delete_documents/", params=params)

        assert response.status_code == 200, f"Unexpected status code: {response.status_code}, response: {response.text}"
        assert response.json()["message"] == "Documents deleted successfully."


@pytest.mark.asyncio
async def test_create_index_invalid_type(client):
    params = {"index_name": "test_invalid", "index_type": "invalid_type", "file_path": ["test.txt"]}

    response = await client.post("/create_index/", params=params)

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid index type provided."
