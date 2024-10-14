import shutil
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi import APIRouter, FastAPI
from httpx import AsyncClient

from swarmzero.filestore import FileStore
from swarmzero.sdk_context import SDKContext
from swarmzero.server.routes.files import setup_files_routes
from swarmzero.tools.retriever.base_retrieve import IndexStore, RetrieverBase

BASE_DIR = "test_files"


@pytest.fixture(scope="module")
def file_store():
    store = FileStore(BASE_DIR)
    yield store
    shutil.rmtree(BASE_DIR, ignore_errors=True)


@pytest.fixture(scope="module")
def sdk_context():
    context = MagicMock(spec=SDKContext)
    mock_agent = MagicMock()
    mock_agent.recreate_agent = MagicMock()
    context.get_resource.return_value = mock_agent
    return context


@pytest.fixture(scope="module")
def app(file_store, sdk_context):
    fastapi_app = FastAPI()
    router = APIRouter()
    setup_files_routes(router, "test_id", sdk_context)
    fastapi_app.include_router(router)
    return fastapi_app


@pytest.fixture
async def client(app):
    with (
        patch.object(RetrieverBase, "create_basic_index", return_value=(MagicMock(), ["test_file"])),
        patch.object(RetrieverBase, "insert_documents"),
        patch.object(IndexStore, "save_to_file", MagicMock()),
    ):
        async with AsyncClient(app=app, base_url="http://test") as test_client:
            yield test_client


@pytest.mark.asyncio
async def test_list_files(client):
    with patch.object(FileStore, 'list_files', return_value=["test_list.txt"]):
        response = await client.get("/files/")
        assert response.status_code == 200
        assert "test_list.txt" in response.json()["files"]


@pytest.mark.asyncio
async def test_delete_file(client):
    with patch.object(FileStore, 'delete_file', return_value=True):
        response = await client.delete("/files/test_delete.txt")
        assert response.status_code == 200
        assert response.json() == {"message": "File test_delete.txt deleted successfully."}


@pytest.mark.asyncio
async def test_rename_file(client):
    with (
        patch.object(FileStore, 'rename_file', return_value=True),
        patch.object(FileStore, 'save_file', return_value="old_name.txt"),
    ):
        files = [
            ("files", ("old_name.txt", BytesIO(b"test content"), "text/plain")),
        ]
        response = await client.post("/uploadfiles/", files=files)
        assert response.status_code == 200

        response = await client.put("/files/old_name.txt/new_name.txt")
        assert response.status_code == 200
        assert response.json() == {"message": "File old_name.txt renamed to new_name.txt successfully."}


@pytest.mark.asyncio
async def test_create_upload_files(client):
    with patch.object(FileStore, 'save_file', return_value="test.txt"):
        files = [
            ("files", ("test.txt", BytesIO(b"test content"), "text/plain")),
        ]
        response = await client.post("/uploadfiles/", files=files)
        assert response.status_code == 200
        assert response.json() == {"filenames": ["swarmzero-data/files/user/test.txt"]}


@pytest.mark.asyncio
async def test_upload_image_jpeg(client):
    with patch.object(FileStore, 'save_file', return_value="test.jpeg"):
        files = [
            ("files", ("test.jpeg", BytesIO(b"JPEG content"), "image/jpeg")),
        ]
        response = await client.post("/uploadfiles/", files=files)
        assert response.status_code == 200
        assert response.json() == {"filenames": ["swarmzero-data/files/user/test.jpeg"]}


@pytest.mark.asyncio
async def test_upload_image_png(client):
    with patch.object(FileStore, 'save_file', return_value="test.png"):
        files = [
            ("files", ("test.png", BytesIO(b"PNG content"), "image/png")),
        ]
        response = await client.post("/uploadfiles/", files=files)
        assert response.status_code == 200
        assert response.json() == {"filenames": ["swarmzero-data/files/user/test.png"]}


@pytest.mark.asyncio
async def test_upload_image_jpg(client):
    with patch.object(FileStore, 'save_file', return_value="test.jpg"):
        files = [
            ("files", ("test.jpg", BytesIO(b"JPG content"), "image/jpg")),
        ]
        response = await client.post("/uploadfiles/", files=files)
        assert response.status_code == 200
        assert response.json() == {"filenames": ["swarmzero-data/files/user/test.jpg"]}


@pytest.mark.asyncio
async def test_upload_application_pdf(client):
    with patch.object(FileStore, 'save_file', return_value="test.pdf"):
        files = [
            ("files", ("test.pdf", BytesIO(b"PDF content"), "application/pdf")),
        ]
        response = await client.post("/uploadfiles/", files=files)
        assert response.status_code == 200
        assert response.json() == {"filenames": ["swarmzero-data/files/user/test.pdf"]}


@pytest.mark.asyncio
async def test_upload_application_msword(client):
    with patch.object(FileStore, 'save_file', return_value="test.doc"):
        files = [
            ("files", ("test.doc", BytesIO(b"MS Word content"), "application/msword")),
        ]
        response = await client.post("/uploadfiles/", files=files)
        assert response.status_code == 200
        assert response.json() == {"filenames": ["swarmzero-data/files/user/test.doc"]}


@pytest.mark.asyncio
async def test_upload_application_vnd_ms_excel(client):
    with patch.object(FileStore, 'save_file', return_value="test.xls"):
        files = [
            ("files", ("test.xls", BytesIO(b"MS Excel content"), "application/vnd.ms-excel")),
        ]
        response = await client.post("/uploadfiles/", files=files)
        assert response.status_code == 200
        assert response.json() == {"filenames": ["swarmzero-data/files/user/test.xls"]}


@pytest.mark.asyncio
async def test_upload_text_csv(client):
    with patch.object(FileStore, 'save_file', return_value="test.csv"):
        files = [
            ("files", ("test.csv", BytesIO(b"CSV content"), "text/csv")),
        ]
        response = await client.post("/uploadfiles/", files=files)
        assert response.status_code == 200
        assert response.json() == {"filenames": ["swarmzero-data/files/user/test.csv"]}


@pytest.mark.asyncio
async def test_upload_text_markdown(client):
    with patch.object(FileStore, 'save_file', return_value="test.md"):
        files = [
            ("files", ("test.md", BytesIO(b"Markdown content"), "text/markdown")),
        ]
        response = await client.post("/uploadfiles/", files=files)
        assert response.status_code == 200
        assert response.json() == {"filenames": ["swarmzero-data/files/user/test.md"]}


@pytest.mark.asyncio
async def test_upload_unsupported_file_type(client):
    files = [
        ("files", ("test.exe", BytesIO(b"EXE content"), "application/x-msdownload")),
    ]
    response = await client.post("/uploadfiles/", files=files)
    assert response.status_code == 400
    assert "File type application/x-msdownload is not allowed" in response.json()["detail"]
