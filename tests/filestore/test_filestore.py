import os
import shutil
import base64
from io import BytesIO

import pytest
from fastapi import UploadFile

from swarmzero.filestore.filestore import FileStore


@pytest.fixture(scope="module")
def file_store():
    base_dir = "test_files"
    store = FileStore(base_dir)
    yield store
    shutil.rmtree(base_dir)


@pytest.mark.asyncio
async def test_save_file(file_store):
    content = b"test file content"
    upload_file = UploadFile(filename="test.txt", file=BytesIO(content))

    filename = await file_store.save_file(upload_file)
    assert filename == "test.txt"
    assert os.path.exists(os.path.join(file_store.base_dir, filename))


def test_delete_file(file_store):
    filename = "test_delete.txt"
    file_path = os.path.join(file_store.base_dir, filename)
    with open(file_path, "w") as f:
        f.write("test")

    assert file_store.delete_file(filename)
    assert not os.path.exists(file_path)


def test_list_files(file_store):
    filename = "test_list.txt"
    file_path = os.path.join(file_store.base_dir, filename)
    with open(file_path, "w") as f:
        f.write("test")

    files = file_store.list_files()
    assert filename in files


def test_rename_file(file_store):
    old_filename = "old_name.txt"
    new_filename = "new_name.txt"
    old_file_path = os.path.join(file_store.base_dir, old_filename)
    new_file_path = os.path.join(file_store.base_dir, new_filename)

    with open(old_file_path, "w") as f:
        f.write("test")

    assert file_store.rename_file(old_filename, new_filename)
    assert not os.path.exists(old_file_path)
    assert os.path.exists(new_file_path)


def test_get_file(file_store):
    filename = "get_file.txt"
    content = b"hello"
    # create file manually
    file_path = os.path.join(file_store.base_dir, filename)
    with open(file_path, "wb") as f:
        f.write(content)

    encoded = file_store.get_file(filename)
    assert encoded == base64.b64encode(content).decode("utf-8")
