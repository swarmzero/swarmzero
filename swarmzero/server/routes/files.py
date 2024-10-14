import logging
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from swarmzero.filestore import BASE_DIR, FileStore

# TODO: get log level from config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from swarmzero.sdk_context import SDKContext
from swarmzero.tools.retriever.base_retrieve import IndexStore, RetrieverBase

ALLOWED_FILE_TYPES = [
    "application/json",
    "text/csv",
    "text/plain",
    "application/pdf",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "application/msword",
    "application/vnd.ms-excel",
    "text/markdown",
]

file_store = FileStore(BASE_DIR)

index_store = IndexStore.get_instance()


async def insert_files_to_index(files: List[UploadFile], id: str, sdk_context: SDKContext):
    saved_files = []
    for file in files:
        if not file.content_type:
            logger.warning(f"File {file.filename} has no content type.")
            raise HTTPException(status_code=400, detail="File content type is missing.")

        if file.content_type not in ALLOWED_FILE_TYPES:
            logger.warning(f"Disallowed file type upload attempted: {file.content_type}")
            raise HTTPException(
                status_code=400,
                detail=f"File type {file.content_type} is not allowed.",
            )
        try:
            agent = sdk_context.get_resource(id)
            filename = await file_store.save_file(file)
            file_path = "{BASE_DIR}/{filename}".format(BASE_DIR=BASE_DIR, filename=filename)
            saved_files.append(file_path)

            if "BaseRetriever" in index_store.list_indexes():
                index = index_store.get_index("BaseRetriever")
                RetrieverBase().insert_documents(index, [file_path])
                index_store.update_index("BaseRetriever", index)
                index_store.insert_index_files("BaseRetriever", [filename])
                logger.info("Inserting data to existing basic index")
                logger.info(f"Index: {index_store.list_indexes()}")
                agent.recreate_agent()
                index_store.save_to_file()

            else:
                retriever = RetrieverBase()
                index, file_names = retriever.create_basic_index([file_path])
                index_store.add_index(retriever.name, index, file_names)
                logger.info("Inserting data to new basic index")
                logger.info(f"Index: {index_store.list_indexes()}")
                agent.recreate_agent()
                index_store.save_to_file()

        except ValueError as e:
            logger.error(f"Value error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except IOError as e:
            logger.error(f"I/O error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            await file.close()
            logger.info(f"Closed file {file.filename}")

    return saved_files


def setup_files_routes(router: APIRouter, id: str, sdk_context: SDKContext):
    @router.post("/uploadfiles/")
    async def create_upload_files(files: List[UploadFile] = File(...)):

        saved_files = await insert_files_to_index(files, id, sdk_context)

        logger.info(f"Uploaded files: {saved_files}")
        return {"filenames": saved_files}

    @router.get("/files/")
    async def list_files():
        try:
            files = file_store.list_files()
            logger.info("Listed files")
            return {"files": files}
        except IOError as e:
            logger.error(f"I/O error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/files/{filename}")
    async def delete_file(filename: str):
        try:
            if file_store.delete_file(filename):
                logger.info(f"Deleted file {filename}")
                return {"message": f"File {filename} deleted successfully."}
            else:
                logger.warning(f"File {filename} not found for deletion")
                raise HTTPException(status_code=404, detail=f"File {filename} not found.")
        except ValueError as e:
            logger.error(f"Value error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except IOError as e:
            logger.error(f"I/O error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.put("/files/{old_filename}/{new_filename}")
    async def rename_file(old_filename: str, new_filename: str):
        try:
            if file_store.rename_file(old_filename, new_filename):
                logger.info(f"Renamed file from {old_filename} to {new_filename}")
                return {"message": f"File {old_filename} renamed to {new_filename} successfully."}
            else:
                logger.warning(f"File {old_filename} not found for renaming")
                raise HTTPException(status_code=404, detail=f"File {old_filename} not found.")
        except ValueError as e:
            logger.error(f"Value error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except IOError as e:
            logger.error(f"I/O error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
