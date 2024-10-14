import logging
from typing import List

from fastapi import APIRouter, HTTPException

from swarmzero.tools.retriever.base_retrieve import IndexStore, RetrieverBase
from swarmzero.tools.retriever.chroma_retrieve import ChromaRetriever
from swarmzero.tools.retriever.pinecone_retrieve import PineconeRetriever

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the router and index store
index_store = IndexStore.get_instance()


def setup_vectorindex_routes(router: APIRouter):
    @router.post("/create_index/")
    async def create_index(
        index_name: str,
        file_path: List[str] = None,
        folder_path: str = None,
        index_type=None,
    ):
        try:
            if index_type is None or index_type == "basic":
                retriever = RetrieverBase()
                index, file_names = retriever.create_basic_index(file_path, folder_path)
            elif index_type == "chroma":
                retriever = ChromaRetriever()
                index, file_names = retriever.create_index(file_path, folder_path, collection_name=index_name)
            elif index_type == "pinecone-serverless":
                retriever = PineconeRetriever()
                index, file_names = retriever.create_serverless_index(file_path, folder_path, name=index_name)
            elif index_type == "pinecone-pod":
                retriever = PineconeRetriever()
                index, file_names = retriever.create_pod_index(file_path, folder_path, name=index_name)
            else:
                raise HTTPException(status_code=400, detail="Invalid index type provided.")

            index_store.add_index(index_name, index, file_names)
            return {
                "message": f"Index {index_name} created successfully.",
                "index_details": index,
                "file_names": file_names,
            }
        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/insert_documents/")
    async def insert_documents(index_name: str, file_path: List[str] = None, folder_path: str = None):
        try:
            index = index_store.get_index(index_name)
            result, file_names = RetrieverBase().insert_documents(index, file_path, folder_path)
            index_store.update_index(index_name, index)
            index_store.insert_index_files(index_name, file_names)
            return {"message": "Documents inserted successfully.", "details": result, "file_names": file_names}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Index {index_name} not found.")
        except Exception as e:
            logger.error(f"Error inserting documents: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.put("/update_documents/")
    async def update_documents(index_name: str, file_path: List[str] = None, folder_path: str = None):
        try:
            index = index_store.get_index(index_name)
            result, file_names = RetrieverBase().update_documents(index, file_path, folder_path)
            index_store.update_index(index_name, index)
            index_store.update_index_files(index_name, file_names)
            return {"message": "Documents updated successfully.", "details": result, "file_names": file_names}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Index {index_name} not found.")
        except Exception as e:
            logger.error(f"Error updating documents: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/delete_documents/")
    async def delete_documents(index_name: str, file_path: List[str] = None, folder_path: str = None):
        try:
            index = index_store.get_index(index_name)
            result, file_names = RetrieverBase().delete_documents(index, file_path, folder_path)
            index_store.delete_index(index_name)
            index_store.delete_index_files(index_name)
            return {"message": "Documents deleted successfully.", "details": result, "file_names": file_names}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Index {index_name} not found.")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise HTTPException(status_code=500, detail=str(e))
