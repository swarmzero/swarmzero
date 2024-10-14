import os

from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.readers.s3 import S3Reader  # Add S3Reader import
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, PodSpec, ServerlessSpec

from swarmzero.tools.retriever.base_retrieve import RetrieverBase

load_dotenv()


class PineconeRetriever(RetrieverBase):

    def __init__(self):
        PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
        super().__init__(
            name="PineconeRetriever",
            description="This tool creates a Pinecone Retriever index",
        )
        self.pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

    def _load_documents_from_s3(self, bucket, prefix):
        s3_reader = S3Reader(
            bucket=bucket,
            prefix=prefix,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
        documents = s3_reader.load_data()
        file_names = [doc.metadata['file_name'] for doc in documents]
        return documents, file_names

    def create_serverless_index(
        self,
        file_path=None,
        folder_path=None,
        prefix='',
        bucket=None,
        collection_name="swarmzero-pinecone",
        dimension=1536,
        metric="euclidean",
        cloud="aws",
        region="us-east-1",
    ):
        if bucket:
            documents, file_names = self._load_documents_from_s3(bucket, prefix)
        else:
            documents, file_names = self._load_documents(file_path, folder_path)
        self.pinecone_client.create_index(
            name=collection_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        pinecone_index = self.pinecone_client.Index(collection_name)

        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        return index, file_names

    def create_pod_index(
        self,
        file_path=None,
        folder_path=None,
        s3_endpoint_url=None,
        bucket=None,
        collection_name="swarmzero-pinecone-pod",
        dimension=1536,
        metric="cosine",
        environment="us-east1-gcp",
        pod_type="p1.x1",
        pods=1,
    ):
        if bucket:
            documents, file_names = self._load_documents_from_s3(bucket, s3_endpoint_url)
        else:
            documents, file_names = self._load_documents(file_path, folder_path)
        self.pinecone_client.create_index(
            name=collection_name,
            dimension=dimension,
            metric=metric,
            spec=PodSpec(environment=environment, pod_type=pod_type, pods=pods),
        )
        pinecone_index = self.pinecone_client.Index(collection_name)

        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        return index, file_names

    def delete_index(self, index_name):
        self.pinecone_client.delete_index(index_name)
