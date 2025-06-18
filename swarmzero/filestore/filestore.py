import base64
import logging
import os
import shutil
from typing import Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv
from fastapi import UploadFile

load_dotenv()

BASE_DIR = "swarmzero-data/files/user"
USE_S3 = os.getenv("USE_S3", "false").lower() == "true"
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")
S3_PREFIX = "agent_uploads/"
S3_PRESIGNED_URL_EXPIRATION = 300

# TODO: get log level from config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileStore:
    def __init__(self, base_dir: str):
        self.use_s3 = USE_S3
        self.base_dir = base_dir

        if self.use_s3:
            if not S3_BUCKET_NAME:
                logger.error("S3_BUCKET_NAME environment variable is not set.")
                raise ValueError("S3_BUCKET_NAME environment variable is required when USE_S3 is true.")
            self.s3_client = boto3.client("s3", region_name=AWS_REGION)
            logger.info(f"Initialized FileStore with S3 bucket: {S3_BUCKET_NAME}/{S3_PREFIX}")
        else:
            os.makedirs(self.base_dir, exist_ok=True)
            logger.info(f"Initialized FileStore with base directory: {self.base_dir}")

    def _get_s3_key(self, filename: str) -> str:
        return f"{S3_PREFIX}{filename}"

    async def save_file(self, file: UploadFile):
        filename = os.path.basename(str(file.filename))
        if not filename:
            logger.error("Attempted to save a file with an empty name.")
            raise ValueError("Filename cannot be empty.")

        if self.use_s3:
            s3_key = self._get_s3_key(filename)
            try:
                self.s3_client.upload_fileobj(file.file, S3_BUCKET_NAME, s3_key)
                logger.info(f"Uploaded file: {filename} to S3 bucket: {S3_BUCKET_NAME}/{S3_PREFIX}")
            except NoCredentialsError:
                logger.error("AWS credentials not available.")
                raise
            except ClientError as e:
                logger.error(f"Failed to upload file to S3: {e}")
                raise IOError(f"Error uploading file {filename} to S3")
        else:
            file_location = os.path.join(self.base_dir, filename)
            try:
                with open(file_location, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                logger.info(f"Saved file: {filename} at {file_location}")
            except Exception as e:
                logger.error(f"Failed to save file {filename}: {e}")
                raise IOError(f"Error saving file {filename}")

        return filename

    def delete_file(self, filename: str):
        if not filename:
            logger.error("Attempted to delete a file with an empty name.")
            raise ValueError("Filename cannot be empty.")

        if self.use_s3:
            s3_key = self._get_s3_key(filename)
            try:
                self.s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                logger.info(f"Deleted file: {filename} from S3 bucket: {S3_BUCKET_NAME}/{S3_PREFIX}")
                return True
            except ClientError as e:
                logger.error(f"Failed to delete file from S3: {e}")
                raise IOError(f"Error deleting file {filename} from S3")
        else:
            file_location = os.path.join(self.base_dir, filename)
            if os.path.exists(file_location):
                try:
                    os.remove(file_location)
                    logger.info(f"Deleted file: {filename}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to delete file {filename}: {e}")
                    raise IOError(f"Error deleting file {filename}")
            else:
                logger.warning(f"Attempted to delete non-existent file: {filename}")
                return False

    def list_files(self):
        if self.use_s3:
            try:
                paginator = self.s3_client.get_paginator("list_objects_v2")
                pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX)

                files = []
                for page in pages:
                    if "Contents" in page:
                        for obj in page["Contents"]:
                            key = obj["Key"]
                            if key.endswith("/"):  # Skip directories
                                continue
                            # Remove prefix before adding to the list
                            filename = key[len(S3_PREFIX) :] if key.startswith(S3_PREFIX) else key
                            files.append(filename)

                logger.info(f"Listed files from S3 bucket under {S3_PREFIX}: {files}")
                return files
            except ClientError as e:
                logger.error(f"Failed to list files from S3: {e}")
                raise IOError("Error listing files from S3")
        else:
            try:
                files = os.listdir(self.base_dir)
                logger.info(f"Listed files: {files}")
                return files
            except Exception as e:
                logger.error(f"Failed to list files: {e}")
                raise IOError("Error listing files")

    def rename_file(self, old_filename: str, new_filename: str):
        if not old_filename or not new_filename:
            logger.error("Attempted to rename with an empty name.")
            raise ValueError("Filenames cannot be empty.")

        if self.use_s3:
            old_s3_key = self._get_s3_key(old_filename)
            new_s3_key = self._get_s3_key(new_filename)
            copy_source = {"Bucket": S3_BUCKET_NAME, "Key": old_s3_key}
            try:
                # Copy the object to the new key
                self.s3_client.copy(copy_source, S3_BUCKET_NAME, new_s3_key)
                # Delete the old object
                self.s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=old_s3_key)
                logger.info(f"Renamed {old_filename} to {new_filename} in s3: {S3_BUCKET_NAME}/{S3_PREFIX}")
                return True
            except ClientError as e:
                logger.error(f"Failed to rename file in S3: {e}")
                raise IOError(f"Error renaming file {old_filename} to {new_filename} in S3")
        else:
            old_file_location = os.path.join(self.base_dir, old_filename)
            new_file_location = os.path.join(self.base_dir, new_filename)

            if os.path.exists(old_file_location):
                try:
                    os.rename(old_file_location, new_file_location)
                    logger.info(f"Renamed file from {old_filename} to {new_filename}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to rename file from {old_filename} to {new_filename}: {e}")
                    raise IOError(f"Error renaming file {old_filename}")
            else:
                logger.warning(f"Attempted to rename non-existent file: {old_filename}")
                return False

    def get_file(self, filename: str) -> Optional[str]:
        if not filename:
            logger.error("Attempted to get a file with an empty name.")
            raise ValueError("Filename cannot be empty.")

        if self.use_s3:
            s3_key = self._get_s3_key(filename)
            try:
                response = self.s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                file_content = response['Body'].read()
                encoded_content = base64.b64encode(file_content).decode('utf-8')
                logger.info(f"Retrieved and encoded file: {filename} from S3 bucket: {S3_BUCKET_NAME}")
                return encoded_content
            except self.s3_client.exceptions.NoSuchKey:
                logger.warning(f"File {filename} does not exist in S3.")
                return None
            except ClientError as e:
                logger.error(f"Failed to retrieve file from S3: {e}")
                raise IOError(f"Error retrieving file {filename} from S3")
        else:
            # When storing files locally, ensure we look in the configured
            # base directory.  Previously this function used
            # ``os.path.join(filename)`` which simply returned ``filename``
            # without the base directory and caused lookups to fail.
            file_location = os.path.join(self.base_dir, filename)
            if not os.path.exists(file_location):
                logger.warning(f"File {filename} does not exist locally.")
                return None
            try:
                with open(file_location, "rb") as f:
                    file_content = f.read()
                    encoded_content = base64.b64encode(file_content).decode('utf-8')
                    logger.info(f"Retrieved and encoded file: {filename} from local storage.")
                    return encoded_content
            except Exception as e:
                logger.error(f"Failed to read file {filename}: {e}")
                raise IOError(f"Error retrieving file {filename}")
