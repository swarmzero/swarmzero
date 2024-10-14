import logging
import os
import shutil

from fastapi import UploadFile

BASE_DIR = "swarmzero-data/files/user"

# TODO: get log level from config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileStore:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"Initialized FileStore with base directory: {self.base_dir}")

    async def save_file(self, file: UploadFile):
        filename = os.path.basename(str(file.filename))
        if not filename:
            logger.error("Attempted to save a file with an empty name.")
            raise ValueError("Filename cannot be empty.")

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
