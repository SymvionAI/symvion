"""
Document storage utility for downloading and managing documents locally.
Handles downloading documents from URLs and storing them in a local folder.
"""

import logging
import os
import httpx
import pathlib
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentStorage:
    """Manages local storage of documents for processing."""

    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize document storage.

        Args:
            storage_dir: Directory to store documents. Defaults to ai-runtime/documents
        """
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            # Default to ai-runtime/documents
            base_dir = Path(__file__).parent.parent.parent.parent
            self.storage_dir = base_dir / "documents"
        
        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Document storage directory: {self.storage_dir.absolute()}")

    async def download_document(
        self, document_url: str, document_id: str, file_extension: Optional[str] = None
    ) -> Optional[str]:
        """
        Download a document from a URL and save it locally.

        Args:
            document_url: URL to download the document from
            document_id: Unique identifier for the document
            file_extension: Optional file extension (e.g., '.pdf', '.jpg')

        Returns:
            Local file path if successful, None otherwise
        """
        try:
            # Determine file extension if not provided
            if not file_extension:
                # Try to get extension from URL
                url_path = pathlib.PurePath(document_url)
                file_extension = url_path.suffix
                if not file_extension:
                    # Default to no extension if we can't determine
                    file_extension = ""

            # Create unique filename
            filename = f"{document_id}{file_extension}"
            file_path = self.storage_dir / filename

            # Download the file
            logger.info(f"Attempting to download from URL: {document_url}")
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                response = await client.get(document_url)
                response.raise_for_status()
                
                logger.info(
                    f"Download response status: {response.status_code}, content-type: {response.headers.get('content-type', 'unknown')}, size: {len(response.content)} bytes"
                )

                # Save to disk
                with open(file_path, "wb") as f:
                    f.write(response.content)

            logger.info(
                f"Downloaded document {document_id} to {file_path.absolute()}"
            )
            return str(file_path.absolute())

        except Exception as e:
            logger.error(
                f"Failed to download document {document_id} from {document_url}: {e}",
                exc_info=True
            )
            return None

    def get_document_path(self, document_id: str, file_extension: Optional[str] = None) -> str:
        """
        Get the expected file path for a document.

        Args:
            document_id: Unique identifier for the document
            file_extension: Optional file extension

        Returns:
            Expected file path
        """
        if file_extension:
            filename = f"{document_id}{file_extension}"
        else:
            filename = document_id
        return str(self.storage_dir / filename)

    def delete_document(self, document_id: str, file_extension: Optional[str] = None) -> bool:
        """
        Delete a document from local storage.

        Args:
            document_id: Unique identifier for the document
            file_extension: Optional file extension

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            file_path = Path(self.get_document_path(document_id, file_extension))
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted document {document_id} from local storage")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to delete document {document_id}: {e}")
            return False

    def cleanup_old_documents(self, max_age_hours: int = 24) -> int:
        """
        Clean up documents older than specified hours.

        Args:
            max_age_hours: Maximum age in hours before deletion

        Returns:
            Number of documents deleted
        """
        import time
        deleted_count = 0
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        try:
            for file_path in self.storage_dir.iterdir():
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"Cleaned up old document: {file_path.name}")

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old document(s)")

        except Exception as e:
            logger.error(f"Error during document cleanup: {e}")

        return deleted_count


# Global document storage instance
_document_storage: Optional[DocumentStorage] = None


def get_document_storage() -> DocumentStorage:
    """Get or create the global document storage instance."""
    global _document_storage
    if _document_storage is None:
        storage_dir = os.getenv("DOCUMENT_STORAGE_DIR")
        _document_storage = DocumentStorage(storage_dir=storage_dir)
    return _document_storage
