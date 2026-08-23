"""
OCR Service Module
Provides a modular interface for OCR (Optical Character Recognition) operations.
Currently implements pytesseract, but designed to be easily swappable.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class OCRService(ABC):
    """Abstract base class for OCR services."""

    @abstractmethod
    async def extract_text(
        self, file_path: str, mime_type: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Extract text from a document or image.

        Args:
            file_path: Path to the file to process
            mime_type: MIME type of the file (e.g., 'application/pdf', 'image/png')
            **kwargs: Additional parameters specific to the OCR implementation

        Returns:
            Dictionary containing:
                - text: Extracted text content
                - confidence: Optional confidence score
                - metadata: Optional additional metadata
        """
        pass


class TesseractOCRService(OCRService):
    """OCR service implementation using pytesseract."""

    def __init__(self):
        """Initialize Tesseract OCR service."""
        try:
            import pytesseract
            from PIL import Image
            import pdf2image

            self.pytesseract = pytesseract
            self.Image = Image
            self.pdf2image = pdf2image
            self._available = True
            logger.info("TesseractOCRService initialized successfully")
        except ImportError as e:
            self._available = False
            logger.error(f"Failed to import OCR dependencies: {e}")
            logger.error("Please install: pip install pytesseract pillow pdf2image")

    @property
    def is_available(self) -> bool:
        """Check if OCR service is available."""
        return self._available

    async def extract_text(
        self, file_path: str, mime_type: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Extract text using pytesseract.

        Args:
            file_path: Path to the file to process
            mime_type: MIME type of the file
            **kwargs: Additional parameters (e.g., lang for language)

        Returns:
            Dictionary with extracted text and metadata
        """
        if not self._available:
            raise RuntimeError(
                "TesseractOCRService is not available. Please install required dependencies."
            )

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Determine file type and process accordingly
            if mime_type:
                if mime_type.startswith("image/"):
                    text = await self._extract_from_image(file_path, **kwargs)
                elif mime_type == "application/pdf":
                    text = await self._extract_from_pdf(file_path, **kwargs)
                else:
                    # Try to infer from extension
                    text = await self._extract_by_extension(file_path, **kwargs)
            else:
                # Infer from file extension
                text = await self._extract_by_extension(file_path, **kwargs)

            return {
                "text": text,
                "confidence": None,  # pytesseract doesn't provide confidence by default
                "metadata": {
                    "file_path": file_path,
                    "mime_type": mime_type,
                    "method": "pytesseract",
                },
            }
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}", exc_info=True)
            raise

    async def _extract_from_image(self, file_path: str, **kwargs: Any) -> str:
        """Extract text from an image file."""
        lang = kwargs.get("lang", "eng")  # Default to English

        image = self.Image.open(file_path)
        text = self.pytesseract.image_to_string(image, lang=lang)
        return text.strip()

    async def _extract_from_pdf(self, file_path: str, **kwargs: Any) -> str:
        """Extract text from a PDF file."""
        lang = kwargs.get("lang", "eng")

        # Convert PDF pages to images
        images = self.pdf2image.convert_from_path(file_path)

        # Extract text from each page
        texts = []
        for i, image in enumerate(images):
            logger.info(f"Processing PDF page {i + 1}/{len(images)}")
            page_text = self.pytesseract.image_to_string(image, lang=lang)
            if page_text.strip():
                texts.append(f"--- Page {i + 1} ---\n{page_text.strip()}")

        return "\n\n".join(texts)

    async def _extract_by_extension(self, file_path: str, **kwargs: Any) -> str:
        """Extract text by inferring file type from extension."""
        file_path_obj = Path(file_path)
        extension = file_path_obj.suffix.lower()

        if extension in [".pdf"]:
            return await self._extract_from_pdf(file_path, **kwargs)
        elif extension in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]:
            return await self._extract_from_image(file_path, **kwargs)
        else:
            # Try as image first
            try:
                return await self._extract_from_image(file_path, **kwargs)
            except Exception:
                raise ValueError(
                    f"Unsupported file type: {extension}. "
                    "Supported types: PDF, PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP"
                )


def get_ocr_service(service_type: str = "tesseract") -> OCRService:
    """
    Factory function to get an OCR service instance.

    Args:
        service_type: Type of OCR service ('tesseract' or future implementations)

    Returns:
        OCRService instance

    Raises:
        ValueError: If service_type is not supported
    """
    if service_type == "tesseract":
        return TesseractOCRService()
    else:
        raise ValueError(
            f"Unsupported OCR service type: {service_type}. "
            "Supported types: 'tesseract'"
        )
