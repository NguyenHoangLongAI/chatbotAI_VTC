# Embedding_vectorDB/docling_processor.py
"""
Docling-based Document Processor
High-quality document to Markdown conversion
"""

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from pathlib import Path
from typing import Optional
import logging
import os

logger = logging.getLogger(__name__)


class DoclingProcessor:
    """
    Advanced document processor using Docling
    """

    def __init__(self, use_ocr: bool = True, ocr_lang: list = None):
        """
        Initialize Docling processor

        Args:
            use_ocr: Enable OCR for scanned documents
            ocr_lang: OCR languages (default: ['vi', 'en'])
        """
        self.use_ocr = use_ocr
        self.ocr_lang = ocr_lang or ['vi', 'en']

        # Configure pipeline options
        pipeline_options = PdfPipelineOptions()

        if self.use_ocr:
            # Enable EasyOCR for better Vietnamese support
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options = EasyOcrOptions(
                lang=self.ocr_lang,
                use_gpu=True  # Use GPU if available
            )

        # Configure format options
        self.format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }

        # Initialize converter
        self.converter = DocumentConverter(
            format_options=self.format_options
        )

        logger.info(
            f"✅ Docling processor initialized (OCR={'enabled' if use_ocr else 'disabled'})"
        )

    def process_document(self, file_path: str) -> Optional[str]:
        """
        Process document to Markdown

        Args:
            file_path: Path to document file

        Returns:
            Markdown content or None if failed
        """
        try:
            logger.info(f"📄 Processing with Docling: {os.path.basename(file_path)}")

            # Convert document
            result = self.converter.convert(file_path)

            # Export to Markdown
            markdown_content = result.document.export_to_markdown()

            # Clean up markdown
            markdown_content = self._clean_markdown(markdown_content)

            logger.info(
                f"✅ Docling conversion successful: {len(markdown_content)} chars"
            )

            return markdown_content

        except Exception as e:
            logger.error(f"❌ Docling processing failed: {e}", exc_info=True)
            return None

    def process_pdf(self, file_path: str) -> Optional[str]:
        """
        Process PDF file specifically

        Args:
            file_path: Path to PDF file

        Returns:
            Markdown content or None if failed
        """
        return self.process_document(file_path)

    def process_docx(self, file_path: str) -> Optional[str]:
        """
        Process DOCX file

        Args:
            file_path: Path to DOCX file

        Returns:
            Markdown content or None if failed
        """
        return self.process_document(file_path)

    def process_image(self, file_path: str) -> Optional[str]:
        """
        Process image file with OCR

        Args:
            file_path: Path to image file

        Returns:
            Markdown content or None if failed
        """
        if not self.use_ocr:
            logger.warning("OCR is disabled, cannot process image")
            return None

        return self.process_document(file_path)

    def _clean_markdown(self, markdown: str) -> str:
        """
        Clean and normalize markdown content

        Args:
            markdown: Raw markdown from Docling

        Returns:
            Cleaned markdown
        """
        import re

        # Remove excessive blank lines
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)

        # Remove page numbers (common pattern)
        markdown = re.sub(r'^Trang \d+.*\n', '', markdown, flags=re.MULTILINE)
        markdown = re.sub(r'^Page \d+.*\n', '', markdown, flags=re.MULTILINE)

        # Normalize whitespace
        markdown = re.sub(r'[ \t]+', ' ', markdown)

        # Remove leading/trailing whitespace
        markdown = markdown.strip()

        return markdown

    def batch_process(self, file_paths: list) -> dict:
        """
        Process multiple documents

        Args:
            file_paths: List of file paths

        Returns:
            Dictionary mapping file_path -> markdown_content
        """
        results = {}

        for file_path in file_paths:
            try:
                markdown = self.process_document(file_path)
                results[file_path] = markdown
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results[file_path] = None

        return results


# Global processor instance (lazy loading)
_docling_processor = None


def get_docling_processor(use_ocr: bool = True) -> DoclingProcessor:
    """
    Get or create global Docling processor instance

    Args:
        use_ocr: Enable OCR

    Returns:
        DoclingProcessor instance
    """
    global _docling_processor

    if _docling_processor is None:
        _docling_processor = DoclingProcessor(use_ocr=use_ocr)

    return _docling_processor