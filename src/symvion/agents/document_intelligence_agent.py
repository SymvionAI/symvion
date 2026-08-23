"""
Document Intelligence Agent
Handles document processing, extraction, and analysis using OCR services.
"""

import logging
import os
import pathlib
from typing import Dict, Any, List, Optional
from symvion.providers.factory import ProviderFactory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from symvion.utils.document_storage import get_document_storage
from symvion.utils.ocr_service import get_ocr_service

logger = logging.getLogger(__name__)


class DocumentIntelligenceAgent:
    """Agent specialized in document processing and intelligence."""

    def __init__(self, tenant_id: str, config: Dict[str, Any] = None):
        """
        Initialize the Document Intelligence Agent.

        Args:
            tenant_id: Unique tenant identifier
            config: Agent-specific configuration
        """
        self.tenant_id = tenant_id
        self.config = config or {}

        # Initialize LLM
        self.llm = ProviderFactory.get_provider(
            self.config.get("provider", "openai"),
            self.config
        )

        # Initialize OCR service (modular, can be swapped for different implementations)
        ocr_service_type = self.config.get("ocr_service", "tesseract")
        self.ocr_service = get_ocr_service(ocr_service_type)

        # Initialize document storage
        self.document_storage = get_document_storage()

    async def process(
        self,
        user_message: str,
        conversation_history: List[Dict[str, Any]],
        documents: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a document-related query.

        Args:
            user_message: The user's message
            conversation_history: Previous conversation messages
            documents: List of document metadata (id, fileName, mimeType, etc.)

        Returns:
            Dict containing agent_response and agent_type
        """
        logger.info(
            f"[DOCUMENT_INTELLIGENCE] process() called with user_message: '{user_message}', documents: {documents}, documents_count: {len(documents) if documents else 0}"
        )

        # Build system prompt
        system_prompt = """You are a Document Intelligence Agent specialized in processing and analyzing documents.

Your capabilities include:
- Extracting text from PDFs and images using OCR
- Analyzing document content
- Answering questions about document contents
- Summarizing documents
- Identifying key information in documents

When documents are provided:
1. The text has already been extracted from the documents using OCR
2. You will receive the extracted text in the conversation context
3. Analyze the extracted content and provide meaningful insights
4. If the user just uploaded a document, provide a helpful summary or analysis
5. Answer any questions the user has about the document content

IMPORTANT: 
- Always provide a response after processing documents - don't just extract and stop
- Give users useful information about what was found in their document
- Be concise but informative
- If the extracted text is empty or minimal, let the user know and suggest next steps

Be helpful, accurate, and focus on document-related tasks."""

        # Build conversation history
        messages = [SystemMessage(content=system_prompt)]

        # Add conversation history
        for msg in conversation_history:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg.get("content", "")))

        # Add current user message with document context
        user_content = user_message
        if documents:
            doc_info = "\n\nAvailable documents:\n"
            for doc in documents:
                doc_info += f"- {doc.get('fileName', 'Unknown')} ({doc.get('mimeType', 'Unknown type')}) [ID: {doc.get('id', 'N/A')}]\n"
            user_content = user_message + doc_info

        messages.append(HumanMessage(content=user_content))

        try:
            # Always download documents if they exist, even if extraction keywords aren't present
            # The user might want to process them later or the agent should proactively analyze them
            extracted_texts = {}
            local_file_paths = []  # Track downloaded files for cleanup

            # Check if user wants to extract text from documents
            extraction_keywords = [
                "extract",
                "read",
                "text from",
                "ocr",
                "content",
                "what does",
                "what is in",
                "analyze",
                "summarize",
                "what",
                "tell me",
                "say",  # Added for "What does this document say?"
            ]
            needs_extraction = any(
                keyword in user_message.lower() for keyword in extraction_keywords
            )

            # If documents are present and message is just a file attachment placeholder,
            # automatically trigger extraction (user likely wants document analyzed)
            if documents and not needs_extraction:
                attachment_patterns = [
                    "[attached",
                    "attached",
                    "file(s)]",
                    "uploaded",
                ]
                is_attachment_message = any(
                    pattern in user_message.lower() for pattern in attachment_patterns
                )
                if is_attachment_message:
                    needs_extraction = True
                    logger.info(
                        f"Auto-triggering extraction for file attachment message: '{user_message}'"
                    )

            logger.info(
                f"Document extraction needed: {needs_extraction} (message: '{user_message}')"
            )

            # If documents are present, ALWAYS download them (even without explicit extraction request)
            # This ensures documents are available for processing
            if documents:
                logger.info(
                    f"Processing {len(documents)} document(s) for tenant {self.tenant_id}"
                )
                logger.info(
                    f"Document details: {[{'id': d.get('id'), 'downloadUrl': d.get('downloadUrl'), 'cloudinaryUrl': d.get('cloudinaryUrl'), 'mimeType': d.get('mimeType')} for d in documents]}"
                )

                # Download all documents first
                for doc in documents:
                    doc_id = doc.get("id")
                    mime_type = doc.get("mimeType", "")
                    file_name = doc.get("fileName", "")

                    if doc_id:
                        try:
                            # Get document URL - prefer cloudinaryUrl, then downloadUrl, then fallback to backend URL
                            backend_url = os.getenv(
                                "BACKEND_URL", "http://localhost:3000"
                            )

                            # Check for cloudinaryUrl first, then downloadUrl
                            document_url = (
                                doc.get("cloudinaryUrl")
                                or doc.get("downloadUrl")
                                or f"{backend_url}/api/v1/documents/{doc_id}/download"
                            )

                            logger.info(
                                f"Document {doc_id} - Using URL: {document_url} (cloudinaryUrl: {doc.get('cloudinaryUrl')}, downloadUrl: {doc.get('downloadUrl')})"
                            )

                            # Determine file extension from mime type or file name
                            file_extension = None
                            if mime_type:
                                # Map common mime types to extensions
                                mime_to_ext = {
                                    "application/pdf": ".pdf",
                                    "image/jpeg": ".jpg",
                                    "image/jpg": ".jpg",
                                    "image/png": ".png",
                                    "image/gif": ".gif",
                                    "image/webp": ".webp",
                                }
                                file_extension = mime_to_ext.get(mime_type.lower())

                            # If no extension from mime type, try to get from file name
                            if not file_extension and file_name:
                                file_path = pathlib.PurePath(file_name)
                                file_extension = file_path.suffix

                            # Download document to local storage
                            logger.info(
                                f"Downloading document {doc_id} from {document_url} (mimeType: {mime_type}, extension: {file_extension})"
                            )
                            local_file_path = (
                                await self.document_storage.download_document(
                                    document_url, doc_id, file_extension
                                )
                            )

                            if not local_file_path:
                                error_msg = f"Failed to download document {doc_id} from {document_url}"
                                logger.error(error_msg)
                                raise Exception(error_msg)

                            local_file_paths.append(local_file_path)
                            logger.info(
                                f"Successfully downloaded document {doc_id} to {local_file_path}"
                            )

                            # Only call OCR service for extraction if user explicitly requested it
                            # or if the message suggests they want content analysis
                            if needs_extraction:
                                # Call OCR service with local file path
                                logger.info(
                                    f"Extracting text from document {doc_id} using OCR service (file: {local_file_path})"
                                )
                                try:
                                    ocr_result = await self.ocr_service.extract_text(
                                        local_file_path,
                                        mime_type=mime_type,
                                    )
                                    extracted_text = ocr_result.get("text", "")
                                    extracted_texts[doc_id] = (
                                        extracted_text
                                        if extracted_text
                                        else "Text extraction completed (no text found)"
                                    )
                                    logger.info(
                                        f"Successfully extracted {len(extracted_text)} characters from document {doc_id}"
                                    )
                                except Exception as ocr_error:
                                    logger.error(
                                        f"OCR extraction failed for document {doc_id}: {ocr_error}",
                                        exc_info=True,
                                    )
                                    extracted_texts[doc_id] = (
                                        f"Error during OCR extraction: {str(ocr_error)}"
                                    )
                            else:
                                # Document downloaded but not extracted yet
                                # Inform user that document is ready for processing
                                extracted_texts[doc_id] = (
                                    f"Document '{file_name}' has been downloaded and is ready for processing. You can ask me to extract text, analyze, or summarize it."
                                )
                        except Exception as e:
                            logger.warning(
                                f"Failed to extract text from document {doc_id}: {e}",
                                exc_info=True,
                            )
                            extracted_texts[doc_id] = f"Error extracting text: {str(e)}"
                        finally:
                            # Cleanup: Optionally delete downloaded files after processing
                            # For now, we keep them for potential reuse, but they'll be cleaned up by periodic cleanup
                            pass

                # Add extracted text to context
                if extracted_texts:
                    extraction_context = "\n\n=== EXTRACTED TEXT FROM DOCUMENTS ===\n"
                    for doc_id, text in extracted_texts.items():
                        doc_name = next(
                            (
                                d.get("fileName", "Unknown")
                                for d in documents
                                if d.get("id") == doc_id
                            ),
                            "Unknown",
                        )
                        # Include more text for better analysis (first 5000 chars per document)
                        # The LLM can handle this and provide better analysis
                        text_preview = text[:5000] if len(text) > 5000 else text
                        extraction_context += (
                            f"\n--- Document: {doc_name} ---\n{text_preview}\n"
                        )
                        if len(text) > 5000:
                            extraction_context += (
                                f"[... {len(text) - 5000} more characters ...]\n"
                            )
                    extraction_context += "\n=== END OF EXTRACTED TEXT ===\n\n"
                    extraction_context += "Please analyze the above extracted text and provide a helpful response to the user. "
                    extraction_context += "If the user just uploaded the document, provide a summary or key insights. "
                    extraction_context += "If they asked a question, answer it based on the extracted content."

                    user_content = user_content + extraction_context
                    # Update the last message with extraction context
                    messages[-1] = HumanMessage(content=user_content)

            # Invoke LLM for main analysis/summary
            response = self.llm.invoke(messages)
            response_content = response.content

            # Classify document type to suggest downstream agent (e.g. claims, billing)
            suggested_next_agent = "none"
            if extracted_texts and any(
                t
                and t != "Text extraction completed (no text found)"
                and not t.startswith("Error")
                for t in extracted_texts.values()
            ):
                classification_prompt = (
                    "Given the following document summary or content preview, "
                    "which single domain does this document most clearly belong to? "
                    "Consider: claims forms, claim details, policy numbers, incident reports → claims. "
                    "Invoices, payments, billing statements → billing. "
                    "Insurance policies, coverage documents → insurance. "
                    "Registration or onboarding forms → registration. "
                    "Complaint or grievance documents → complaint. "
                    "If the document does not clearly fit any domain, respond with none.\n\n"
                    "Document summary/content:\n"
                    f"{response_content[:2000]}\n\n"
                    "Respond with ONLY one word: claims, billing, insurance, registration, complaint, or none."
                )
                try:
                    classification_msg = [HumanMessage(content=classification_prompt)]
                    classification_response = self.llm.invoke(classification_msg)
                    raw = (classification_response.content or "").strip().lower()
                    valid = {
                        "claims",
                        "billing",
                        "insurance",
                        "registration",
                        "complaint",
                        "none",
                    }
                    suggested_next_agent = raw if raw in valid else "none"
                    logger.info(
                        f"[DOCUMENT_INTELLIGENCE] Classified document as: {suggested_next_agent}"
                    )
                except Exception as class_err:
                    logger.warning(
                        f"[DOCUMENT_INTELLIGENCE] Classification failed: {class_err}"
                    )

            return {
                "agent_response": response_content,
                "agent_type": "document_intelligence",
                "suggested_next_agent": suggested_next_agent,
                "document_analysis": response_content,
            }
        except Exception as e:
            logger.error(
                f"Error in DocumentIntelligenceAgent for tenant {self.tenant_id}: {e}",
                exc_info=True,
            )
            import traceback

            logger.error(
                f"Full traceback in DocumentIntelligenceAgent: {traceback.format_exc()}"
            )
            return {
                "agent_response": "I apologize, but I encountered an error processing your document request. Please try again.",
                "agent_type": "document_intelligence",
                "suggested_next_agent": "none",
                "document_analysis": "",
            }
