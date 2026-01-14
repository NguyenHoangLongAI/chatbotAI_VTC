from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import uuid
import re
from typing import List, Dict, Any
import json

# Import processing modules
from document_processor import DocumentProcessor
from embedding_service import EmbeddingService
from milvus_client import MilvusManager

app = FastAPI(
    title="Document Processing API",
    version="2.0.0",  # Version bump for smart chunking
    description="API for document processing, embedding, and FAQ management with Smart Chunking"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
milvus_host = os.getenv('MILVUS_HOST', 'localhost')
milvus_port = os.getenv('MILVUS_PORT', '19530')
milvus_manager = MilvusManager(host=milvus_host, port=milvus_port)

doc_processor = DocumentProcessor(use_docling=True, use_ocr=True)
embedding_service = EmbeddingService()


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to be safe for file operations"""
    if not filename:
        return "unknown_file"

    name, ext = os.path.splitext(filename)
    safe_name = re.sub(r'[^\w\-_.]', '_', name)
    safe_name = re.sub(r'_+', '_', safe_name)
    safe_name = safe_name.strip('_')

    if not safe_name:
        safe_name = "document"

    return safe_name + ext.lower()


def get_safe_temp_filename(original_filename: str) -> str:
    """Generate a safe temporary filename with unique identifier"""
    _, ext = os.path.splitext(original_filename)
    unique_id = str(uuid.uuid4())[:8]
    safe_name = f"temp_doc_{unique_id}{ext.lower()}"
    return safe_name


@app.on_event("startup")
async def startup_event():
    """Initialize Milvus connection and create collections"""
    try:
        await milvus_manager.initialize()
        print("✅ Document API started successfully")
        print(f"✅ Milvus connected: {milvus_host}:{milvus_port}")
        print("✅ Smart Chunking enabled")
    except Exception as e:
        print(f"⚠️  Warning during startup: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Document Processing API",
        "version": "2.0.0",
        "status": "running",
        "features": {
            "smart_chunking": "enabled",
            "chunking_modes": ["smart", "sentence", "legacy"],
            "docling_integration": "enabled"
        },
        "endpoints": {
            "process_document": "/api/v1/process-document",
            "embed_markdown": "/api/v1/embed-markdown",
            "add_faq": "/api/v1/faq/add",
            "delete_faq": "/api/v1/faq/delete/{faq_id}",
            "delete_document": "/api/v1/document/delete/{document_id}",
            "health": "/api/v1/health"
        }
    }


@app.post("/api/v1/process-document")
async def process_document(file: UploadFile = File(...)):
    """
    API 1: Convert various document formats to structured markdown
    Accepts: PDF, DOC, DOCX, XLS, XLSX, TXT files
    Returns: Processed markdown content
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        allowed_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.txt']
        original_filename = file.filename
        file_extension = os.path.splitext(original_filename)[1].lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported. Allowed: {allowed_extensions}"
            )

        safe_temp_name = get_safe_temp_filename(original_filename)
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, safe_temp_name)

        try:
            content = await file.read()

            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")

            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(content)

            # Process document based on type
            if file_extension == '.pdf':
                markdown_content = doc_processor.process_pdf(temp_file_path)
            elif file_extension in ['.doc', '.docx']:
                markdown_content = doc_processor.process_word(temp_file_path)
            elif file_extension in ['.xls', '.xlsx']:
                markdown_content = doc_processor.process_excel(temp_file_path)
            elif file_extension == '.txt':
                text_content = None
                for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                    try:
                        with open(temp_file_path, 'r', encoding=encoding) as f:
                            text_content = f.read()
                        break
                    except UnicodeDecodeError:
                        continue

                if text_content is None:
                    raise HTTPException(status_code=400, detail="Could not decode text file")

                markdown_content = doc_processor.process_text(text_content)

            if not markdown_content or len(markdown_content.strip()) == 0:
                raise HTTPException(
                    status_code=422,
                    detail="Could not extract content from file"
                )

            safe_original_name = sanitize_filename(original_filename)

            return {
                "status": "success",
                "filename": safe_original_name,
                "original_filename": original_filename,
                "markdown_content": markdown_content,
                "processing_info": {
                    "file_type": file_extension,
                    "content_length": len(markdown_content),
                    "file_size_bytes": len(content),
                    "processor": "docling+smart_chunker"
                }
            }

        finally:
            try:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up temp file: {cleanup_error}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/api/v1/embed-markdown")
async def embed_markdown(request: dict):
    """
    API 2: Convert markdown content to embeddings and store in Milvus

    UPDATED: Với Smart Chunking Strategy

    Input: {
        "markdown_content": "...",
        "document_id": "...",
        "chunk_mode": "smart|sentence|legacy"
    }

    Chunk Modes:
    - "smart" (RECOMMENDED): Semantic chunking với context preservation
    - "sentence": Legacy sentence-based chunking (5 sentences/chunk)
    - "legacy": Backward compatible với code cũ

    Returns: List of embeddings with metadata
    """
    try:
        markdown_content = request.get("markdown_content")
        document_id = request.get("document_id")
        chunk_mode = request.get("chunk_mode", "smart").lower()  # Default to smart

        if not markdown_content:
            raise HTTPException(status_code=400, detail="markdown_content is required")

        # Validate chunk_mode
        valid_modes = ["smart", "sentence", "legacy"]
        if chunk_mode not in valid_modes:
            raise HTTPException(
                status_code=400,
                detail=f"chunk_mode must be one of: {valid_modes}"
            )

        # Sanitize document_id
        if document_id:
            document_id = re.sub(r'[^\w\-_.]', '_', str(document_id))
            document_id = re.sub(r'_+', '_', document_id).strip('_')

        if not document_id:
            document_id = f"doc_{str(uuid.uuid4())[:8]}"

        if len(markdown_content.strip()) == 0:
            raise HTTPException(status_code=400, detail="markdown_content cannot be empty")

        # ================================================================
        # CHUNKING STRATEGY SELECTION
        # ================================================================

        print(f"🔍 Using chunking mode: {chunk_mode}")

        if chunk_mode == "smart":
            # NEW: Smart semantic chunking
            chunks = doc_processor.parse_markdown_to_chunks(markdown_content)
            chunking_strategy = "semantic_smart_chunking"

        elif chunk_mode == "sentence":
            # LEGACY: Sentence-based chunking (kept for backward compatibility)
            # You need to add this method back to DocumentProcessor if needed
            if hasattr(doc_processor, 'parse_markdown_to_sentences'):
                chunks = doc_processor.parse_markdown_to_sentences(markdown_content)
                chunking_strategy = "sentence_based_legacy"
            else:
                # Fallback to smart if sentence method not available
                print("⚠️ sentence mode not available, falling back to smart")
                chunks = doc_processor.parse_markdown_to_chunks(markdown_content)
                chunking_strategy = "semantic_smart_chunking"

        else:  # legacy
            # LEGACY: Old paragraph-based
            if hasattr(doc_processor, 'parse_markdown_to_sentences'):
                chunks = doc_processor.parse_markdown_to_sentences(markdown_content)
                chunking_strategy = "legacy_paragraph"
            else:
                chunks = doc_processor.parse_markdown_to_chunks(markdown_content)
                chunking_strategy = "semantic_smart_chunking"

        if not chunks:
            raise HTTPException(status_code=422, detail="Could not parse markdown into chunks")

        print(f"✅ Created {len(chunks)} chunks using {chunking_strategy}")

        # ================================================================
        # EMBEDDING GENERATION
        # ================================================================

        embeddings_data = []
        successful_embeddings = 0
        failed_chunks = []

        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding
                embedding = embedding_service.get_embedding(chunk['content'])

                # Prepare metadata based on chunk structure
                metadata = {
                    "section_title": chunk.get('section_title', 'Unknown Section'),
                    "chunk_index": i,
                    "content_length": len(chunk['content']),
                    "chunk_mode": chunk_mode,
                    "chunking_strategy": chunking_strategy
                }

                # Add smart chunking specific metadata
                if chunk_mode == "smart":
                    metadata.update({
                        "token_count": chunk.get('token_count', 0),
                        "chunk_type": chunk.get('chunk_type', 'unknown'),
                        "context": chunk.get('context', ''),
                        "level": chunk.get('level', 0)
                    })

                    if 'context_path' in chunk:
                        metadata["context_path"] = " > ".join(chunk['context_path'])

                # Add sentence-specific metadata if available (legacy)
                elif chunk_mode == "sentence" and 'sentence' in chunk:
                    metadata.update({
                        "sentence": chunk.get('sentence', ''),
                        "sentence_length": len(chunk.get('sentence', ''))
                    })

                embedding_data = {
                    "id": f"{document_id}_{chunk_mode}_{i}",
                    "document_id": document_id,
                    "description": chunk['content'],
                    "description_vector": embedding,
                    "metadata": metadata
                }

                embeddings_data.append(embedding_data)
                successful_embeddings += 1

            except Exception as embedding_error:
                print(f"❌ Error creating embedding for chunk {i}: {embedding_error}")
                failed_chunks.append({
                    "chunk_index": i,
                    "error": str(embedding_error)
                })
                continue

        if not embeddings_data:
            raise HTTPException(
                status_code=422,
                detail="Could not create embeddings for any chunks"
            )

        # ================================================================
        # STORE IN MILVUS
        # ================================================================

        stored_count = await milvus_manager.insert_embeddings(embeddings_data)

        # ================================================================
        # PREPARE RESPONSE
        # ================================================================

        chunks_info = []
        for item in embeddings_data[:10]:  # Preview first 10
            chunk_info = {
                "id": item["id"],
                "section_title": item["metadata"]["section_title"],
                "content_preview": item["description"][:200] + "..."
                if len(item["description"]) > 200
                else item["description"]
            }

            # Add smart chunking info
            if chunk_mode == "smart":
                chunk_info.update({
                    "token_count": item["metadata"].get("token_count", 0),
                    "chunk_type": item["metadata"].get("chunk_type", "unknown"),
                    "context_path": item["metadata"].get("context_path", "")
                })

            # Add sentence info (legacy)
            elif chunk_mode == "sentence" and 'sentence' in item["metadata"]:
                chunk_info["sentence_preview"] = (
                    item["metadata"]["sentence"][:100] + "..."
                    if len(item["metadata"]["sentence"]) > 100
                    else item["metadata"]["sentence"]
                )

            chunks_info.append(chunk_info)

        # Calculate statistics
        avg_chunk_length = (
            sum(len(chunk['content']) for chunk in chunks) / len(chunks)
            if chunks else 0
        )

        success_rate = (
            (successful_embeddings / len(chunks) * 100)
            if chunks else 0
        )

        return {
            "status": "success",
            "document_id": document_id,
            "chunk_mode": chunk_mode,
            "chunking_strategy": chunking_strategy,
            "total_chunks": len(chunks),
            "successful_embeddings": successful_embeddings,
            "failed_embeddings": len(failed_chunks),
            "stored_count": stored_count,
            "processing_stats": {
                "original_content_length": len(markdown_content),
                "average_chunk_length": avg_chunk_length,
                "success_rate": f"{success_rate:.1f}%",
                "average_tokens_per_chunk": (
                    sum(chunk.get('token_count', 0) for chunk in chunks) / len(chunks)
                    if chunk_mode == "smart" and chunks else 0
                )
            },
            "chunks_preview": chunks_info,
            "failed_chunks": failed_chunks if failed_chunks else None
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding error: {str(e)}"
        )


@app.post("/api/v1/faq/add")
async def add_faq(request: dict):
    """
    API 4: Add FAQ - Thêm câu hỏi và câu trả lời FAQ
    Input: {"question": "Câu hỏi", "answer": "Câu trả lời", "faq_id": "optional_id"}
    Returns: Success response with FAQ ID
    """
    try:
        question = request.get("question", "").strip()
        answer = request.get("answer", "").strip()
        faq_id = request.get("faq_id", "").strip()

        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        if not answer:
            raise HTTPException(status_code=400, detail="Answer is required")

        if not faq_id:
            faq_id = f"faq_{str(uuid.uuid4())[:8]}"
        else:
            faq_id = re.sub(r'[^\w\-_.]', '_', str(faq_id))
            faq_id = re.sub(r'_+', '_', faq_id).strip('_')

        question_embedding = embedding_service.get_embedding(question)
        success = await milvus_manager.insert_faq(faq_id, question, answer, question_embedding)

        return {
            "status": "success",
            "faq_id": faq_id,
            "question": question,
            "answer": answer,
            "message": "FAQ added successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Add FAQ error: {str(e)}")


@app.delete("/api/v1/faq/delete/{faq_id}")
async def delete_faq(faq_id: str):
    """API 5: Delete FAQ - Xóa FAQ theo ID"""
    try:
        if not faq_id or not faq_id.strip():
            raise HTTPException(status_code=400, detail="FAQ ID is required")

        faq_id = faq_id.strip()
        success = await milvus_manager.delete_faq(faq_id)

        return {
            "status": "success",
            "faq_id": faq_id,
            "message": "FAQ deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete FAQ error: {str(e)}")


@app.delete("/api/v1/document/delete/{document_id}")
async def delete_document(document_id: str):
    """API 6: Delete Document - Xóa tất cả embeddings của một document_id"""
    try:
        if not document_id or not document_id.strip():
            raise HTTPException(status_code=400, detail="Document ID is required")

        document_id = document_id.strip()
        success = await milvus_manager.delete_document(document_id)

        return {
            "status": "success",
            "document_id": document_id,
            "message": "Document and all its embeddings deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete document error: {str(e)}")


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    try:
        milvus_status = await milvus_manager.health_check()
        embedding_status = embedding_service.is_ready()

        return {
            "status": "healthy",
            "service": "document-api",
            "version": "2.0.0",
            "port": 8000,
            "services": {
                "milvus": milvus_status,
                "embedding_model": embedding_status,
                "document_processor": "ready",
                "smart_chunking": "enabled"
            },
            "environment": {
                "milvus_host": os.getenv('MILVUS_HOST', 'milvus'),
                "milvus_port": os.getenv('MILVUS_PORT', '19530')
            },
            "chunking_modes": {
                "smart": "Semantic smart chunking (recommended)",
                "sentence": "Legacy sentence-based chunking",
                "legacy": "Backward compatible mode"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "document-api",
            "port": 8000,
            "error": str(e)
        }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )