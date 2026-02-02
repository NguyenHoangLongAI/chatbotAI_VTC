# RAG_Core/tools/personalization_vector_search.py
"""
Vector search tools cho Personalization Database
Tìm kiếm trong personalization_db thay vì default database
"""

from langchain_core.tools import tool
from typing import List, Dict, Any
import numpy as np
from models.embedding_model import embedding_model
from database.personalization_milvus_client import personalization_milvus_client
from config.settings import settings
import logging
import os

logger = logging.getLogger(__name__)

# ============================================================================
# COHERE RERANKER (shared từ vector_search.py)
# ============================================================================

cohere_client = None
COHERE_RERANK_MODEL = 'rerank-multilingual-v3.0'

try:
    import cohere

    cohere_api_key = None

    if hasattr(settings, 'COHERE_API_KEY'):
        cohere_api_key = settings.COHERE_API_KEY
        logger.info("📍 Found COHERE_API_KEY in settings")

    if not cohere_api_key:
        cohere_api_key = os.getenv('COHERE_API_KEY')
        if cohere_api_key:
            logger.info("📍 Found COHERE_API_KEY in environment")

    if not cohere_api_key:
        cohere_api_key = "NoQ9Jjvz5r1JeRWZG8L9dnl8BxYljmnOdiUfTnfk"
        logger.warning("⚠️ Using hardcoded COHERE_API_KEY")

    if cohere_api_key and cohere_api_key != "your-api-key-here":
        cohere_client = cohere.Client(cohere_api_key)

        if hasattr(settings, 'COHERE_RERANK_MODEL'):
            COHERE_RERANK_MODEL = settings.COHERE_RERANK_MODEL

        logger.info(f"✅ Cohere Reranker initialized for personalization")

except ImportError:
    logger.error("❌ Cohere library not installed")
    cohere_client = None
except Exception as e:
    logger.error(f"❌ Failed to initialize Cohere: {e}")
    cohere_client = None


# ============================================================================
# PERSONALIZATION SEARCH FUNCTIONS
# ============================================================================

def pad_vector_to_dimension(vector: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad vector with zeros to reach target dimension"""
    current_dim = vector.shape[0] if vector.ndim == 1 else vector.shape[1]

    if current_dim >= target_dim:
        return vector[:target_dim] if vector.ndim == 1 else vector[:, :target_dim]

    if vector.ndim == 1:
        padding = np.zeros(target_dim - current_dim, dtype=vector.dtype)
        return np.concatenate([vector, padding])
    else:
        padding = np.zeros((vector.shape[0], target_dim - current_dim), dtype=vector.dtype)
        return np.concatenate([vector, padding], axis=1)


def safe_encode_and_fix_dimension(
        query: str,
        target_collection: str,
        target_field: str
) -> np.ndarray:
    """Encode query and fix dimension for personalization_db"""
    try:
        query_vector = embedding_model.encode_single(query)
        expected_dim = personalization_milvus_client._get_collection_dimension(
            target_collection,
            target_field
        )

        if expected_dim > 0 and query_vector.shape[0] != expected_dim:
            logger.warning(
                f"Dimension mismatch. Expected: {expected_dim}, Got: {query_vector.shape[0]}. "
                f"Auto-fixing..."
            )
            query_vector = pad_vector_to_dimension(query_vector, expected_dim)
            logger.info(f"Vector dimension fixed to {expected_dim}")

        return query_vector

    except Exception as e:
        logger.error(f"Error encoding query: {str(e)}")
        raise


@tool
def search_personalization_documents(
        query: str,
        top_k: int = None
) -> List[Dict[str, Any]]:
    """
    Tìm kiếm tài liệu trong personalization_default collection

    Args:
        query: User query
        top_k: Number of results (default from settings)
    """
    try:
        if top_k is None:
            top_k = settings.TOP_K

        logger.info(f"🔍 Searching personalization_db documents: {query[:60]}")

        query_vector = safe_encode_and_fix_dimension(
            query,
            "personalization_default",
            "description_vector"
        )

        results = personalization_milvus_client.search_documents(
            query_vector,
            top_k=top_k,
            collection_name="personalization_default"
        )

        logger.info(f"✅ Found {len(results)} documents in personalization_db")
        return results

    except Exception as e:
        logger.error(f"Error in search_personalization_documents: {str(e)}")
        return [{"error": f"Lỗi tìm kiếm tài liệu: {str(e)}"}]


@tool
def search_personalization_faq(
        query: str,
        top_k: int = None
) -> List[Dict[str, Any]]:
    """
    Tìm kiếm FAQ trong personalization_faq_embeddings collection

    Args:
        query: User query
        top_k: Number of results (default from settings)
    """
    try:
        if top_k is None:
            top_k = getattr(settings, 'FAQ_TOP_K', 10)

        logger.info(f"🔍 Searching personalization_db FAQ: {query[:60]}")

        query_vector = safe_encode_and_fix_dimension(
            query,
            "personalization_faq_embeddings",
            "question_vector"
        )

        results = personalization_milvus_client.search_faq(
            query_vector,
            top_k=top_k,
            collection_name="personalization_faq_embeddings"
        )

        logger.info(f"✅ Found {len(results)} FAQs in personalization_db")
        return results

    except Exception as e:
        logger.error(f"Error in search_personalization_faq: {str(e)}")
        return [{"error": f"Lỗi tìm kiếm FAQ: {str(e)}"}]


@tool
def rerank_personalization_faq(
        query: str,
        faq_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Rerank personalization FAQ sử dụng Cohere

    Args:
        query: User query
        faq_results: FAQ results from vector search
    """
    try:
        logger.info("🔄 PERSONALIZATION FAQ RERANKER")
        logger.info(f"📝 Query: '{query[:100]}'")
        logger.info(f"   FAQs to rerank: {len(faq_results)}")

        if not faq_results:
            logger.warning("⚠️  No FAQ to rerank")
            return []

        if cohere_client is None:
            logger.warning("⚠️  Cohere not available, returning original FAQs")
            return faq_results

        # Prepare documents
        documents = []
        for faq in faq_results:
            question = faq.get('question', '').strip()
            answer = faq.get('answer', '').strip()
            combined = f"Câu hỏi: {question}\nTrả lời: {answer}"
            documents.append(combined)

        if not documents:
            logger.warning("⚠️  No valid FAQ documents")
            return faq_results

        # Call Cohere Rerank API
        logger.info(f"🌐 Calling Cohere API for personalization FAQ")

        import time
        start_time = time.time()

        rerank_response = cohere_client.rerank(
            query=query,
            documents=documents,
            model=COHERE_RERANK_MODEL,
            top_n=len(documents),
            return_documents=False
        )

        api_time = time.time() - start_time
        logger.info(f"⏱️  Cohere API completed in {api_time:.3f}s")

        # Map scores back to FAQs
        reranked_faq = []
        for result in rerank_response.results:
            idx = result.index
            score = result.relevance_score

            faq_copy = faq_results[idx].copy()
            faq_copy['rerank_score'] = float(score)
            faq_copy['rerank_source'] = 'cohere'
            reranked_faq.append(faq_copy)

        # Sort by rerank_score
        reranked_faq.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)

        logger.info(f"✅ Reranked {len(reranked_faq)} personalization FAQs")
        logger.info(f"   Best score: {reranked_faq[0].get('rerank_score', 0):.3f}")

        return reranked_faq

    except Exception as e:
        logger.error(f"❌ Error in personalization FAQ reranking: {e}", exc_info=True)
        return sorted(
            faq_results,
            key=lambda x: x.get('similarity_score', 0),
            reverse=True
        )


@tool
def rerank_personalization_documents(
        query: str,
        documents: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Rerank personalization documents sử dụng Cohere

    Args:
        query: User query
        documents: Document results from vector search
    """
    try:
        if not documents:
            logger.warning("No personalization documents to rerank")
            return []

        if cohere_client is None:
            logger.warning("Cohere not available, returning original documents")
            return documents

        # Prepare document texts
        doc_texts = []
        for doc in documents:
            doc_text = doc.get('description', '') or doc.get('answer', '') or doc.get('content', '')
            doc_texts.append(doc_text)

        if not doc_texts:
            logger.warning("No valid personalization document texts")
            return documents

        logger.info(f"🔄 Reranking {len(doc_texts)} personalization documents")

        rerank_response = cohere_client.rerank(
            query=query,
            documents=doc_texts,
            model=COHERE_RERANK_MODEL,
            top_n=len(doc_texts),
            return_documents=False
        )

        # Map scores back
        reranked_docs = []
        for result in rerank_response.results:
            idx = result.index
            score = result.relevance_score

            doc_copy = documents[idx].copy()
            doc_copy['rerank_score'] = float(score)
            doc_copy['rerank_source'] = 'cohere'
            reranked_docs.append(doc_copy)

        # Sort
        reranked_docs.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)

        logger.info(f"✅ Reranked {len(reranked_docs)} personalization documents")
        logger.info(f"   Best score: {reranked_docs[0].get('rerank_score', 0):.3f}")

        return reranked_docs

    except Exception as e:
        logger.error(f"Error in personalization document reranking: {e}", exc_info=True)
        return documents


@tool
def check_personalization_db_connection() -> Dict[str, Any]:
    """Kiểm tra kết nối personalization_db"""
    try:
        is_connected = personalization_milvus_client.check_connection()

        result = {
            "connected": is_connected,
            "database": personalization_milvus_client.database_name,
            "message": "Kết nối bình thường" if is_connected else "Mất kết nối personalization_db"
        }

        if is_connected:
            try:
                # List collections
                collections = personalization_milvus_client.list_collections()
                result["collections"] = collections

                # Check dimensions
                test_vector = embedding_model.encode_single("test")
                embedding_dim = test_vector.shape[0]

                doc_dim = personalization_milvus_client._get_collection_dimension(
                    "personalization_default", "description_vector"
                )
                faq_dim = personalization_milvus_client._get_collection_dimension(
                    "personalization_faq_embeddings", "question_vector"
                )

                result["dimension_info"] = {
                    "embedding_model_dimension": embedding_dim,
                    "document_collection_dimension": doc_dim,
                    "faq_collection_dimension": faq_dim,
                    "dimension_match": {
                        "documents": embedding_dim == doc_dim,
                        "faq": embedding_dim == faq_dim
                    }
                }

            except Exception as dim_error:
                result["dimension_check_error"] = str(dim_error)

        # Add Cohere status
        result["cohere_reranker"] = {
            "available": cohere_client is not None,
            "model": COHERE_RERANK_MODEL if cohere_client else None
        }

        return result

    except Exception as e:
        return {
            "connected": False,
            "database": "personalization_db",
            "message": f"Lỗi kiểm tra kết nối: {str(e)}"
        }