# RAG_Core/database/personalization_milvus_client.py
"""
Milvus Client cho Personalization Database
Kết nối vào personalization_db thay vì database default
"""

from pymilvus import connections, Collection, utility, db
from typing import List, Dict, Any
import numpy as np
from config.settings import settings
import logging
import os

logger = logging.getLogger(__name__)


class PersonalizationMilvusClient:
    """Milvus Client cho personalization_db"""

    def __init__(self, database_name: str = "personalization_db"):
        self.database_name = database_name
        self.connected = False
        self.expected_dimension = None
        self._connect()

    def _connect(self):
        """Connect to Milvus và switch sang personalization_db"""
        try:
            try:
                connections.disconnect("personalization")
            except:
                pass

            connections.connect(
                alias="personalization",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT
            )

            logger.info(f"✅ Connected to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")

            # Tạo database nếu chưa có
            databases = db.list_database()
            if self.database_name not in databases:
                logger.info(f"📁 Creating database: {self.database_name}")
                db.create_database(self.database_name)

            # Switch sang personalization database
            db.using_database(self.database_name)
            logger.info(f"✅ Using database: {self.database_name}")

            self.connected = True

        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            self.connected = False

    def check_connection(self) -> bool:
        """Check if connected"""
        if not self.connected:
            return False

        try:
            current_db = db.using_database()
            if current_db != self.database_name:
                logger.warning(f"Wrong database, switching to {self.database_name}")
                db.using_database(self.database_name)

            utility.list_collections(timeout=2, using="personalization")
            return True
        except Exception as e:
            logger.warning(f"Lost connection: {e}")
            self.connected = False
            return False

    def _get_collection_dimension(self, collection_name: str, vector_field: str) -> int:
        """Get expected dimension"""
        try:
            collection = Collection(collection_name, using="personalization")
            schema = collection.schema
            for field in schema.fields:
                if field.name == vector_field:
                    return field.params.get('dim', 0)
            return 0
        except Exception as e:
            logger.error(f"Error getting dimension: {str(e)}")
            return 0

    def _validate_vector_dimension(
            self, vector: np.ndarray, collection_name: str,
            vector_field: str, auto_fix: bool = True
    ) -> np.ndarray:
        """Validate and adjust vector dimension"""
        expected_dim = self._get_collection_dimension(collection_name, vector_field)
        actual_dim = vector.shape[0] if vector.ndim == 1 else vector.shape[1]

        if expected_dim == 0:
            return vector

        if actual_dim != expected_dim:
            if auto_fix:
                return self._adjust_vector_dimension(vector, expected_dim)
            else:
                raise ValueError(f"Dimension mismatch: expected {expected_dim}, got {actual_dim}")

        return vector

    def _adjust_vector_dimension(self, vector: np.ndarray, target_dim: int) -> np.ndarray:
        """Adjust vector to target dimension"""
        if vector.ndim > 1:
            current_dim = vector.shape[1]
            if current_dim < target_dim:
                padding = np.zeros((vector.shape[0], target_dim - current_dim), dtype=vector.dtype)
                return np.concatenate([vector, padding], axis=1)
            elif current_dim > target_dim:
                return vector[:, :target_dim]
        else:
            current_dim = vector.shape[0]
            if current_dim < target_dim:
                padding = np.zeros(target_dim - current_dim, dtype=vector.dtype)
                return np.concatenate([vector, padding])
            elif current_dim > target_dim:
                return vector[:target_dim]

        return vector

    def search_documents(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search in personalization_db collection"""
        if not self.check_connection():
            raise ConnectionError("Milvus connection lost")

        try:
            collection_name = "personalization_db"
            collection = Collection(collection_name, using="personalization")
            collection.load()

            query_vector = self._validate_vector_dimension(
                query_vector, collection_name, "description_vector"
            )

            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}
            }

            results = collection.search(
                data=[query_vector.tolist()],
                anns_field="description_vector",
                param=search_params,
                limit=top_k,
                output_fields=["document_id", "description"]
            )

            documents = []
            for hits in results:
                for hit in hits:
                    documents.append({
                        "document_id": hit.entity.get("document_id"),
                        "description": hit.entity.get("description"),
                        "similarity_score": hit.score
                    })

            logger.info(f"✅ Found {len(documents)} docs from personalization_db")
            return documents

        except Exception as e:
            logger.error(f"Error searching personalization documents: {str(e)}")
            raise

    def search_faq(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search in personalization_faq_embeddings collection"""
        if not self.check_connection():
            raise ConnectionError("Milvus connection lost")

        try:
            collection_name = "personalization_faq_embeddings"
            collection = Collection(collection_name, using="personalization")
            collection.load()

            query_vector = self._validate_vector_dimension(
                query_vector, collection_name, "question_vector"
            )

            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}
            }

            results = collection.search(
                data=[query_vector.tolist()],
                anns_field="question_vector",
                param=search_params,
                limit=top_k,
                output_fields=["faq_id", "question", "answer"]
            )

            faqs = []
            for hits in results:
                for hit in hits:
                    faqs.append({
                        "faq_id": hit.entity.get("faq_id"),
                        "question": hit.entity.get("question"),
                        "answer": hit.entity.get("answer"),
                        "similarity_score": hit.score
                    })

            logger.info(f"✅ Found {len(faqs)} FAQs from personalization_faq_embeddings")
            return faqs

        except Exception as e:
            logger.error(f"Error searching personalization FAQ: {str(e)}")
            raise


# Global instance
personalization_milvus_client = PersonalizationMilvusClient(
    database_name=os.getenv("MILVUS_PERSONALIZATION_DATABASE", "personalization_db")
)
