import logging
from typing import Optional
from app.services.document_processor import DocumentProcessor
from app.services.core.repository import DatabaseRepository
logger = logging.getLogger(__name__)

class ReadProcessDocument:
    __slots__ = ('db_repository', 'processor')

    def __init__(self,
        db_repository: DatabaseRepository = None,
        processor: DocumentProcessor = None):
        
        self.db_repository = db_repository or DatabaseRepository()
        self.processor = processor or DocumentProcessor()

    async def _get_doc_from_sql(self,city_doc_id:int = None):
        where_clause = f"where IsDeleted=0 and CityDocumentID={city_doc_id}" 
        return await self.db_repository.engine.fetch_df_async(
        f"select CityDocumentID,CityID,FilePath,FileType ,PillarID from CityDocuments  {where_clause}")
    

    async def process_document(self,city_doc_id: int):
        """Triggered after file upload — processes in background"""
        # Fetch doc info from SQL, then process
        df = await self._get_doc_from_sql(city_doc_id)

        for doc in df.itertuples(index=False):
            try:
               await self.processor.process_document(
                    file_path = doc.FilePath,
                    file_type = doc.FileType,
                    city_doc_id = doc.CityDocumentID,
                    city_id = doc.CityID,
                    pillar_id = doc.PillarID
                )
            except Exception as e:
                logger.error(f"Failed to analyze city {doc.CityID}: {e}")
                continue        

    async def delete_document(self,city_id: int, city_doc_id: int):
        """Delete all chunks from vector DB for a document"""

        try:   
            self.processor.delete_document(city_id, city_doc_id)

            deleteQuery = """
                Delete from DocumentChunks
                WHERE CityDocumentID = ?
            """
            await self.db_repository.engine.execute_write_async(deleteQuery,(city_doc_id,))

        except Exception as e:
            logger.error(f"Failed to delete chunks from vector DB city_doc_id {city_doc_id}: {e}")



read_process_document = ReadProcessDocument()