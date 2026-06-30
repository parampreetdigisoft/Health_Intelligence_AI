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

    async def _get_doc_from_sql(self,country_doc_id:int = None):
        where_clause = f"where IsDeleted=0 and CountryDocumentID={country_doc_id}" 
        return await self.db_repository.engine.fetch_df_async(
        f"select CountryDocumentID,CountryID,FilePath,FileType ,PillarID,DocumentLevel from CountryDocuments  {where_clause}")
    

    async def process_document(self,country_doc_id: int):
        """Triggered after file upload — processes in background"""
        # Fetch doc info from SQL, then process
        df = await self._get_doc_from_sql(country_doc_id)

        for doc in df.itertuples(index=False):
            try:
               await self.processor.process_document(
                    file_path = doc.FilePath,
                    file_type = doc.FileType,
                    document_level = doc.DocumentLevel,
                    country_doc_id = doc.CountryDocumentID,
                    country_id = doc.CountryID,
                    pillar_id = doc.PillarID
                )
            except Exception as e:
                logger.error(f"Failed to analyze country {doc.CountryID}: {e}")
                continue        

    async def delete_document(self,country_doc_id: int):
        """Delete all chunks from vector DB for a document"""
        try:   

            df = await self._get_doc_from_sql(country_doc_id)
            for doc in df.itertuples(index=False):              
                self.processor.delete_document(
                    document_level = doc.DocumentLevel,
                    country_doc_id = doc.CountryDocumentID,
                    country_id = doc.CountryID)

                deleteQuery = """
                    DELETE FROM DocumentChunks
                    WHERE CountryDocumentID = ?;

                    DELETE FROM DocumentTOC
                    WHERE CountryDocumentID = ?;
                """

                await self.db_repository.engine.execute_write_async(
                    deleteQuery,
                    (country_doc_id, country_doc_id)
                )

        except Exception as e:
            logger.error(f"Failed to delete chunks from vector DB country_doc_id {country_doc_id}: {e}")



read_process_document = ReadProcessDocument()