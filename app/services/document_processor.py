import os
import uuid
import fitz          # PyMuPDF for PDF
import docx          # python-docx for DOCX
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import logging
from app.services.core.repository import DatabaseRepository

CHROMA_PATH = "./chroma_store"

logger = logging.getLogger(__name__)

class DocumentProcessor:
    __slots__ = ('repository', 'client', 'embed_fn')

    def __init__(self):
        # Ensure directory exists
        if not os.path.exists(CHROMA_PATH):
            os.makedirs(CHROMA_PATH)

        try:         
            self.client = chromadb.PersistentClient(
                path=CHROMA_PATH,
                settings=chromadb.config.Settings(
                    anonymized_telemetry=False
                )
            )
            
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            raise

        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.repository = DatabaseRepository()

    def get_or_create_collection(self, city_id: int):
        """One ChromaDB collection per city (or per pillar if you want finer grain)"""
        name =  f"city_{city_id}"
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"}
        )
    
    def get_collection(self, city_id: int, city_doc_id:int):
        name =  f"city_{city_id}"
        return self.client.get_collection(name=name)

    async def process_document(self, file_path: str, file_type: str,
                         city_doc_id: int, city_id: int,
                         pillar_id: Optional[int]) -> Dict[str, Any]:
        """Main entry — extracts TOC, chunks, embeds, stores."""
        if file_type == ".pdf":
            sections = self._extract_pdf_sections(file_path)
        elif file_type in (".docx", ".doc"):
            sections = self._extract_docx_sections(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        toc_records = []
        chunk_records = []
        collection = self.get_or_create_collection(city_id)

        texts, ids, metadatas = [], [], []

        for section in sections:
            toc_id = await self.repository.save_toc_section(section, city_doc_id, city_id, pillar_id)
            toc_records.append({**section, "toc_id": toc_id})

            for idx, chunk_text in enumerate(self._chunk_text(section["text"])):
                chunk_id = str(uuid.uuid4())
                texts.append(chunk_text)
                ids.append(chunk_id)
                metadatas.append({
                    "city_id": city_id,
                    "pillar_id": pillar_id or -1,
                    "city_doc_id": city_doc_id,
                    "toc_id": toc_id,
                    "section_path": section["path"],
                    "section_title": section["title"],
                    "chunk_index": idx
                })
                chunk_records.append({
                    "chunk_id": chunk_id,
                    "toc_id": toc_id,
                    "chunk_text": chunk_text,
                    "chunk_index": idx
                })

        # Batch upsert into ChromaDB (embeddings generated automatically)
        if texts:
            collection.upsert(documents=texts, ids=ids, metadatas=metadatas)

        # Save chunk metadata to SQL
        await self.repository.save_document_chunks(chunk_records, city_doc_id, city_id, pillar_id)

        return {"toc": toc_records, "chunk_count": len(chunk_records)}

    def _extract_pdf_sections(self, path: str) -> List[Dict]:
        """Extract sections using PDF outline/bookmarks + heading detection"""
        doc = fitz.open(path)
        sections = []

        toc = doc.get_toc()  # [[level, title, page], ...]
        if toc:
            for i, (level, title, page) in enumerate(toc):
                next_page = toc[i + 1][2] if i + 1 < len(toc) else doc.page_count
                text = ""
                for pg in range(page - 1, min(next_page - 1, doc.page_count)):
                    text += doc[pg].get_text()
                path_parts = [t for l, t, _ in toc[:i+1] if l <= level]
                sections.append({
                    "title": title,
                    "level": level,
                    "path": " > ".join(path_parts),
                    "page_start": page,
                    "page_end": next_page - 1,
                    "text": text.strip()
                })
        else:
            # Fallback: treat whole doc as one section
            full_text = "".join(doc[p].get_text() for p in range(doc.page_count))
            sections.append({
                "title": "Full Document",
                "level": 1,
                "path": "Full Document",
                "page_start": 1,
                "page_end": doc.page_count,
                "text": full_text.strip()
            })
        return sections

    def _extract_docx_sections(self, path: str) -> List[Dict]:
        """Extract sections using Word heading styles"""
        doc = docx.Document(path)
        sections = []
        current = {"title": "Introduction", "level": 1, "path": "Introduction",
                   "page_start": 1, "page_end": 1, "text": ""}
        breadcrumb = ["Introduction"]

        for para in doc.paragraphs:
            style = para.style.name
            if style.startswith("Heading"):
                if current["text"].strip():
                    sections.append(current)
                level = int(style.split()[-1]) if style[-1].isdigit() else 1
                breadcrumb = breadcrumb[:level - 1] + [para.text]
                current = {
                    "title": para.text,
                    "level": level,
                    "path": " > ".join(breadcrumb),
                    "page_start": 1,
                    "page_end": 1,
                    "text": ""
                }
            else:
                current["text"] += para.text + "\n"

        if current["text"].strip():
            sections.append(current)
        return sections

    def _chunk_text(self, text: str, chunk_size: int = 400,
                    overlap: int = 80) -> List[str]:
        """Sliding window chunker — words based"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:   # skip tiny fragments
                chunks.append(chunk)
        return chunks

    def delete_document(self,city_id: int, city_doc_id: int):
        collection = self.get_collection(city_id, city_doc_id)
        if collection:
            collection.delete(
                where={"city_doc_id": city_doc_id}
            )




    