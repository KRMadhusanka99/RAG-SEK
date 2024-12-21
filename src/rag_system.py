from pathlib import Path
from typing import List, Dict, Tuple, Optional
import os
from dataclasses import dataclass
from datetime import datetime
import faiss
import numpy as np
from openai import OpenAI
import PyPDF2
from document_store import DocumentStore, DocumentMetadata
import time
from functools import lru_cache

@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: float

@dataclass
class SourceDocument:
    content: str
    metadata: DocumentMetadata
    relevance_score: float

class EnhancedRAG:
    def __init__(self, api_key: str, document_store: DocumentStore):
        self.client = OpenAI(api_key=api_key)
        self.document_store = document_store
        self.index = None
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.chat_history: List[ChatMessage] = []
        
    def process_pdf(self, pdf_path: Path, chunk_size: int = 500):
        chunks_to_process = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text = pdf_reader.pages[page_num].extract_text()
                chunks = self._create_chunks(text, chunk_size)
                chunks_to_process.extend([
                    (chunk, page_num, idx) 
                    for idx, chunk in enumerate(chunks)
                ])
        
        # Batch process embeddings (8 chunks at a time)
        batch_size = 8
        for i in range(0, len(chunks_to_process), batch_size):
            batch = chunks_to_process[i:i + batch_size]
            texts = [chunk[0] for chunk in batch]
            
            # Get embeddings for batch
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            embeddings = [e.embedding for e in response.data]
            
            # Process results
            for (chunk, page_num, chunk_idx), embedding in zip(batch, embeddings):
                metadata = DocumentMetadata(
                    file_name=pdf_path.name,
                    page_number=page_num + 1,
                    chunk_index=chunk_idx,
                    timestamp=time.time()
                )
                self.documents.append(chunk)
                self.embeddings.append(embedding)
                self.metadata.append(metadata)
        
        self._update_index()
    
    def chat(self, user_input: str, k: int = 3) -> Tuple[str, List[SourceDocument]]:
        self.chat_history.append(ChatMessage(
            role="user",
            content=user_input,
            timestamp=time.time()
        ))
        
        relevant_docs = self._get_relevant_documents(user_input, k)
        
        # Stricter check for relevant documents
        if not relevant_docs or all(doc.relevance_score < 0.7 for doc in relevant_docs):
            answer = "I don't have enough information in the provided documents to answer this question."
            self.chat_history.append(ChatMessage(
                role="assistant",
                content=answer,
                timestamp=time.time()
            ))
            return answer, []
        
        context = self._create_context(relevant_docs)
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system", 
                    "content": """You are a document-based Q&A system that MUST follow these rules strictly:

                    1. ONLY use information explicitly present in the provided context
                    2. If the exact answer is not in the context, respond with EXACTLY: "I don't have enough information in the provided documents to answer this question."
                    3. Do not make assumptions or use external knowledge
                    4. Every statement must be directly supported by the context
                    5. Always cite the specific document and page number in this format: [Document: X, Page: Y]
                    6. If the question is not about the content in the documents, respond with: "I can only answer questions about the content in the provided documents."
                    7. Do not try to be helpful by providing related or general information
                    8. If the context is only partially relevant, still respond with "I don't have enough information..."
                    """
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"}
            ],
            temperature=0.0  # Set to 0 for most deterministic responses
        )
        
        answer = response.choices[0].message.content
        
        # Add assistant response to history
        self.chat_history.append(ChatMessage(
            role="assistant",
            content=answer,
            timestamp=time.time()
        ))
        
        return answer, relevant_docs
    
    def _create_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into smaller chunks"""
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_size += len(word) + 1  # +1 for space
            if current_size > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    @lru_cache(maxsize=1000)
    def _get_cached_embedding(self, text: str) -> List[float]:
        """Cached version of embedding generation"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    def _update_index(self):
        """Update or create FAISS index"""
        if not self.embeddings:
            return
        
        embeddings_array = np.array(self.embeddings).astype('float32')
        dimension = len(self.embeddings[0])
        
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)
    
    def _get_relevant_documents(self, query: str, k: int = 3) -> List[SourceDocument]:
        """Get relevant documents for the query"""
        query_embedding = self._get_cached_embedding(query)
        query_embedding_array = np.array([query_embedding]).astype('float32')
        
        if self.index is None:
            return []
        
        D, I = self.index.search(query_embedding_array, k)
        
        # Add relevance threshold (0.6 similarity score)
        relevant_docs = []
        for idx, i in enumerate(I[0]):
            similarity = 1 - D[0][idx]
            if similarity >= 0.6:  # Only include if similarity is high enough
                relevant_docs.append(
                    SourceDocument(
                        content=self.documents[i],
                        metadata=self.metadata[i],
                        relevance_score=similarity
                    )
                )
        return relevant_docs
    
    def _create_context(self, relevant_docs: List[SourceDocument]) -> str:
        """Create context from relevant documents"""
        context_parts = []
        for doc in relevant_docs:
            context_parts.append(
                f"Source: {doc.metadata.file_name} (Page {doc.metadata.page_number})\n"
                f"Content: {doc.content}\n"
            )
        return "\n".join(context_parts)
    
    def remove_document(self, filename: str) -> bool:
        """Remove document and its embeddings from the system"""
        # Find all indices to remove
        indices_to_remove = [
            i for i, meta in enumerate(self.metadata) 
            if meta.file_name == filename
        ]
        
        if not indices_to_remove:
            return False
        
        # Remove from all lists in reverse order to maintain correct indices
        for idx in sorted(indices_to_remove, reverse=True):
            del self.documents[idx]
            del self.embeddings[idx]
            del self.metadata[idx]
        
        # Rebuild the FAISS index if we have remaining documents
        if self.embeddings:
            self._update_index()
        else:
            self.index = None
        
        return True