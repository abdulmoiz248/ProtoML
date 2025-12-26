"""
PDF Processor with Embeddings
Downloads, parses PDFs, and generates embeddings for semantic search
"""

import os
import requests
import fitz  # PyMuPDF
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import torch
import config


class PDFProcessor:
    """Process PDF papers and generate embeddings"""
    
    def __init__(self):
        self.cache_dir = config.PDF_CACHE_DIR
        self.embeddings_dir = config.EMBEDDINGS_DIR
        self.chunk_size = config.CHUNK_SIZE
        
        # Load embedding model
        print(f"🔄 Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        print("✅ Embedding model loaded")
    
    def download_pdf(self, paper: Dict) -> str:
        """
        Download PDF from arXiv
        
        Args:
            paper: Paper dictionary with pdf_url
            
        Returns:
            Path to downloaded PDF
        """
        pdf_url = paper['pdf_url']
        arxiv_id = paper['arxiv_id']
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        pdf_path = os.path.join(self.cache_dir, f"{arxiv_id}.pdf")
        
        # Check if already downloaded
        if os.path.exists(pdf_path):
            print(f"  ✓ PDF already cached: {arxiv_id}")
            return pdf_path
        
        print(f"  📥 Downloading PDF: {arxiv_id}")
        
        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            print(f"  ✅ Downloaded: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            print(f"  ❌ Error downloading PDF: {str(e)}")
            return None
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract text from PDF by sections
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of text chunks with metadata
        """
        print(f"  📄 Extracting text from PDF...")
        
        doc = None
        try:
            doc = fitz.open(pdf_path)
            chunks = []
            total_pages = len(doc)
            
            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text()
                
                # Split page text into chunks
                words = text.split()
                current_chunk = []
                current_length = 0
                
                for word in words:
                    current_chunk.append(word)
                    current_length += len(word) + 1  # +1 for space
                    
                    if current_length >= self.chunk_size:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append({
                            'text': chunk_text,
                            'page': page_num + 1,
                            'chunk_id': len(chunks)
                        })
                        current_chunk = []
                        current_length = 0
                
                # Add remaining text
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'page': page_num + 1,
                        'chunk_id': len(chunks)
                    })
            
            print(f"  ✅ Extracted {len(chunks)} chunks from {total_pages} pages")
            return chunks
            
        except Exception as e:
            print(f"  ❌ Error extracting text: {str(e)}")
            return []
        finally:
            # Ensure document is always closed
            if doc is not None:
                doc.close()
    
    def generate_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """
        Generate embeddings for text chunks
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Numpy array of embeddings
        """
        print(f"  🔢 Generating embeddings for {len(chunks)} chunks...")
        
        try:
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                batch_size=32
            )
            
            print(f"  ✅ Generated embeddings: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            print(f"  ❌ Error generating embeddings: {str(e)}")
            return np.array([])
    
    def process_paper(self, paper: Dict) -> Tuple[List[Dict], np.ndarray]:
        """
        Complete pipeline: download, extract, embed
        
        Args:
            paper: Paper dictionary
            
        Returns:
            Tuple of (chunks, embeddings)
        """
        print(f"\n📊 Processing paper: {paper['title'][:60]}...")
        
        # Download PDF
        pdf_path = self.download_pdf(paper)
        if not pdf_path:
            return [], np.array([])
        
        # Extract text
        chunks = self.extract_text_from_pdf(pdf_path)
        if not chunks:
            return [], np.array([])
        
        # Generate embeddings
        embeddings = self.generate_embeddings(chunks)
        if len(embeddings) == 0:
            return chunks, np.array([])
        
        print(f"  ✅ Embeddings ready ({embeddings.shape[0]} vectors)")
        
        return chunks, embeddings
    
    def search_similar_chunks(
        self,
        query: str,
        chunks: List[Dict],
        embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for similar chunks using semantic search
        
        Args:
            query: Search query
            chunks: List of text chunks
            embeddings: Chunk embeddings
            top_k: Number of results to return
            
        Returns:
            List of most similar chunks
        """
        if len(embeddings) == 0:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Calculate similarity
        similarities = np.dot(embeddings, query_embedding.T).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return top chunks with similarity scores
        results = []
        for idx in top_indices:
            chunk = chunks[idx].copy()
            chunk['similarity_score'] = float(similarities[idx])
            results.append(chunk)
        
        return results
