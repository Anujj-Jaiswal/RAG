"""
RAG System - Single Unified Execution Script
Complete RAG solution in one file
Run: python rag_system.py
"""

import os
import time
import sqlite3
import json
import numpy as np
from typing import List, Tuple
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

PDF_DIRECTORY = "./pdfs"
DB_PATH = "./vector_db/embeddings.db"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVAL_THRESHOLD = 0.6
TOP_K_RESULTS = 3
MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 1024

os.makedirs(PDF_DIRECTORY, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


# ============================================================================
# PDF PROCESSOR
# ============================================================================

class PDFProcessor:
    """Extract and chunk text from PDFs"""
    
    def __init__(self, pdf_directory: str):
        self.pdf_directory = pdf_directory
    
    def extract_text_from_pdfs(self) -> str:
        """Extract text from all PDFs in directory"""
        all_text = ""
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f" No PDF files found in {self.pdf_directory}")
            return ""
        
        print(f"\nProcessing {len(pdf_files)} PDF file(s)...")
        
        for pdf_file in pdf_files:
            try:
                pdf_path = os.path.join(self.pdf_directory, pdf_file)
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    for page in pdf_reader.pages:
                        all_text += page.extract_text() + "\n"
                print(f"   ✓ Extracted: {pdf_file}")
            except Exception as e:
                print(f"   ✗ Error reading {pdf_file}: {str(e)}")
        
        return all_text
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks for better context retention"""
        if not text:
            return []
        
        chunks = []
        step = chunk_size - overlap
        
        for i in range(0, len(text), step):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        print(f"\n Text split into {len(chunks)} chunks")
        print(f"   Average chunk size: {sum(len(c) for c in chunks) // len(chunks) if chunks else 0} characters")
        
        return chunks


# ============================================================================
# EMBEDDING MANAGER
# ============================================================================

class EmbeddingManager:
    """Manage embeddings and vector database operations using SQLite"""
    
    def __init__(self, model_name: str, db_path: str):
        print(f"\n Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✓ Database initialized")
    
    def generate_embeddings(self, chunks: List[str]) -> List[np.ndarray]:
        """Generate embeddings for text chunks - lightweight model, CPU friendly"""
        print(f"\n Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
        print("✓ Embeddings generated")
        return embeddings
    
    def store_embeddings(self, chunks: List[str], embeddings: List[np.ndarray]):
        """Store chunks and embeddings in SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM embeddings')
        
        for chunk, embedding in zip(chunks, embeddings):
            embedding_blob = embedding.astype(np.float32).tobytes()
            cursor.execute('''
                INSERT INTO embeddings (text, embedding, metadata)
                VALUES (?, ?, ?)
            ''', (chunk, embedding_blob, json.dumps({"source": "pdf", "length": len(chunk)})))
        
        conn.commit()
        conn.close()
        print(f"✓ Stored {len(chunks)} embeddings in database")
    
    def retrieve_similar_chunks(self, query: str, top_k: int = 3, threshold: float = 0.6) -> List[Tuple[str, float]]:
        """Retrieve most similar chunks using cosine similarity"""
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, text, embedding FROM embeddings')
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
        
        similarities = []
        for row_id, text, embedding_blob in rows:
            stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            # Cosine similarity: dot product / (norm1 * norm2)
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding) + 1e-10
            )
            similarities.append((text, float(similarity)))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = [(text, score) for text, score in similarities[:top_k] if score >= threshold]
        
        return results


# ============================================================================
# LLM INTERFACE
# ============================================================================

class GeminiInterface:
    """Interface with Google Gemini 2.5 Flash LLM"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash", temperature: float = 0.7):
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.model = genai.GenerativeModel(model_name)
        print(f"✓ Gemini model loaded: {model_name}")
    
    def generate_response(self, query: str, context: str, max_tokens: int = 1024) -> str:
        """Generate response using Gemini with RAG context"""
        system_prompt = """You are a helpful assistant that answers questions based on provided context.
- Answer only using the provided context
- If context doesn't contain relevant information, say "The provided documents don't contain information about this"
- Be concise and clear
- Cite relevant parts when possible"""
        
        full_prompt = f"""Context:
{context}

Question: {query}

Please provide a concise answer based on the context above."""
        
        try:
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    @staticmethod
    def format_context(retrieved_chunks: list) -> str:
        """Format retrieved chunks into context string"""
        context = ""
        for i, (chunk, score) in enumerate(retrieved_chunks, 1):
            context += f"\n--- Document {i} (Relevance: {score:.2%}) ---\n{chunk}\n"
        return context


# ============================================================================
# RAG SYSTEM ORCHESTRATOR
# ============================================================================

class RAGSystem:
    """Complete RAG system - coordinates all components"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor(PDF_DIRECTORY)
        self.embedding_manager = EmbeddingManager(EMBEDDINGS_MODEL, DB_PATH)
        self.llm = GeminiInterface(GEMINI_API_KEY, MODEL_NAME, TEMPERATURE)
        self.chunks = []
        self.processed = False
    
    def index_documents(self):
        """Process PDFs and build vector index"""
        start_time = time.time()
        
        print("\n" + "="*60)
        print(" INDEXING DOCUMENTS")
        print("="*60)
        
        text = self.pdf_processor.extract_text_from_pdfs()
        if not text:
            print(" No text extracted from PDFs")
            return False
        
        self.chunks = self.pdf_processor.chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not self.chunks:
            print(" No chunks created")
            return False
        
        embeddings = self.embedding_manager.generate_embeddings(self.chunks)
        self.embedding_manager.store_embeddings(self.chunks, embeddings)
        
        elapsed_time = time.time() - start_time
        print(f"\n Indexing complete in {elapsed_time:.2f}s")
        self.processed = True
        return True
    
    def query(self, question: str) -> dict:
        """Process query and return answer with metrics"""
        if not self.processed:
            return {"error": "Documents not indexed. Run index_documents() first"}
        
        start_time = time.time()
        
        print("\n" + "="*60)
        print(" PROCESSING QUERY")
        print("="*60)
        print(f"Question: {question}\n")
        
        # Retrieve relevant chunks
        retrieval_start = time.time()
        retrieved_chunks = self.embedding_manager.retrieve_similar_chunks(
            question, TOP_K_RESULTS, RETRIEVAL_THRESHOLD
        )
        retrieval_time = time.time() - retrieval_start
        
        if not retrieved_chunks:
            return {
                "query": question,
                "answer": "No relevant information found in the documents.",
                "retrieved_chunks": [],
                "metrics": {
                    "retrieval_time": retrieval_time,
                    "generation_time": 0,
                    "total_time": retrieval_time,
                    "chunks_retrieved": 0
                }
            }
        
        # Format context and generate response
        context = GeminiInterface.format_context(retrieved_chunks)
        
        generation_start = time.time()
        answer = self.llm.generate_response(question, context, MAX_OUTPUT_TOKENS)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        avg_chunk_length = sum(len(chunk[0]) for chunk in retrieved_chunks) / len(retrieved_chunks)
        
        metrics = {
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "chunks_retrieved": len(retrieved_chunks),
            "avg_chunk_length": int(avg_chunk_length),
            "avg_relevance_score": sum(score for _, score in retrieved_chunks) / len(retrieved_chunks)
        }
        
        return {
            "query": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "metrics": metrics
        }
    
    @staticmethod
    def print_results(result: dict):
        """Pretty print query results and metrics"""
        print("\n" + "="*60)
        print(" ANSWER")
        print("="*60)
        print(result["answer"])
        
        print("\n" + "="*60)
        print("RETRIEVED CONTEXT")
        print("="*60)
        for i, (chunk, score) in enumerate(result["retrieved_chunks"], 1):
            print(f"\n Document {i} (Relevance: {score:.1%})")
            print(f"   {chunk[:200]}..." if len(chunk) > 200 else f"   {chunk}")
        
        print("\n" + "="*60)
        print(" METRICS & INSIGHTS")
        print("="*60)
        metrics = result["metrics"]
        print(f"  Chunks Retrieved:      {metrics['chunks_retrieved']}")
        print(f"  Avg Chunk Length:      {metrics['avg_chunk_length']} characters")
        print(f"  Avg Relevance Score:   {metrics['avg_relevance_score']:.1%}")
        print(f"  Retrieval Time:        {metrics['retrieval_time']:.3f}s")
        print(f"  LLM Generation Time:   {metrics['generation_time']:.3f}s")
        print(f"  Total Time:            {metrics['total_time']:.3f}s")
        print("="*60 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution flow"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "   RAG SYSTEM - Retrieval Augmented Generation".center(58) + "║")
    print("║" + "  Gemini 2.5 Flash + Sentence Transformers".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    # Check PDFs
    if not os.path.exists(PDF_DIRECTORY):
        os.makedirs(PDF_DIRECTORY)
        print(f"\n  Created '{PDF_DIRECTORY}' directory")
        print("   Please add your PDF files and run again.\n")
        return
    
    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"\n  No PDF files found in '{PDF_DIRECTORY}'")
        print("   Please add your PDF files and run again.\n")
        return
    
    print(f"\n Found {len(pdf_files)} PDF file(s)")
    print("\n INSTRUCTIONS:")
    print("   1. The system will index all PDFs automatically")
    print("   2. Ask questions about your documents")
    print("   3. Type 'quit' to exit\n")
    
    # Initialize and index
    print(" Initializing RAG System...")
    rag = RAGSystem()
    
    if not rag.index_documents():
        print(" Failed to initialize system")
        return
    
    # Query loop
    print("\n READY FOR QUERIES - Start asking questions!")
    print("-" * 60)
    
    while True:
        try:
            user_query = input("\n Ask your question (or 'quit' to exit): ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("\n Goodbye!\n")
                break
            
            if not user_query:
                print("  Please enter a valid question")
                continue
            
            result = rag.query(user_query)
            RAGSystem.print_results(result)
            
        except KeyboardInterrupt:
            print("\n\n Interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n Error: {str(e)}\n")


if __name__ == "__main__":
    main()