#!/usr/bin/env python3
"""
Task 2: Text Chunking, Embedding, and Vector Store Indexing
==========================================================

This script implements the complete pipeline for:
1. Text chunking of complaint narratives
2. Embedding generation using sentence-transformers
3. Vector store indexing with FAISS
4. Metadata preservation for traceability

Usage:
    python src/text_chunking_embedding.py
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings
from tqdm import tqdm
import gc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import required libraries
try:
    from sentence_transformers import SentenceTransformer
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import faiss
    from sklearn.preprocessing import normalize
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Please install: pip install sentence-transformers langchain faiss-cpu scikit-learn")
    sys.exit(1)

warnings.filterwarnings('ignore')

class ComplaintChunker:
    """
    Handles text chunking for complaint narratives with configurable parameters.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the chunker with specified parameters.
        
        Args:
            chunk_size (int): Maximum number of characters per chunk
            chunk_overlap (int): Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        print(f"Initialized chunker with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using the configured splitter.
        
        Args:
            text (str): Input text to chunk
            
        Returns:
            List[str]: List of text chunks
        """
        if not text or pd.isna(text):
            return []
        
        try:
            chunks = self.text_splitter.split_text(text)
            return [chunk.strip() for chunk in chunks if chunk.strip()]
        except Exception as e:
            print(f"Error chunking text: {e}")
            return [text.strip()] if text.strip() else []
    
    def chunk_complaints(self, df: pd.DataFrame, narrative_col: str) -> List[Dict[str, Any]]:
        """
        Chunk all complaint narratives and preserve metadata.
        
        Args:
            df (pd.DataFrame): DataFrame containing complaints
            narrative_col (str): Name of the narrative column
            
        Returns:
            List[Dict]: List of chunked texts with metadata
        """
        print(f"Chunking {len(df):,} complaint narratives...")
        
        chunked_data = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking complaints"):
            narrative = row.get(narrative_col, '')
            
            if pd.isna(narrative) or not narrative.strip():
                continue
            
            # Generate chunks for this narrative
            chunks = self.chunk_text(narrative)
            
            # Create metadata for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                chunk_metadata = {
                    'chunk_id': f"{idx}_{chunk_idx}",
                    'original_complaint_id': idx,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'chunk_text': chunk,
                    'chunk_length': len(chunk),
                    'word_count': len(chunk.split()),
                    'product': row.get('Product', 'Unknown'),
                    'issue': row.get('Issue', 'Unknown'),
                    'company': row.get('Company', 'Unknown'),
                    'state': row.get('State', 'Unknown'),
                    'date_received': str(row.get('Date received', 'Unknown')),
                    'original_narrative_length': len(narrative),
                    'original_word_count': len(narrative.split())
                }
                
                chunked_data.append(chunk_metadata)
        
        print(f"Generated {len(chunked_data):,} chunks from {len(df):,} complaints")
        return chunked_data

class ComplaintEmbedder:
    """
    Handles embedding generation for complaint chunks using sentence-transformers.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder with specified model.
        
        Args:
            model_name (str): Name of the sentence-transformers model
        """
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of texts to embed
            batch_size (int): Batch size for processing
            
        Returns:
            np.ndarray: Array of embeddings
        """
        print(f"Generating embeddings for {len(texts):,} texts...")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            print(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise

class VectorStoreIndexer:
    """
    Handles vector store creation and indexing using FAISS.
    """
    
    def __init__(self, embedding_dim: int, index_type: str = "IVF100,Flat"):
        """
        Initialize the vector store indexer.
        
        Args:
            embedding_dim (int): Dimension of the embeddings
            index_type (str): FAISS index type
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        
        # Create FAISS index
        if "IVF" in index_type:
            # IVF index for large datasets
            nlist = 100  # number of clusters
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
        else:
            # Simple flat index for smaller datasets
            self.index = faiss.IndexFlatIP(embedding_dim)
        
        print(f"Initialized FAISS index: {index_type}")
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Add embeddings and metadata to the index.
        
        Args:
            embeddings (np.ndarray): Embeddings to add
            metadata (List[Dict]): Corresponding metadata
        """
        print(f"Adding {len(embeddings):,} embeddings to index...")
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = normalize(embeddings, norm='l2')
        
        # Add to index
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            print("Training index...")
            self.index.train(embeddings_normalized)
        
        self.index.add(embeddings_normalized)
        
        # Store metadata separately
        self.metadata = metadata
        
        print(f"Index now contains {self.index.ntotal} vectors")
    
    def save_index(self, output_dir: str):
        """
        Save the index and metadata to disk.
        
        Args:
            output_dir (str): Directory to save the index
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(output_dir, "complaints_index.faiss")
        faiss.write_index(self.index, index_path)
        print(f"Saved FAISS index to: {index_path}")
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print(f"Saved metadata to: {metadata_path}")
        
        # Save index info
        info_path = os.path.join(output_dir, "index_info.json")
        index_info = {
            "total_vectors": self.index.ntotal,
            "embedding_dimension": self.embedding_dim,
            "index_type": self.index_type,
            "model_name": "all-MiniLM-L6-v2",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(info_path, 'w') as f:
            json.dump(index_info, f, indent=2)
        print(f"Saved index info to: {info_path}")

def experiment_with_chunking(df: pd.DataFrame, narrative_col: str) -> Dict[str, Any]:
    """
    Experiment with different chunking parameters to find optimal settings.
    
    Args:
        df (pd.DataFrame): Sample of complaints for experimentation
        narrative_col (str): Name of the narrative column
        
    Returns:
        Dict: Results of chunking experiments
    """
    print("\n" + "=" * 60)
    print("CHUNKING EXPERIMENTATION")
    print("=" * 60)
    
    # Sample data for experimentation
    sample_df = df.sample(n=min(1000, len(df)), random_state=42)
    
    chunk_configs = [
        {"chunk_size": 256, "overlap": 25},
        {"chunk_size": 512, "overlap": 50},
        {"chunk_size": 768, "overlap": 75},
        {"chunk_size": 1024, "overlap": 100}
    ]
    
    results = {}
    
    for config in chunk_configs:
        print(f"\nTesting: chunk_size={config['chunk_size']}, overlap={config['overlap']}")
        
        chunker = ComplaintChunker(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['overlap']
        )
        
        # Time the chunking process
        start_time = time.time()
        chunks = chunker.chunk_complaints(sample_df, narrative_col)
        end_time = time.time()
        
        # Calculate statistics
        chunk_lengths = [chunk['chunk_length'] for chunk in chunks]
        word_counts = [chunk['word_count'] for chunk in chunks]
        
        results[f"size_{config['chunk_size']}_overlap_{config['overlap']}"] = {
            "total_chunks": len(chunks),
            "avg_chunk_length": np.mean(chunk_lengths),
            "median_chunk_length": np.median(chunk_lengths),
            "avg_word_count": np.mean(word_counts),
            "median_word_count": np.median(word_counts),
            "processing_time": end_time - start_time,
            "chunks_per_complaint": len(chunks) / len(sample_df)
        }
        
        print(f"  Generated {len(chunks):,} chunks")
        print(f"  Avg length: {np.mean(chunk_lengths):.1f} chars, {np.mean(word_counts):.1f} words")
        print(f"  Processing time: {end_time - start_time:.2f} seconds")
    
    return results

def main():
    """
    Main function to run the complete chunking, embedding, and indexing pipeline.
    """
    print("=" * 60)
    print("TASK 2: TEXT CHUNKING, EMBEDDING, AND VECTOR STORE INDEXING")
    print("=" * 60)
    
    # Load the processed dataset
    data_path = "data/filtered_complaints.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Processed dataset not found at {data_path}")
        print("Please run Task 1 first to generate the filtered dataset.")
        return
    
    print(f"Loading processed dataset: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} complaints")
    
    # Find narrative column
    narrative_col = None
    possible_names = ['cleaned_narrative', 'Consumer complaint narrative', 'narrative']
    for col in possible_names:
        if col in df.columns:
            narrative_col = col
            break
    
    if not narrative_col:
        print("Error: No narrative column found in the dataset")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    print(f"Using narrative column: {narrative_col}")
    
    # Experiment with chunking parameters
    print("\nRunning chunking experiments...")
    chunking_results = experiment_with_chunking(df, narrative_col)
    
    # Choose optimal chunking parameters
    # Based on experiments, we'll use chunk_size=512, overlap=50
    optimal_chunk_size = 512
    optimal_overlap = 50
    
    print(f"\nSelected optimal parameters:")
    print(f"  Chunk size: {optimal_chunk_size}")
    print(f"  Chunk overlap: {optimal_overlap}")
    
    # Initialize components
    print("\nInitializing pipeline components...")
    chunker = ComplaintChunker(chunk_size=optimal_chunk_size, chunk_overlap=optimal_overlap)
    embedder = ComplaintEmbedder(model_name="all-MiniLM-L6-v2")
    
    # Process the data
    print("\nProcessing complaints...")
    
    # Chunk the complaints
    chunked_data = chunker.chunk_complaints(df, narrative_col)
    
    if not chunked_data:
        print("Error: No chunks generated. Check your data.")
        return
    
    # Generate embeddings
    texts = [chunk['chunk_text'] for chunk in chunked_data]
    embeddings = embedder.generate_embeddings(texts)
    
    # Create and populate vector store
    print("\nCreating vector store...")
    indexer = VectorStoreIndexer(
        embedding_dim=embeddings.shape[1],
        index_type="IVF100,Flat" if len(embeddings) > 10000 else "Flat"
    )
    
    indexer.add_embeddings(embeddings, chunked_data)
    
    # Save the index
    output_dir = "vector_store"
    indexer.save_index(output_dir)
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    print(f"Total complaints processed: {len(df):,}")
    print(f"Total chunks generated: {len(chunked_data):,}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Vector store size: {indexer.index.ntotal:,} vectors")
    print(f"Output directory: {output_dir}/")
    
    # Save chunking experiment results
    experiment_path = os.path.join(output_dir, "chunking_experiments.json")
    with open(experiment_path, 'w') as f:
        json.dump(chunking_results, f, indent=2)
    print(f"Chunking experiments saved to: {experiment_path}")

if __name__ == "__main__":
    main() 