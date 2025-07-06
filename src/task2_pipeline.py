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
    python src/task2_pipeline.py
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import warnings
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

warnings.filterwarnings('ignore')

def load_processed_data():
    """Load the processed dataset from Task 1."""
    data_path = "data/filtered_complaints.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Processed dataset not found at {data_path}")
        print("Please run Task 1 first to generate the filtered dataset.")
        return None, None
    
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
        return None, None
    
    print(f"Using narrative column: {narrative_col}")
    return df, narrative_col

def chunk_complaints(df, narrative_col, chunk_size=512, chunk_overlap=50):
    """
    Chunk complaint narratives using LangChain's RecursiveCharacterTextSplitter.
    
    Args:
        df: DataFrame containing complaints
        narrative_col: Name of the narrative column
        chunk_size: Maximum characters per chunk
        chunk_overlap: Character overlap between chunks
    
    Returns:
        List of chunked texts with metadata
    """
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        print("Error: langchain not installed. Please run: pip install langchain")
        return []
    
    print(f"Chunking {len(df):,} complaint narratives...")
    print(f"Parameters: chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    
    chunked_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking complaints"):
        narrative = row.get(narrative_col, '')
        
        if pd.isna(narrative) or not narrative.strip():
            continue
        
        # Generate chunks for this narrative
        try:
            chunks = text_splitter.split_text(narrative)
            chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        except Exception as e:
            print(f"Error chunking complaint {idx}: {e}")
            chunks = [narrative.strip()] if narrative.strip() else []
        
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

def generate_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    """
    Generate embeddings using sentence-transformers.
    
    Args:
        texts: List of texts to embed
        model_name: Name of the sentence-transformers model
    
    Returns:
        Array of embeddings
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.preprocessing import normalize
    except ImportError:
        print("Error: sentence-transformers not installed. Please run: pip install sentence-transformers scikit-learn")
        return None
    
    print(f"Loading embedding model: {model_name}")
    
    try:
        model = SentenceTransformer(model_name)
        print(f"Model loaded successfully. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    print(f"Generating embeddings for {len(texts):,} texts...")
    
    try:
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        print(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def create_vector_store(embeddings, metadata, output_dir="vector_store"):
    """
    Create and save FAISS vector store with metadata.
    
    Args:
        embeddings: Array of embeddings
        metadata: List of metadata dictionaries
        output_dir: Directory to save the index
    """
    try:
        import faiss
        from sklearn.preprocessing import normalize
    except ImportError:
        print("Error: faiss-cpu not installed. Please run: pip install faiss-cpu")
        return False
    
    print(f"Creating vector store with {len(embeddings):,} embeddings...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalize embeddings for cosine similarity
    embeddings_normalized = normalize(embeddings, norm='l2')
    
    # Create FAISS index
    embedding_dim = embeddings.shape[1]
    
    if len(embeddings) > 10000:
        # Use IVF index for large datasets
        nlist = min(100, len(embeddings) // 100)  # number of clusters
        quantizer = faiss.IndexFlatIP(embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
        print(f"Using IVF index with {nlist} clusters")
    else:
        # Use flat index for smaller datasets
        index = faiss.IndexFlatIP(embedding_dim)
        print("Using flat index")
    
    # Train and add embeddings
    if hasattr(index, 'is_trained') and not index.is_trained:
        print("Training index...")
        index.train(embeddings_normalized)
    
    index.add(embeddings_normalized)
    
    # Save FAISS index
    index_path = os.path.join(output_dir, "complaints_index.faiss")
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to: {index_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata to: {metadata_path}")
    
    # Save index info
    info_path = os.path.join(output_dir, "index_info.json")
    index_info = {
        "total_vectors": index.ntotal,
        "embedding_dimension": embedding_dim,
        "index_type": "IVF100,Flat" if len(embeddings) > 10000 else "Flat",
        "model_name": "all-MiniLM-L6-v2",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(info_path, 'w') as f:
        json.dump(index_info, f, indent=2)
    print(f"Saved index info to: {info_path}")
    
    return True

def main():
    """Main function to run the complete pipeline."""
    print("=" * 60)
    print("TASK 2: TEXT CHUNKING, EMBEDDING, AND VECTOR STORE INDEXING")
    print("=" * 60)
    
    # Load processed data
    df, narrative_col = load_processed_data()
    if df is None:
        return
    
    # Chunk the complaints
    print("\nStep 1: Chunking complaint narratives...")
    chunked_data = chunk_complaints(df, narrative_col)
    
    if not chunked_data:
        print("Error: No chunks generated. Check your data.")
        return
    
    # Generate embeddings
    print("\nStep 2: Generating embeddings...")
    texts = [chunk['chunk_text'] for chunk in chunked_data]
    embeddings = generate_embeddings(texts)
    
    if embeddings is None:
        print("Error: Failed to generate embeddings.")
        return
    
    # Create vector store
    print("\nStep 3: Creating vector store...")
    success = create_vector_store(embeddings, chunked_data)
    
    if not success:
        print("Error: Failed to create vector store.")
        return
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    print(f"Total complaints processed: {len(df):,}")
    print(f"Total chunks generated: {len(chunked_data):,}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Vector store saved to: vector_store/")
    
    # Calculate statistics
    chunk_lengths = [chunk['chunk_length'] for chunk in chunked_data]
    word_counts = [chunk['word_count'] for chunk in chunked_data]
    
    print(f"\nChunking Statistics:")
    print(f"  Average chunk length: {np.mean(chunk_lengths):.1f} characters")
    print(f"  Average word count: {np.mean(word_counts):.1f} words")
    print(f"  Chunks per complaint: {len(chunked_data) / len(df):.2f}")

if __name__ == "__main__":
    main() 