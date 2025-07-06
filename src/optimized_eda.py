#!/usr/bin/env python3
"""
Optimized EDA Script for Large CFPB Dataset
===========================================

This script provides optimized processing for large complaint datasets
with progress tracking and memory-efficient operations.

Usage:
    python src/optimized_eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
import os
import sys
from pathlib import Path
from tqdm import tqdm
import gc

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)

def load_data_in_chunks(file_path, chunk_size=10000):
    """
    Load large dataset in chunks to manage memory usage.
    
    Args:
        file_path (str): Path to the CSV file
        chunk_size (int): Number of rows to process at once
    
    Yields:
        pd.DataFrame: Chunk of data
    """
    print(f"Loading data in chunks of {chunk_size:,} rows...")
    
    # First, get total number of rows for progress tracking
    total_rows = sum(1 for _ in open(file_path, 'r', encoding='utf-8')) - 1  # Subtract header
    print(f"Total rows in dataset: {total_rows:,}")
    
    # Load in chunks
    chunk_reader = pd.read_csv(file_path, chunksize=chunk_size)
    
    for chunk_num, chunk in enumerate(chunk_reader):
        print(f"Processing chunk {chunk_num + 1} ({len(chunk):,} rows)")
        yield chunk
        gc.collect()  # Force garbage collection

def analyze_narratives_optimized(df, narrative_col):
    """
    Optimized narrative analysis with progress tracking.
    
    Args:
        df (pd.DataFrame): DataFrame containing narratives
        narrative_col (str): Name of the narrative column
    
    Returns:
        dict: Analysis results
    """
    print(f"\nAnalyzing narratives in column: '{narrative_col}'")
    
    # Count non-null narratives
    total_complaints = len(df)
    complaints_with_narratives = df[narrative_col].notna().sum()
    complaints_without_narratives = total_complaints - complaints_with_narratives
    
    print(f"Total complaints: {total_complaints:,}")
    print(f"With narratives: {complaints_with_narratives:,} ({complaints_with_narratives/total_complaints*100:.1f}%)")
    print(f"Without narratives: {complaints_without_narratives:,} ({complaints_without_narratives/total_complaints*100:.1f}%)")
    
    # Calculate word counts efficiently
    print("Calculating word counts...")
    narratives_df = df[df[narrative_col].notna()].copy()
    
    # Use vectorized operations for better performance
    word_counts = narratives_df[narrative_col].str.split().str.len()
    
    # Calculate statistics
    stats = {
        'total_complaints': total_complaints,
        'with_narratives': complaints_with_narratives,
        'without_narratives': complaints_without_narratives,
        'word_count_stats': word_counts.describe(),
        'short_narratives': len(word_counts[word_counts <= 10]),
        'long_narratives': len(word_counts[word_counts >= 500]),
        'word_counts': word_counts
    }
    
    return stats

def process_large_dataset_optimized(file_path):
    """
    Process large dataset with memory optimization and progress tracking.
    """
    print("=" * 60)
    print("OPTIMIZED LARGE DATASET PROCESSING")
    print("=" * 60)
    
    # Initialize counters
    total_stats = {
        'total_rows': 0,
        'product_counts': {},
        'narrative_stats': {
            'total': 0,
            'with_narratives': 0,
            'without_narratives': 0,
            'word_counts': []
        }
    }
    
    # Find column names from first chunk
    print("Identifying column structure...")
    first_chunk = next(pd.read_csv(file_path, chunksize=1000))
    
    # Find narrative and product columns
    narrative_col = None
    product_col = None
    
    possible_narrative_names = ['Consumer complaint narrative', 'consumer_complaint_narrative', 'narrative', 'Narrative']
    possible_product_names = ['Product', 'product']
    
    for col in first_chunk.columns:
        if col in possible_narrative_names:
            narrative_col = col
        if col in possible_product_names:
            product_col = col
    
    print(f"Narrative column: {narrative_col}")
    print(f"Product column: {product_col}")
    
    # Process data in chunks
    chunk_size = 5000  # Smaller chunks for better memory management
    
    for chunk in load_data_in_chunks(file_path, chunk_size):
        total_stats['total_rows'] += len(chunk)
        
        # Update product counts
        if product_col and product_col in chunk.columns:
            chunk_products = chunk[product_col].value_counts()
            for product, count in chunk_products.items():
                if pd.isna(product):
                    continue
                total_stats['product_counts'][product] = total_stats['product_counts'].get(product, 0) + count
        
        # Update narrative statistics
        if narrative_col and narrative_col in chunk.columns:
            total_stats['narrative_stats']['total'] += len(chunk)
            total_stats['narrative_stats']['with_narratives'] += chunk[narrative_col].notna().sum()
            total_stats['narrative_stats']['without_narratives'] += chunk[narrative_col].isna().sum()
            
            # Calculate word counts for non-null narratives
            narratives = chunk[chunk[narrative_col].notna()][narrative_col]
            if len(narratives) > 0:
                word_counts = narratives.str.split().str.len()
                total_stats['narrative_stats']['word_counts'].extend(word_counts.tolist())
        
        # Memory cleanup
        del chunk
        gc.collect()
        
        # Progress update
        if total_stats['total_rows'] % 50000 == 0:
            print(f"Processed {total_stats['total_rows']:,} rows...")
    
    return total_stats, narrative_col, product_col

def generate_quick_summary(stats, narrative_col, product_col):
    """
    Generate a quick summary of the analysis results.
    """
    print("\n" + "=" * 60)
    print("QUICK ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\n1. DATASET OVERVIEW:")
    print(f"   - Total records processed: {stats['total_rows']:,}")
    
    if narrative_col:
        narrative_stats = stats['narrative_stats']
        print(f"\n2. NARRATIVE ANALYSIS:")
        print(f"   - Records with narratives: {narrative_stats['with_narratives']:,} ({narrative_stats['with_narratives']/narrative_stats['total']*100:.1f}%)")
        print(f"   - Records without narratives: {narrative_stats['without_narratives']:,} ({narrative_stats['without_narratives']/narrative_stats['total']*100:.1f}%)")
        
        if narrative_stats['word_counts']:
            word_counts = np.array(narrative_stats['word_counts'])
            print(f"   - Average narrative length: {word_counts.mean():.1f} words")
            print(f"   - Median narrative length: {np.median(word_counts):.1f} words")
            print(f"   - Shortest narrative: {word_counts.min()} words")
            print(f"   - Longest narrative: {word_counts.max()} words")
            print(f"   - Very short narratives (≤10 words): {len(word_counts[word_counts <= 10]):,}")
            print(f"   - Very long narratives (≥500 words): {len(word_counts[word_counts >= 500]):,}")
    
    if product_col and stats['product_counts']:
        print(f"\n3. PRODUCT ANALYSIS:")
        print(f"   - Total unique products: {len(stats['product_counts'])}")
        print(f"   - Top 5 products by complaint volume:")
        
        sorted_products = sorted(stats['product_counts'].items(), key=lambda x: x[1], reverse=True)
        for i, (product, count) in enumerate(sorted_products[:5]):
            print(f"     {i+1}. {product}: {count:,} complaints")
    
    print(f"\n4. RECOMMENDATIONS:")
    print(f"   - Dataset size: {'Large' if stats['total_rows'] > 1000000 else 'Medium' if stats['total_rows'] > 100000 else 'Small'}")
    print(f"   - Memory usage: {'High' if stats['total_rows'] > 500000 else 'Medium' if stats['total_rows'] > 100000 else 'Low'}")
    print(f"   - Processing approach: {'Chunked processing recommended' if stats['total_rows'] > 100000 else 'Full dataset processing possible'}")

def main():
    """
    Main function for optimized EDA processing.
    """
    file_path = 'data/complaints.csv'
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return
    
    print("Starting optimized EDA processing...")
    print(f"File size: {os.path.getsize(file_path) / (1024**3):.2f} GB")
    
    try:
        # Process the dataset
        stats, narrative_col, product_col = process_large_dataset_optimized(file_path)
        
        # Generate summary
        generate_quick_summary(stats, narrative_col, product_col)
        
        print("\n" + "=" * 60)
        print("OPTIMIZED EDA COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        print("Consider reducing chunk_size or using a smaller sample for testing.")

if __name__ == "__main__":
    main()