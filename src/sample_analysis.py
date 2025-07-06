#!/usr/bin/env python3
"""
Sample Analysis for Quick Testing
=================================

This script analyzes a sample of the large dataset for quick testing
and development purposes.

Usage:
    python src/sample_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

def analyze_sample_data(file_path, sample_size=10000):
    """
    Analyze a sample of the large dataset for quick testing.
    
    Args:
        file_path (str): Path to the CSV file
        sample_size (int): Number of rows to sample
    """
    print("=" * 60)
    print(f"SAMPLE ANALYSIS ({sample_size:,} ROWS)")
    print("=" * 60)
    
    # Load a sample of the data
    print(f"Loading sample of {sample_size:,} rows...")
    
    # Use nrows parameter for quick sampling
    df = pd.read_csv(file_path, nrows=sample_size)
    
    print(f"Sample loaded: {len(df):,} rows")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Basic info
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Find narrative and product columns
    narrative_col = None
    product_col = None
    
    possible_narrative_names = ['Consumer complaint narrative', 'consumer_complaint_narrative', 'narrative', 'Narrative']
    possible_product_names = ['Product', 'product']
    
    for col in df.columns:
        if col in possible_narrative_names:
            narrative_col = col
        if col in possible_product_names:
            product_col = col
    
    print(f"Narrative column: {narrative_col}")
    print(f"Product column: {product_col}")
    
    # Analyze products
    if product_col:
        print(f"\nProduct distribution (sample):")
        product_counts = df[product_col].value_counts()
        print(product_counts.head(10))
        
        # Visualize
        plt.figure(figsize=(12, 6))
        product_counts.head(10).plot(kind='bar', color='skyblue')
        plt.title('Top 10 Products by Complaint Volume (Sample)', fontweight='bold')
        plt.xlabel('Product')
        plt.ylabel('Number of Complaints')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('reports/sample_product_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Analyze narratives
    if narrative_col:
        print(f"\nNarrative analysis (sample):")
        
        # Count with/without narratives
        total = len(df)
        with_narratives = df[narrative_col].notna().sum()
        without_narratives = total - with_narratives
        
        print(f"Total: {total:,}")
        print(f"With narratives: {with_narratives:,} ({with_narratives/total*100:.1f}%)")
        print(f"Without narratives: {without_narratives:,} ({without_narratives/total*100:.1f}%)")
        
        # Word count analysis (only for non-null narratives)
        narratives_df = df[df[narrative_col].notna()].copy()
        
        if len(narratives_df) > 0:
            print(f"\nCalculating word counts for {len(narratives_df):,} narratives...")
            
            # Use vectorized operations
            word_counts = narratives_df[narrative_col].str.split().str.len()
            
            print("Word count statistics:")
            print(word_counts.describe())
            
            # Visualize word count distribution
            plt.figure(figsize=(15, 10))
            
            # Histogram
            plt.subplot(2, 2, 1)
            plt.hist(word_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Word Count Distribution (Sample)', fontweight='bold')
            plt.xlabel('Word Count')
            plt.ylabel('Frequency')
            
            # Box plot
            plt.subplot(2, 2, 2)
            plt.boxplot(word_counts)
            plt.title('Word Count Box Plot (Sample)', fontweight='bold')
            plt.ylabel('Word Count')
            
            # Log scale histogram
            plt.subplot(2, 2, 3)
            plt.hist(word_counts, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.yscale('log')
            plt.title('Word Count Distribution - Log Scale (Sample)', fontweight='bold')
            plt.xlabel('Word Count')
            plt.ylabel('Frequency (Log Scale)')
            
            # Cumulative distribution
            plt.subplot(2, 2, 4)
            sorted_counts = np.sort(word_counts)
            cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
            plt.plot(sorted_counts, cumulative, color='red', linewidth=2)
            plt.title('Cumulative Distribution (Sample)', fontweight='bold')
            plt.xlabel('Word Count')
            plt.ylabel('Cumulative Probability')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('reports/sample_word_count_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Identify very short and very long narratives
            short_threshold = 10
            long_threshold = 500
            
            very_short = word_counts[word_counts <= short_threshold]
            very_long = word_counts[word_counts >= long_threshold]
            
            print(f"\nNarratives with <= {short_threshold} words: {len(very_short):,} ({len(very_short)/len(word_counts)*100:.1f}%)")
            print(f"Narratives with >= {long_threshold} words: {len(very_long):,} ({len(very_long)/len(word_counts)*100:.1f}%)")
            
            # Show examples
            if len(very_short) > 0:
                print(f"\nExamples of very short narratives (<= {short_threshold} words):")
                short_narratives = narratives_df[narratives_df[narrative_col].str.split().str.len() <= short_threshold]
                for i, narrative in enumerate(short_narratives[narrative_col].head(3)):
                    print(f"{i+1}. {narrative}")
            
            if len(very_long) > 0:
                print(f"\nExamples of very long narratives (>= {long_threshold} words, first 200 chars):")
                long_narratives = narratives_df[narratives_df[narrative_col].str.split().str.len() >= long_threshold]
                for i, narrative in enumerate(long_narratives[narrative_col].head(2)):
                    print(f"{i+1}. {str(narrative)[:200]}...")
    
    # Save sample for further analysis
    sample_output = 'data/complaints_sample.csv'
    df.to_csv(sample_output, index=False)
    print(f"\nSample saved to: {sample_output}")
    
    return df, narrative_col, product_col

def main():
    """
    Main function for sample analysis.
    """
    file_path = 'data/complaints.csv'
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return
    
    # Analyze sample
    sample_size = 10000  # Adjust this for faster/slower processing
    df, narrative_col, product_col = analyze_sample_data(file_path, sample_size)
    
    print("\n" + "=" * 60)
    print("SAMPLE ANALYSIS COMPLETED")
    print("=" * 60)
    print(f"Sample size: {len(df):,} rows")
    print(f"Use this sample for development and testing")
    print(f"Full dataset analysis will take much longer")

if __name__ == "__main__":
    main() 