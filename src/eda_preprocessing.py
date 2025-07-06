#!/usr/bin/env python3
"""
Task 1: Exploratory Data Analysis and Data Preprocessing
=======================================================

This script performs comprehensive EDA on the CFPB complaints dataset and 
preprocesses it for the CrediTrust Financial complaint-answering chatbot.

Usage:
    python src/eda_preprocessing.py

Output:
    - Comprehensive EDA report
    - Filtered and cleaned dataset saved to data/filtered_complaints.csv
    - Visualization plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import warnings
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)

def clean_narrative_text(text):
    """
    Clean narrative text for better embedding quality.
    
    Args:
        text (str): Raw narrative text
    
    Returns:
        str: Cleaned narrative text
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to string
    text = str(text)
    
    # Remove boilerplate text (common complaint opening phrases)
    boilerplate_patterns = [
        r'I am writing to file a complaint.*?about',
        r'I am filing this complaint.*?regarding',
        r'I would like to file a complaint.*?about',
        r'This is a complaint about.*?regarding',
        r'I am submitting this complaint.*?concerning'
    ]
    
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = text.strip()
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\$\%\:]', ' ', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[\.\.\.\.\.\\.]+', '.', text)  # Multiple dots to single dot
    text = re.sub(r'[\!\!\!]+', '!', text)  # Multiple exclamation marks
    text = re.sub(r'[\?\?\?]+', '?', text)  # Multiple question marks
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace again
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_and_explore_data(file_path):
    """Load and perform initial exploration of the dataset."""
    print("=" * 60)
    print("LOADING AND EXPLORING DATASET")
    print("=" * 60)
    
    # Load the dataset
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Basic information
    print("\nDataset Info:")
    print(df.info())
    
    print("\nColumn names:")
    print(df.columns.tolist())
    
    # Check for missing values
    print("\nMissing values analysis:")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing Percentage': missing_percentage
    }).sort_values('Missing Count', ascending=False)
    print(missing_df[missing_df['Missing Count'] > 0])
    
    return df

def analyze_products(df):
    """Analyze product distribution."""
    print("\n" + "=" * 60)
    print("PRODUCT DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Find product column
    product_col = 'Product' if 'Product' in df.columns else 'product'
    
    if product_col in df.columns:
        print("Product distribution:")
        product_counts = df[product_col].value_counts()
        print(product_counts)
        
        # Visualize product distribution
        plt.figure(figsize=(14, 8))
        ax = product_counts.plot(kind='bar', color='skyblue')
        plt.title('Distribution of Complaints by Product', fontsize=16, fontweight='bold')
        plt.xlabel('Product', fontsize=12)
        plt.ylabel('Number of Complaints', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add value labels on bars
        for i, v in enumerate(product_counts.values):
            ax.text(i, v + max(product_counts.values) * 0.01, str(v), ha='center', va='bottom')
        
        plt.savefig('reports/product_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return product_col
    else:
        print("Product column not found in the dataset")
        print("Available columns:", df.columns.tolist())
        return None

def analyze_narratives(df):
    """Analyze complaint narratives."""
    print("\n" + "=" * 60)
    print("NARRATIVE ANALYSIS")
    print("=" * 60)
    
    # Find narrative column
    narrative_col = None
    possible_names = ['Consumer complaint narrative', 'consumer_complaint_narrative', 
                     'narrative', 'Narrative']
    for col in possible_names:
        if col in df.columns:
            narrative_col = col
            break
    
    if narrative_col:
        print(f"Found narrative column: '{narrative_col}'")
        
        # Count complaints with and without narratives
        total_complaints = len(df)
        complaints_with_narratives = df[narrative_col].notna().sum()
        complaints_without_narratives = total_complaints - complaints_with_narratives
        
        print(f"\nNarrative Statistics:")
        print(f"Total complaints: {total_complaints:,}")
        print(f"Complaints with narratives: {complaints_with_narratives:,} ({complaints_with_narratives/total_complaints*100:.1f}%)")
        print(f"Complaints without narratives: {complaints_without_narratives:,} ({complaints_without_narratives/total_complaints*100:.1f}%)")
        
        # Analyze narrative word counts
        narratives_df = df[df[narrative_col].notna()].copy()
        narratives_df['word_count'] = narratives_df[narrative_col].str.split().str.len()
        
        print("\nWord count statistics for narratives:")
        print(narratives_df['word_count'].describe())
        
        # Visualize word count distribution
        plt.figure(figsize=(15, 10))
        
        # Histogram
        plt.subplot(2, 2, 1)
        plt.hist(narratives_df['word_count'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Word Counts in Narratives', fontweight='bold')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        
        # Box plot
        plt.subplot(2, 2, 2)
        plt.boxplot(narratives_df['word_count'])
        plt.title('Box Plot of Word Counts', fontweight='bold')
        plt.ylabel('Word Count')
        
        # Log scale histogram
        plt.subplot(2, 2, 3)
        plt.hist(narratives_df['word_count'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.yscale('log')
        plt.title('Distribution of Word Counts (Log Scale)', fontweight='bold')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency (Log Scale)')
        
        # Cumulative distribution
        plt.subplot(2, 2, 4)
        sorted_counts = np.sort(narratives_df['word_count'])
        cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
        plt.plot(sorted_counts, cumulative, color='red', linewidth=2)
        plt.title('Cumulative Distribution of Word Counts', fontweight='bold')
        plt.xlabel('Word Count')
        plt.ylabel('Cumulative Probability')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/narrative_word_counts.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Identify very short and very long narratives
        short_threshold = 10
        long_threshold = 500
        
        very_short = narratives_df[narratives_df['word_count'] <= short_threshold]
        very_long = narratives_df[narratives_df['word_count'] >= long_threshold]
        
        print(f"\nNarratives with <= {short_threshold} words: {len(very_short):,} ({len(very_short)/len(narratives_df)*100:.1f}%)")
        print(f"Narratives with >= {long_threshold} words: {len(very_long):,} ({len(very_long)/len(narratives_df)*100:.1f}%)")
        
        return narrative_col
    else:
        print("Narrative column not found in the dataset")
        print("Available columns:", df.columns.tolist())
        return None

def filter_dataset(df, product_col, narrative_col):
    """Filter dataset for target products and valid narratives."""
    print("\n" + "=" * 60)
    print("FILTERING DATASET")
    print("=" * 60)
    
    # Define target products
    target_products = [
        'Credit card',
        'Personal loan', 
        'Buy Now, Pay Later (BNPL)',
        'Savings account',
        'Money transfers'
    ]
    
    print("Target products for CrediTrust Financial:")
    for i, product in enumerate(target_products, 1):
        print(f"{i}. {product}")
    
    if product_col in df.columns:
        print(f"\nActual products in dataset:")
        actual_products = df[product_col].unique()
        for product in sorted(actual_products):
            if pd.isna(product):
                continue
            print(f"- {product}")
        
        # Try to match target products with actual products
        matched_products = []
        for target in target_products:
            target_lower = target.lower()
            for actual in actual_products:
                if pd.isna(actual):
                    continue
                actual_lower = str(actual).lower()
                if target_lower in actual_lower or actual_lower in target_lower:
                    matched_products.append(actual)
                    print(f"Matched '{target}' with '{actual}'")
                    break
        
        print(f"\nMatched products: {len(matched_products)}")
        
        if len(matched_products) > 0:
            # Filter dataset
            filtered_df = df[df[product_col].isin(matched_products)].copy()
            print(f"After product filtering: {len(filtered_df):,} records")
            
            # Remove records with empty narratives
            if narrative_col:
                filtered_df = filtered_df[filtered_df[narrative_col].notna()].copy()
                filtered_df = filtered_df[filtered_df[narrative_col].str.strip() != ''].copy()
                print(f"After removing empty narratives: {len(filtered_df):,} records")
            
            return filtered_df, matched_products
        else:
            print("No products matched - returning original dataset")
            return df.copy(), []
    else:
        print("Product column not found - returning original dataset")
        return df.copy(), []

def preprocess_text(df, narrative_col):
    """Apply text cleaning and preprocessing."""
    print("\n" + "=" * 60)
    print("TEXT PREPROCESSING")
    print("=" * 60)
    
    if narrative_col and len(df) > 0:
        print("Applying text cleaning to narratives...")
        
        # Show examples before cleaning
        print("\nExamples before cleaning:")
        for i, narrative in enumerate(df[narrative_col].head(3)):
            print(f"{i+1}. {str(narrative)[:200]}...")
        
        # Apply cleaning
        df['cleaned_narrative'] = df[narrative_col].apply(clean_narrative_text)
        
        # Show examples after cleaning
        print("\nExamples after cleaning:")
        for i, narrative in enumerate(df['cleaned_narrative'].head(3)):
            print(f"{i+1}. {str(narrative)[:200]}...")
        
        # Remove empty narratives after cleaning
        initial_count = len(df)
        df = df[df['cleaned_narrative'].str.strip() != ''].copy()
        final_count = len(df)
        
        print(f"\nRecords removed due to empty narratives after cleaning: {initial_count - final_count}")
        print(f"Final dataset size: {final_count:,} records")
        
        # Calculate word count statistics after cleaning
        df['cleaned_word_count'] = df['cleaned_narrative'].str.split().str.len()
        
        print("\nWord count statistics after cleaning:")
        print(df['cleaned_word_count'].describe())
        
        return df
    else:
        print("Cannot apply text cleaning - missing narrative column or empty dataset")
        return df

def save_processed_data(df, product_col, narrative_col):
    """Save the processed dataset."""
    print("\n" + "=" * 60)
    print("SAVING PROCESSED DATASET")
    print("=" * 60)
    
    if len(df) > 0:
        # Select relevant columns
        columns_to_keep = []
        
        if product_col in df.columns:
            columns_to_keep.append(product_col)
        
        if narrative_col in df.columns:
            columns_to_keep.append(narrative_col)
        
        if 'cleaned_narrative' in df.columns:
            columns_to_keep.append('cleaned_narrative')
        
        # Add other useful columns if they exist
        potential_columns = [
            'Date received', 'date_received', 'Date',
            'Issue', 'issue',
            'Sub-issue', 'sub_issue',
            'Company', 'company',
            'State', 'state',
            'Complaint ID', 'complaint_id', 'id'
        ]
        
        for col in potential_columns:
            if col in df.columns:
                columns_to_keep.append(col)
        
        # Remove duplicates and select available columns
        columns_to_keep = list(set(columns_to_keep))
        available_columns = [col for col in columns_to_keep if col in df.columns]
        
        final_df = df[available_columns].copy()
        
        # Save to CSV
        output_path = 'data/filtered_complaints.csv'
        final_df.to_csv(output_path, index=False)
        print(f"Processed dataset saved to: {output_path}")
        print(f"Final dataset shape: {final_df.shape}")
        print(f"Columns: {final_df.columns.tolist()}")
        
        return final_df
    else:
        print("No data to save - dataset is empty")
        return pd.DataFrame()

def generate_summary_report(original_df, filtered_df, product_col, narrative_col, matched_products):
    """Generate comprehensive summary report."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE EDA SUMMARY REPORT")
    print("=" * 60)
    
    if len(original_df) > 0:
        print(f"\n1. DATASET OVERVIEW:")
        print(f"   - Original dataset size: {len(original_df):,} records")
        print(f"   - Total columns: {len(original_df.columns)}")
        print(f"   - Memory usage: {original_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        if narrative_col:
            narratives_with_data = original_df[original_df[narrative_col].notna()]
            print(f"\n2. NARRATIVE ANALYSIS:")
            print(f"   - Records with narratives: {len(narratives_with_data):,} ({len(narratives_with_data)/len(original_df)*100:.1f}%)")
            print(f"   - Records without narratives: {len(original_df) - len(narratives_with_data):,} ({(len(original_df) - len(narratives_with_data))/len(original_df)*100:.1f}%)")
            
            if len(narratives_with_data) > 0:
                word_counts = narratives_with_data[narrative_col].str.split().str.len()
                print(f"   - Average narrative length: {word_counts.mean():.1f} words")
                print(f"   - Median narrative length: {word_counts.median():.1f} words")
                print(f"   - Shortest narrative: {word_counts.min()} words")
                print(f"   - Longest narrative: {word_counts.max()} words")
        
        if product_col in original_df.columns:
            print(f"\n3. PRODUCT ANALYSIS:")
            product_counts = original_df[product_col].value_counts()
            print(f"   - Total unique products: {len(product_counts)}")
            print(f"   - Top 3 products by complaint volume:")
            for i, (product, count) in enumerate(product_counts.head(3).items()):
                print(f"     {i+1}. {product}: {count:,} complaints")
        
        if len(filtered_df) > 0:
            print(f"\n4. FILTERED DATASET (FOR RAG PIPELINE):")
            print(f"   - Final dataset size: {len(filtered_df):,} records")
            print(f"   - Data reduction: {(1 - len(filtered_df)/len(original_df))*100:.1f}% of original data removed")
            print(f"   - Target products included: {len(matched_products)}")
            
            if 'cleaned_word_count' in filtered_df.columns:
                print(f"   - Average cleaned narrative length: {filtered_df['cleaned_word_count'].mean():.1f} words")
                print(f"   - Median cleaned narrative length: {filtered_df['cleaned_word_count'].median():.1f} words")
        
        print(f"\n5. DATA QUALITY INSIGHTS:")
        if narrative_col:
            narratives_with_data = original_df[original_df[narrative_col].notna()]
            if len(narratives_with_data) > 0:
                short_narratives = len(narratives_with_data[narratives_with_data[narrative_col].str.split().str.len() <= 10])
                long_narratives = len(narratives_with_data[narratives_with_data[narrative_col].str.split().str.len() >= 500])
                print(f"   - Very short narratives (≤10 words): {short_narratives:,} ({short_narratives/len(narratives_with_data)*100:.1f}%)")
                print(f"   - Very long narratives (≥500 words): {long_narratives:,} ({long_narratives/len(narratives_with_data)*100:.1f}%)")
        
        missing_data = original_df.isnull().sum().sum()
        total_cells = len(original_df) * len(original_df.columns)
        print(f"   - Overall missing data: {missing_data:,} cells ({missing_data/total_cells*100:.1f}% of all data)")
        
        print(f"\n6. RECOMMENDATIONS FOR RAG PIPELINE:")
        print(f"   - Dataset is ready for embedding generation")
        print(f"   - Text cleaning successfully applied")
        print(f"   - Product filtering completed for target domains")
        print(f"   - Consider chunking very long narratives for better embedding quality")
        print(f"   - Implement semantic search on cleaned narratives for optimal retrieval")
    
    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)

def main():
    """Main function to run the complete EDA pipeline."""
    # Ensure directories exist
    os.makedirs('reports', exist_ok=True)
    
    # Load and explore data
    df = load_and_explore_data('data/complaints.csv')
    
    # Analyze products
    product_col = analyze_products(df)
    
    # Analyze narratives
    narrative_col = analyze_narratives(df)
    
    # Filter dataset
    filtered_df, matched_products = filter_dataset(df, product_col, narrative_col)
    
    # Preprocess text
    processed_df = preprocess_text(filtered_df, narrative_col)
    
    # Save processed data
    final_df = save_processed_data(processed_df, product_col, narrative_col)
    
    # Generate summary report
    generate_summary_report(df, final_df, product_col, narrative_col, matched_products)
    
    print(f"\nAll outputs saved to:")
    print(f"  - Processed dataset: data/filtered_complaints.csv")
    print(f"  - Visualizations: reports/")
    
    return final_df

if __name__ == "__main__":
    main() 