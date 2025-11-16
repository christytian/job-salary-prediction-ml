"""
Hybrid NLP Feature Extraction: TF-IDF + Word2Vec
=================================================

This script extracts hybrid NLP features for text fields:
- description
- title
- all_skills
- description_company
- all_industries
- benefits_list

Uses both:
1. TF-IDF (Term Frequency-Inverse Document Frequency) - keyword-based features
2. Word2Vec - semantic embeddings from pre-trained Google News model

Combining both approaches provides:
- TF-IDF: Captures important keywords and their frequencies
- Word2Vec: Captures semantic meaning and relationships
"""

import csv
import numpy as np
import os
import sys
from typing import List, Dict, Tuple

# Import text preprocessing
from text_preprocessing import preprocess_field

# Import Word2Vec functions
from word2vec import (
    TEXT_FIELDS,
    VECTOR_SIZE,
    load_word2vec_model,
    text_to_vector
)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. TF-IDF features will be skipped.")
    print("Please install: pip install scikit-learn")



# =============================================================================
# CONFIGURATION
# =============================================================================

# TF-IDF parameters for each field
TFIDF_CONFIG = {
    'description': {
        'max_features': 100,
        'min_df': 5,
        'max_df': 0.95,
        'ngram_range': (1, 2)  # unigrams and bigrams
    },
    'title': {
        'max_features': 50,
        'min_df': 3,
        'max_df': 0.95,
        'ngram_range': (1, 2)
    },
    'all_skills': {
        'max_features': 50,
        'min_df': 3,
        'max_df': 0.95,
        'ngram_range': (1, 1)  # mostly single skills
    },
    'description_company': {
        'max_features': 50,
        'min_df': 5,
        'max_df': 0.95,
        'ngram_range': (1, 2)
    },
    'all_industries': {
        'max_features': 30,
        'min_df': 3,
        'max_df': 0.95,
        'ngram_range': (1, 1)  # mostly single industry names
    },
    'benefits_list': {
        'max_features': 30,
        'min_df': 3,
        'max_df': 0.95,
        'ngram_range': (1, 1)  # mostly single benefit names
    }
}


# =============================================================================
# TF-IDF FEATURE EXTRACTION
# =============================================================================

def prepare_text_for_tfidf(text: str, field_name: str) -> str:
    """
    Prepare text for TF-IDF vectorization.
    
    Converts text to a space-separated string of tokens.
    
    Args:
        text: Raw text from CSV
        field_name: Name of the field
        
    Returns:
        Space-separated string of tokens
    """
    tokens = preprocess_field(text, field_name)
    return ' '.join(tokens)


def extract_tfidf_features(data: List[Dict], field_name: str) -> Tuple[np.ndarray, List[str]]:
    """
    Extract TF-IDF features for a specific field.
    
    Args:
        data: List of record dictionaries
        field_name: Name of the field to process
        
    Returns:
        Tuple of (feature matrix, feature names)
    """
    if not SKLEARN_AVAILABLE:
        return np.array([]), []
    
    # Prepare texts
    texts = []
    for record in data:
        text = record.get(field_name, '')
        prepared_text = prepare_text_for_tfidf(text, field_name)
        texts.append(prepared_text)
    
    # Get TF-IDF configuration for this field
    config = TFIDF_CONFIG.get(field_name, {
        'max_features': 50,
        'min_df': 5,
        'max_df': 0.95,
        'ngram_range': (1, 1)
    })
    
    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=config['max_features'],
        min_df=config['min_df'],
        max_df=config['max_df'],
        ngram_range=config['ngram_range'],
        lowercase=True,
        token_pattern=r'(?u)\b\w+\b'  # Match words
    )
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Convert to dense array
    features = tfidf_matrix.toarray()
    
    # Get feature names
    feature_names = [f'{field_name}_tfidf_{name}' for name in vectorizer.get_feature_names_out()]
    
    return features, feature_names


# =============================================================================
# HYBRID FEATURE EXTRACTION
# =============================================================================

def extract_hybrid_features(data: List[Dict], w2v_model) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    """
    Extract hybrid features (TF-IDF + Word2Vec) for all text fields.
    
    Args:
        data: List of record dictionaries
        w2v_model: Word2Vec model
        
    Returns:
        Tuple of:
        - Dictionary mapping field names to feature arrays
        - Dictionary mapping field names to feature name lists
    """
    all_features = {}
    all_feature_names = {}
    
    print("\nExtracting hybrid features...")
    
    for field_name in TEXT_FIELDS:
        print(f"  Processing {field_name}...")
        
        # Extract TF-IDF features
        tfidf_features, tfidf_names = extract_tfidf_features(data, field_name)
        
        # Extract Word2Vec features
        w2v_features = []
        for record in data:
            text = record.get(field_name, '')
            tokens = preprocess_field(text, field_name)
            vector = text_to_vector(tokens, w2v_model)
            w2v_features.append(vector)
        
        w2v_features = np.array(w2v_features)
        w2v_names = [f'{field_name}_w2v_{i}' for i in range(VECTOR_SIZE)]
        
        # Combine TF-IDF and Word2Vec features
        if tfidf_features.size > 0:
            combined_features = np.hstack([tfidf_features, w2v_features])
            combined_names = tfidf_names + w2v_names
        else:
            combined_features = w2v_features
            combined_names = w2v_names
        
        all_features[field_name] = combined_features
        all_feature_names[field_name] = combined_names
        
        print(f"    TF-IDF: {len(tfidf_names)} features")
        print(f"    Word2Vec: {len(w2v_names)} features")
        print(f"    Total: {len(combined_names)} features")
    
    return all_features, all_feature_names


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_csv(input_file: str, output_file: str):
    """
    Process CSV file and extract hybrid (TF-IDF + Word2Vec) features.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file with features
    """
    print("=" * 80)
    print("Hybrid NLP Feature Extraction (TF-IDF + Word2Vec)")
    print("=" * 80)
    print(f"\nInput file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"\nProcessing fields: {', '.join(TEXT_FIELDS)}")
    
    # Load Word2Vec model
    w2v_model = load_word2vec_model()
    
    # Read input CSV
    print("\nReading data file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        records = list(reader)
        fieldnames = reader.fieldnames
    
    print(f"Read {len(records):,} records")
    
    # Extract hybrid features
    all_features, all_feature_names = extract_hybrid_features(records, w2v_model)
    
    # Prepare output fieldnames
    output_fieldnames = list(fieldnames) if fieldnames else []
    
    # Add feature columns
    for field_name in TEXT_FIELDS:
        output_fieldnames.extend(all_feature_names[field_name])
    
    # Process records
    print("\nCombining features with original data...")
    processed_records = []
    
    for idx, record in enumerate(records):
        new_record = dict(record)
        
        # Add hybrid features for each field
        for field_name in TEXT_FIELDS:
            features = all_features[field_name][idx]
            feature_names = all_feature_names[field_name]
            
            for i, name in enumerate(feature_names):
                new_record[name] = features[i]
        
        processed_records.append(new_record)
        
        # Progress update
        if (idx + 1) % 1000 == 0:
            print(f"  Processed: {idx + 1:,}/{len(records):,} records")
    
    # Write output CSV
    print("\nWriting output file...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(processed_records)
    
    # Calculate feature counts
    total_tfidf = sum(len([n for n in all_feature_names[f] if '_tfidf_' in n]) for f in TEXT_FIELDS)
    total_w2v = sum(len([n for n in all_feature_names[f] if '_w2v_' in n]) for f in TEXT_FIELDS)
    
    print("\nCompleted!")
    print(f"Output file: {output_file}")
    print(f"Original features: {len(fieldnames) if fieldnames else 0}")
    print(f"New TF-IDF features: {total_tfidf}")
    print(f"New Word2Vec features: {total_w2v}")
    print(f"Total new features: {total_tfidf + total_w2v}")
    print(f"Total features: {len(output_fieldnames)}")




# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../../..'))
    
    # Default file paths
    input_file = os.path.join(project_root, 'salary_data_cleaned_final.csv')
    output_file = os.path.join(project_root, 'salary_data_with_nlp_hybrid_features.csv')
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    try:
        # Process CSV and extract features
        process_csv(input_file, output_file)
        print("\n💡 Tip: Run evaluate_hybrid.py to evaluate the features:")
        print(f"   python3 evaluate_hybrid.py {output_file}")
        
    except (FileNotFoundError, OSError, KeyError, ValueError) as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

