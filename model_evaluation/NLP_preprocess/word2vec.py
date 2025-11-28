"""
NLP Feature Extraction using Word2Vec
======================================

This script extracts Word2Vec embeddings for text fields:
- description
- title
- all_skills
- description_company
- all_industries
- benefits_list

Uses pre-trained Google News Word2Vec model (300 dimensions).
"""

import csv
import numpy as np
import gensim.downloader as api
from typing import List, Dict

# Import text preprocessing functions from text_preprocessing module
# preprocess_field handles all text cleaning, tokenization, and field-specific processing
from text_preprocessing import preprocess_field


# =============================================================================
# CONFIGURATION
# =============================================================================

# Text fields to process
TEXT_FIELDS = [
    'description',
    'title',
    'all_skills',
    'description_company',
    'all_industries',
    'benefits_list'
]

# Word2Vec model name
W2V_MODEL_NAME = 'word2vec-google-news-300'
VECTOR_SIZE = 300  # Google News Word2Vec uses 300 dimensions


# =============================================================================
# WORD2VEC FEATURE EXTRACTION
# =============================================================================

def load_word2vec_model():
    """
    Load pre-trained Word2Vec model.
    
    Returns:
        Word2Vec model (KeyedVectors)
    """
    print("Loading Word2Vec model...")
    try:
        model = api.load(W2V_MODEL_NAME)
        print("Model loaded successfully!")
        print(f"Vector dimension: {model.vector_size}")
        print(f"Vocabulary size: {len(model.key_to_index):,}")
        return model
    except (OSError, KeyError, ValueError) as e:
        print(f"Failed to load model: {e}")
        raise


def text_to_vector(text_tokens: List[str], model) -> np.ndarray:
    """
    Convert a list of tokens to a single vector using Word2Vec.
    
    Strategy: Average pooling - average all word vectors in the text.
    If no words are found in the vocabulary, return zero vector.
    
    Args:
        text_tokens: List of word tokens
        model: Word2Vec model
        
    Returns:
        numpy array of shape (vector_size,)
    """
    vectors = []
    
    for token in text_tokens:
        # Try to get vector for the token
        if token in model.key_to_index:
            vectors.append(model[token])
        # Try lowercase version if original not found
        elif token.lower() in model.key_to_index:
            vectors.append(model[token.lower()])
    
    if vectors:
        # Average all word vectors
        return np.mean(vectors, axis=0)
    else:
        # Return zero vector if no words found
        return np.zeros(model.vector_size)


def extract_features_for_record(record: Dict[str, str], model) -> Dict[str, np.ndarray]:
    """
    Extract Word2Vec features for all text fields in a single record.
    
    This function uses text_preprocessing.preprocess_field() to clean and tokenize
    text from each field before converting to Word2Vec vectors.
    
    Args:
        record: Dictionary containing CSV row data
        model: Word2Vec model
        
    Returns:
        Dictionary mapping field names to feature vectors
    """
    features = {}
    
    for field_name in TEXT_FIELDS:
        # Get raw text from record
        text = record.get(field_name, '')
        
        # Preprocess text using text_preprocessing module
        # This handles cleaning, tokenization, and field-specific processing
        tokens = preprocess_field(text, field_name)
        
        # Convert tokens to Word2Vec vector
        vector = text_to_vector(tokens, model)
        features[field_name] = vector
    
    return features


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_csv(input_file: str, output_file: str):
    """
    Process CSV file and extract Word2Vec features.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file with features
    """
    print("=" * 80)
    print("Word2Vec NLP Feature Extraction")
    print("=" * 80)
    print(f"\nInput file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"\nProcessing fields: {', '.join(TEXT_FIELDS)}")
    
    # Load Word2Vec model
    model = load_word2vec_model()
    
    # Read input CSV
    print("\nReading data file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        records = list(reader)
        fieldnames = reader.fieldnames
    
    print(f"Read {len(records):,} records")
    
    # Prepare output fieldnames
    # Keep all original fields, add new feature fields
    output_fieldnames = list(fieldnames) if fieldnames else []
    
    # Add feature columns for each text field
    for field_name in TEXT_FIELDS:
        for i in range(VECTOR_SIZE):
            output_fieldnames.append(f'{field_name}_w2v_{i}')
    
    # Process records
    print("\nExtracting features...")
    processed_records = []
    
    for idx, record in enumerate(records):
        # Extract features
        features = extract_features_for_record(record, model)
        
        # Create new record with original data + features
        new_record = dict(record)
        
        # Add feature vectors as separate columns
        for field_name in TEXT_FIELDS:
            vector = features[field_name]
            for i in range(VECTOR_SIZE):
                new_record[f'{field_name}_w2v_{i}'] = vector[i]
        
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
    
    print("\nCompleted!")
    print(f"Output file: {output_file}")
    print(f"Original features: {len(fieldnames) if fieldnames else 0}")
    print(f"New features: {len(TEXT_FIELDS) * VECTOR_SIZE} ({len(TEXT_FIELDS)} fields × {VECTOR_SIZE} dimensions)")
    print(f"Total features: {len(output_fieldnames)}")


def main():
    """Main entry point."""
    import sys
    import os
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to project root (NLP -> model_running -> job-salary-prediction-ml -> code)
    # Up 3 levels: NLP -> model_running -> job-salary-prediction-ml -> code
    project_root = os.path.abspath(os.path.join(script_dir, '../../..'))
    
    # Default file paths (relative to project root)
    input_file = os.path.join(project_root, 'salary_data_cleaned_final.csv')
    output_file = os.path.join(project_root, 'salary_data_with_w2v_features.csv')
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    try:
        process_csv(input_file, output_file)
    except (FileNotFoundError, OSError, KeyError, ValueError) as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

