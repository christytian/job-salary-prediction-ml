"""
Extract NLP Features for Comparison
====================================
Extract TF-IDF only, SBERT only, or Hybrid (TF-IDF + SBERT) features.

This script can generate three different feature sets for performance comparison:
1. TF-IDF only
2. SBERT only  
3. Hybrid (TF-IDF + SBERT)

Usage:
    python extract_nlp_features_comparison.py --method tfidf
    python extract_nlp_features_comparison.py --method sbert
    python extract_nlp_features_comparison.py --method hybrid
"""

import csv
import numpy as np
import pickle
import argparse
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "input_file": "salary_data_cleaned_final.csv",
    "output_dir": "nlp_features",  # Output directory for all NLP feature files
    "tfidf_max_features": 100,  # Top 100 TF-IDF features per field
    "sbert_model": "all-MiniLM-L6-v2",  # Lightweight SBERT model (384 dims)
    "random_state": 42,
}

# Fields to extract NLP features from
NLP_FIELDS = {
    "description": {
        "description": "Job description (most important)"
    },
    "title": {
        "description": "Job title"
    },
    "all_skills": {
        "description": "Skills list"
    },
    "description_company": {
        "description": "Company description"
    },
    "all_industries": {
        "description": "All industries"
    },
    "benefits_list": {
        "description": "Benefits list"
    }
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract NLP features for comparison (TF-IDF, SBERT, or Hybrid)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["tfidf", "sbert", "hybrid"],
        default="hybrid",
        help="NLP method to use: tfidf, sbert, or hybrid (default: hybrid)"
    )
    return parser.parse_args()


def load_data(input_file):
    """Load data from CSV file."""
    print(f"\n📖 Loading data from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    print(f"   ✓ Loaded {len(rows):,} records")
    print(f"   ✓ {len(fieldnames)} columns")
    
    return rows, fieldnames


def preprocess_text(text, field_name=""):
    """Preprocess text for NLP."""
    if not text or not isinstance(text, str):
        return ""
    
    # Basic cleaning
    text = text.strip()
    
    # Replace pipe separators with spaces for list fields
    if 'skills' in field_name.lower() or 'industries' in field_name.lower() or 'benefits' in field_name.lower():
        text = text.replace('|', ' ')
    
    return text


def extract_tfidf_features(texts, field_name, max_features=100):
    """Extract TF-IDF features from text list."""
    print(f"\n   🔤 Extracting TF-IDF features from {field_name}...")
    
    # Preprocess texts
    processed_texts = [preprocess_text(text, field_name) for text in texts]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,  # Word must appear in at least 2 documents
        max_df=0.95,  # Ignore words that appear in >95% of documents
    )
    
    # Fit and transform
    try:
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        feature_names = [f"{field_name}_tfidf_{word}" for word in vectorizer.get_feature_names_out()]
        
        print(f"      ✓ Extracted {len(feature_names)} TF-IDF features")
        
        return tfidf_matrix.toarray(), feature_names, vectorizer
    except Exception as e:
        print(f"      ⚠️  Error extracting TF-IDF: {e}")
        print(f"      → Using empty features")
        return np.zeros((len(texts), max_features)), [f"{field_name}_tfidf_{i}" for i in range(max_features)], None


def extract_sbert_embeddings(texts, field_name, model):
    """Extract SBERT embeddings from text list."""
    print(f"\n   🧠 Extracting SBERT embeddings from {field_name}...")
    
    # Preprocess texts
    processed_texts = [preprocess_text(text, field_name) for text in texts]
    
    # Generate embeddings
    try:
        embeddings = model.encode(processed_texts, show_progress_bar=True, batch_size=32)
        feature_names = [f"{field_name}_emb_{i}" for i in range(embeddings.shape[1])]
        
        print(f"      ✓ Extracted {len(feature_names)} SBERT embedding features")
        
        return embeddings, feature_names
    except Exception as e:
        print(f"      ⚠️  Error extracting SBERT: {e}")
        print(f"      → Using empty features")
        dim = 384  # Default SBERT dimension
        return np.zeros((len(texts), dim)), [f"{field_name}_emb_{i}" for i in range(dim)]


def main():
    args = parse_arguments()
    method = args.method
    
    print("=" * 80)
    print("NLP FEATURE EXTRACTION FOR COMPARISON")
    print("=" * 80)
    print(f"Method: {method.upper()}")
    print(f"Input:  {CONFIG['input_file']}")
    
    # Create output directory if it doesn't exist
    output_dir = CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}/")
    
    # Determine output file based on method
    output_file = os.path.join(output_dir, f"salary_data_nlp_{method}.csv")
    print(f"Output file: {output_file}")
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    rows, original_fieldnames = load_data(CONFIG['input_file'])
    
    # Extract text fields
    text_data = {}
    for field in NLP_FIELDS.keys():
        values = []
        for row in rows:
            value = row.get(field, '')
            # Handle missing values - use empty string
            if not value or value.strip() == '':
                values.append('')
            else:
                values.append(value)
        text_data[field] = values
        
        # Report missing values
        missing_count = sum(1 for v in values if not v)
        if missing_count > 0:
            print(f"   ⚠️  {field}: {missing_count:,} missing values ({missing_count/len(rows)*100:.1f}%)")
    
    # =========================================================================
    # STEP 2: Load SBERT Model (if needed)
    # =========================================================================
    sbert_model = None
    if method in ["sbert", "hybrid"]:
        print(f"\n🤖 Loading SBERT model: {CONFIG['sbert_model']}...")
        try:
            sbert_model = SentenceTransformer(CONFIG['sbert_model'])
            print(f"   ✓ Model loaded successfully")
        except Exception as e:
            print(f"   ❌ Error loading SBERT model: {e}")
            if method == "sbert":
                print(f"   → Cannot continue without SBERT model")
                return
            else:
                print(f"   → Continuing with TF-IDF only")
    
    # =========================================================================
    # STEP 3: Extract NLP Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXTRACTING NLP FEATURES")
    print("=" * 80)
    
    all_nlp_features = {}
    all_nlp_feature_names = []
    vectorizers = {}
    
    for field_name, config in NLP_FIELDS.items():
        print(f"\n📝 Processing field: {field_name} ({config['description']})")
        
        texts = text_data[field_name]
        
        # Extract TF-IDF (if method is tfidf or hybrid)
        if method in ["tfidf", "hybrid"]:
            tfidf_matrix, tfidf_names, vectorizer = extract_tfidf_features(
                texts, field_name, CONFIG['tfidf_max_features']
            )
            all_nlp_features[f"{field_name}_tfidf"] = tfidf_matrix
            all_nlp_feature_names.extend(tfidf_names)
            if vectorizer:
                vectorizers[f"{field_name}_tfidf"] = vectorizer
        
        # Extract SBERT embeddings (if method is sbert or hybrid)
        if method in ["sbert", "hybrid"] and sbert_model:
            embeddings, emb_names = extract_sbert_embeddings(texts, field_name, sbert_model)
            all_nlp_features[f"{field_name}_sbert"] = embeddings
            all_nlp_feature_names.extend(emb_names)
    
    # =========================================================================
    # STEP 4: Combine Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMBINING FEATURES")
    print("=" * 80)
    
    # Combine all NLP features into single array
    nlp_feature_arrays = []
    for key in sorted(all_nlp_features.keys()):
        nlp_feature_arrays.append(all_nlp_features[key])
    
    if nlp_feature_arrays:
        combined_nlp_features = np.hstack(nlp_feature_arrays)
        print(f"\n✓ Combined NLP features: {combined_nlp_features.shape}")
        print(f"   Total NLP features: {len(all_nlp_feature_names)}")
    else:
        print("\n⚠️  No NLP features extracted!")
        combined_nlp_features = None
    
    # =========================================================================
    # STEP 5: Save Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Prepare output
    output_fieldnames = list(original_fieldnames) + all_nlp_feature_names
    
    print(f"\n💾 Writing to {output_file}...")
    print(f"   Original columns: {len(original_fieldnames)}")
    print(f"   NLP feature columns: {len(all_nlp_feature_names)}")
    print(f"   Total columns: {len(output_fieldnames)}")
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        
        for i, row in enumerate(rows):
            # Add original row data
            output_row = dict(row)
            
            # Add NLP features
            if combined_nlp_features is not None:
                for j, feat_name in enumerate(all_nlp_feature_names):
                    output_row[feat_name] = combined_nlp_features[i, j]
            
            writer.writerow(output_row)
            
            if (i + 1) % 5000 == 0:
                print(f"   Processed {i+1:,}/{len(rows):,} rows...")
    
    print(f"\n✅ Successfully saved {len(rows):,} records to {output_file}")
    
    # Save vectorizers for future use (if TF-IDF was used)
    if vectorizers and method in ["tfidf", "hybrid"]:
        vectorizer_file = os.path.join(output_dir, f"nlp_vectorizers_{method}.pkl")
        print(f"\n💾 Saving TF-IDF vectorizers to {vectorizer_file}...")
        with open(vectorizer_file, 'wb') as f:
            pickle.dump(vectorizers, f)
        print(f"   ✓ Saved {len(vectorizers)} vectorizers")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\n✅ NLP Feature Extraction Complete!")
    print(f"   Method:           {method.upper()}")
    print(f"   Input file:      {CONFIG['input_file']}")
    print(f"   Output file:     {output_file}")
    print(f"   Records:         {len(rows):,}")
    print(f"   Original features: {len(original_fieldnames)}")
    print(f"   NLP features:      {len(all_nlp_feature_names)}")
    print(f"   Total features:    {len(output_fieldnames)}")
    
    # Feature breakdown by method
    print(f"\n📊 Feature Breakdown by Method ({method.upper()}):")
    tfidf_count = 0
    sbert_count = 0
    
    for field_name in NLP_FIELDS.keys():
        field_tfidf = 0
        field_sbert = 0
        
        if method in ["tfidf", "hybrid"]:
            field_tfidf = CONFIG['tfidf_max_features']
            tfidf_count += field_tfidf
        
        if method in ["sbert", "hybrid"] and sbert_model:
            field_sbert = 384  # SBERT dimension
            sbert_count += field_sbert
        
        total = field_tfidf + field_sbert
        if total > 0:
            print(f"   {field_name:<25} TF-IDF: {field_tfidf:>3}  SBERT: {field_sbert:>3}  Total: {total:>3}")
    
    print(f"\n   Total TF-IDF features: {tfidf_count}")
    print(f"   Total SBERT features: {sbert_count}")
    print(f"   Grand Total: {tfidf_count + sbert_count}")


if __name__ == "__main__":
    main()

