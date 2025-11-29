"""
Extract NLP Features with SBERT PCA for Comparison
===================================================
Extract TF-IDF only, SBERT PCA only, or Hybrid (TF-IDF + SBERT PCA) features.

This script can generate three different feature sets for performance comparison:
1. TF-IDF only (310 features - matches W2V config)
2. SBERT with PCA only (2,304 → 660 dims, 95% variance)
3. Hybrid (TF-IDF + SBERT PCA = 310 + 660 = 970 features)

TF-IDF configuration matches W2V hybrid for fair comparison:
- description: 100, title: 50, all_skills: 50, description_company: 50,
  all_industries: 30, benefits_list: 30 → Total: 310 features

Usage:
    python extract_nlp_features_pca_comparison.py --method tfidf
    python extract_nlp_features_pca_comparison.py --method sbert_pca
    python extract_nlp_features_pca_comparison.py --method hybrid
"""

import csv
import numpy as np
import pickle
import argparse
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "input_file": "salary_data_ready_for_new_nlp.csv",
    "output_dir": "nlp_features_pca",  # Output directory for all NLP feature files
    "sbert_model": "all-MiniLM-L6-v2",  # Lightweight SBERT model (384 dims)
    "sbert_pca_variance": 0.95,  # Preserve 95% variance in SBERT embeddings (optimal performance)
    "random_state": 42,
}

# Fields to extract NLP features from
# TF-IDF config matches W2V hybrid for fair comparison (310 total features)
NLP_FIELDS = {
    "title": {
        "description": "Job title",
        "tfidf_max_features": 50
    },
    "description": {
        "description": "Job description (most important)",
        "tfidf_max_features": 100
    },
    "all_skills": {
        "description": "Skills list",
        "tfidf_max_features": 50
    },
    "all_industries": {
        "description": "All industries",
        "tfidf_max_features": 30
    },
    "benefits_list": {
        "description": "Benefits list",
        "tfidf_max_features": 30
    },
    "description_company": {
        "description": "Company description",
        "tfidf_max_features": 50
    }
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract NLP features for comparison (TF-IDF, SBERT PCA, or Hybrid)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["tfidf", "sbert_pca", "hybrid"],
        default="hybrid",
        help="NLP method to use: tfidf, sbert_pca, or hybrid (default: hybrid)"
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


def extract_tfidf_features(texts, field_name, max_features=199):
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
        feature_names = [f"tfidf_{field_name}_{i}" for i in range(tfidf_matrix.shape[1])]

        print(f"      ✓ Extracted {len(feature_names)} TF-IDF features")

        return tfidf_matrix.toarray(), feature_names, vectorizer
    except Exception as e:
        print(f"      ⚠️  Error extracting TF-IDF: {e}")
        print(f"      → Using empty features")
        return np.zeros((len(texts), max_features)), [f"tfidf_{field_name}_{i}" for i in range(max_features)], None


def extract_sbert_embeddings(texts, field_name, model):
    """Extract SBERT embeddings from text list."""
    print(f"\n   🧠 Extracting SBERT embeddings from {field_name}...")

    # Preprocess texts
    processed_texts = [preprocess_text(text, field_name) for text in texts]

    # Generate embeddings
    try:
        embeddings = model.encode(processed_texts, show_progress_bar=True, batch_size=32)
        feature_names = [f"emb_{field_name}_{i}" for i in range(embeddings.shape[1])]

        print(f"      ✓ Extracted {embeddings.shape[1]} SBERT embedding dimensions")

        return embeddings, feature_names
    except Exception as e:
        print(f"      ⚠️  Error extracting SBERT: {e}")
        print(f"      → Using empty features")
        dim = 384  # Default SBERT dimension
        return np.zeros((len(texts), dim)), [f"emb_{field_name}_{i}" for i in range(dim)]


def apply_pca_to_sbert(sbert_features, variance_ratio=0.95):
    """Apply PCA to SBERT embeddings to reduce dimensionality."""
    print(f"\n🔧 Applying PCA to SBERT embeddings...")
    print(f"   Original SBERT dimensions: {sbert_features.shape[1]:,}")
    print(f"   Target variance preserved: {variance_ratio*100:.0f}%")

    # Apply PCA
    pca = PCA(n_components=variance_ratio, random_state=42)
    sbert_pca = pca.fit_transform(sbert_features)

    # Get new feature names
    pca_feature_names = [f'sbert_pca_{i}' for i in range(sbert_pca.shape[1])]

    # Calculate variance explained
    total_variance = np.sum(pca.explained_variance_ratio_)

    print(f"   ✓ Reduced to {sbert_pca.shape[1]} dimensions")
    print(f"   ✓ Variance preserved: {total_variance*100:.2f}%")
    print(f"   ✓ Dimensionality reduction: {(1 - sbert_pca.shape[1]/sbert_features.shape[1])*100:.1f}%")

    return sbert_pca, pca_feature_names, pca


def get_baseline_features(rows, original_fieldnames):
    """Extract baseline features (excluding NLP and text columns)."""
    print(f"\n📊 Extracting baseline features...")

    # Target variable
    target_col = 'salary_normalized'

    # Columns to exclude
    exclude_cols = set()
    exclude_cols.add(target_col)
    exclude_cols.add('normalized_salary')  # Data leakage
    exclude_cols.add('salary')
    exclude_cols.add('min_salary')
    exclude_cols.add('max_salary')
    exclude_cols.add('salary_min')
    exclude_cols.add('salary_max')

    # Exclude text columns
    text_cols = ['title', 'description', 'company_name', 'location',
                 'description_company', 'title_cleaned', 'description_cleaned',
                 'all_skills_cleaned', 'benefits_list', 'all_industries',
                 'all_skills', 'all_company_industries', 'all_specialities',
                 'primary_industry', 'primary_skill', 'primary_company_industry',
                 'primary_speciality']
    exclude_cols.update(text_cols)

    # Get numeric feature columns
    feature_cols = []
    for col in original_fieldnames:
        if col in exclude_cols:
            continue

        # Sample to check if numeric
        sample_val = None
        for row in rows[:100]:
            if row.get(col) and row[col] != '':
                sample_val = row[col]
                break

        if sample_val:
            try:
                float(sample_val)
                feature_cols.append(col)
            except ValueError:
                pass

    # Extract features
    baseline_features = []
    for row in rows:
        features = [float(row[col]) if row[col] else 0.0 for col in feature_cols]
        baseline_features.append(features)

    baseline_array = np.array(baseline_features)

    print(f"   ✓ Extracted {len(feature_cols)} baseline features")

    return baseline_array, feature_cols


def main():
    args = parse_arguments()
    method = args.method

    print("=" * 80)
    print("NLP FEATURE EXTRACTION WITH SBERT PCA FOR COMPARISON")
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
    print("\n" + "=" * 80)
    print("STEP 1: LOAD DATA")
    print("=" * 80)

    rows, original_fieldnames = load_data(CONFIG['input_file'])

    # =========================================================================
    # STEP 2: Extract Baseline Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: EXTRACT BASELINE FEATURES")
    print("=" * 80)

    baseline_features, baseline_feature_names = get_baseline_features(rows, original_fieldnames)

    # =========================================================================
    # STEP 3: Extract Text Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: EXTRACT TEXT DATA")
    print("=" * 80)

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
    # STEP 4: Load SBERT Model (if needed)
    # =========================================================================
    sbert_model = None
    if method in ["sbert_pca", "hybrid"]:
        print("\n" + "=" * 80)
        print("STEP 4: LOAD SBERT MODEL")
        print("=" * 80)

        print(f"\n🤖 Loading SBERT model: {CONFIG['sbert_model']}...")
        try:
            sbert_model = SentenceTransformer(CONFIG['sbert_model'])
            print(f"   ✓ Model loaded successfully (embedding dimension: {sbert_model.get_sentence_embedding_dimension()})")
        except Exception as e:
            print(f"   ❌ Error loading SBERT model: {e}")
            if method == "sbert_pca":
                print(f"   → Cannot continue without SBERT model")
                return
            else:
                print(f"   → Continuing with TF-IDF only")

    # =========================================================================
    # STEP 5: Extract NLP Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: EXTRACT NLP FEATURES")
    print("=" * 80)

    all_nlp_features = []
    all_nlp_feature_names = []
    vectorizers = {}
    pca_model = None

    # Extract TF-IDF features (if method is tfidf or hybrid)
    if method in ["tfidf", "hybrid"]:
        print(f"\n{'='*80}")
        print("EXTRACTING TF-IDF FEATURES")
        print(f"{'='*80}")

        tfidf_features_all = []

        for field_name, config in NLP_FIELDS.items():
            print(f"\n📝 Processing field: {field_name} ({config['description']})")

            texts = text_data[field_name]

            tfidf_matrix, tfidf_names, vectorizer = extract_tfidf_features(
                texts, field_name, config['tfidf_max_features']
            )

            tfidf_features_all.append(tfidf_matrix)
            all_nlp_feature_names.extend(tfidf_names)

            if vectorizer:
                vectorizers[f"{field_name}_tfidf"] = vectorizer

        # Combine all TF-IDF features
        tfidf_combined = np.hstack(tfidf_features_all)
        all_nlp_features.append(tfidf_combined)

        print(f"\n✓ Total TF-IDF features: {len([n for n in all_nlp_feature_names if 'tfidf' in n]):,}")

    # Extract SBERT embeddings and apply PCA (if method is sbert_pca or hybrid)
    if method in ["sbert_pca", "hybrid"] and sbert_model:
        print(f"\n{'='*80}")
        print("EXTRACTING SBERT EMBEDDINGS")
        print(f"{'='*80}")

        sbert_embeddings_all = []

        for field_name, config in NLP_FIELDS.items():
            print(f"\n📝 Processing field: {field_name} ({config['description']})")

            texts = text_data[field_name]

            embeddings, emb_names = extract_sbert_embeddings(texts, field_name, sbert_model)
            sbert_embeddings_all.append(embeddings)

        # Combine all SBERT embeddings
        sbert_combined = np.hstack(sbert_embeddings_all)

        print(f"\n✓ Total SBERT embeddings before PCA: {sbert_combined.shape[1]:,}")
        print(f"   ({len(NLP_FIELDS)} fields × 384 dims = {len(NLP_FIELDS) * 384} expected)")

        # Apply PCA to combined SBERT embeddings
        print(f"\n{'='*80}")
        print("APPLYING PCA TO SBERT EMBEDDINGS")
        print(f"{'='*80}")

        sbert_pca, sbert_pca_names, pca_model = apply_pca_to_sbert(
            sbert_combined,
            variance_ratio=CONFIG['sbert_pca_variance']
        )

        all_nlp_features.append(sbert_pca)
        all_nlp_feature_names.extend(sbert_pca_names)

        print(f"\n✓ Total SBERT PCA features: {len(sbert_pca_names):,}")

    # =========================================================================
    # STEP 6: Combine All Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: COMBINE ALL FEATURES")
    print("=" * 80)

    # Combine baseline + NLP features
    if all_nlp_features:
        combined_nlp = np.hstack(all_nlp_features)
        all_features = np.hstack([baseline_features, combined_nlp])
    else:
        all_features = baseline_features
        combined_nlp = None

    all_feature_names = baseline_feature_names + all_nlp_feature_names

    print(f"\n📊 Feature Summary:")
    print(f"   • Baseline features: {len(baseline_feature_names):,}")
    print(f"   • NLP features: {len(all_nlp_feature_names):,}")
    print(f"   {'─' * 60}")
    print(f"   • Total features: {len(all_feature_names):,}")

    # =========================================================================
    # STEP 7: Save Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: SAVE OUTPUT")
    print("=" * 80)

    print(f"\n💾 Writing to {output_file}...")

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_feature_names)
        writer.writeheader()

        for i in range(len(rows)):
            output_row = {}

            # Add all features
            for j, col_name in enumerate(all_feature_names):
                output_row[col_name] = all_features[i, j]

            writer.writerow(output_row)

            # Progress
            if (i + 1) % 5000 == 0:
                print(f"   Written: {i+1:,}/{len(rows):,} records")

    print(f"\n✅ Successfully saved {len(rows):,} records to {output_file}")

    # =========================================================================
    # STEP 8: Save Models
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 8: SAVE MODELS")
    print("=" * 80)

    models = {
        'method': method,
        'feature_names': all_feature_names,
        'baseline_feature_names': baseline_feature_names,
        'nlp_feature_names': all_nlp_feature_names,
    }

    if vectorizers:
        models['tfidf_vectorizers'] = vectorizers

    if pca_model:
        models['sbert_pca'] = pca_model
        models['sbert_model_name'] = CONFIG['sbert_model']

    model_file = os.path.join(output_dir, f"nlp_models_{method}.pkl")

    with open(model_file, 'wb') as f:
        pickle.dump(models, f)

    print(f"\n✓ Saved models to: {model_file}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("✅ FEATURE EXTRACTION COMPLETE!")
    print("=" * 80)

    # Feature breakdown
    tfidf_count = len([n for n in all_nlp_feature_names if 'tfidf' in n])
    sbert_pca_count = len([n for n in all_nlp_feature_names if 'sbert_pca' in n])

    print(f"""
📈 Final Results:

Method: {method.upper()}
Input:  {CONFIG['input_file']} ({len(rows):,} records)
Output: {output_file} ({len(rows):,} records, {len(all_feature_names):,} features)

Feature Breakdown:
  ✓ Baseline features:     {len(baseline_feature_names):>4,}
  ✓ TF-IDF features:       {tfidf_count:>4,}
  ✓ SBERT PCA features:    {sbert_pca_count:>4,}
  {'─' * 60}
  ✓ Total features:        {len(all_feature_names):>4,}

Next Steps:
  1. Train models using: {output_file}
  2. Compare performance across methods:
     - {output_dir}/salary_data_nlp_tfidf.csv
     - {output_dir}/salary_data_nlp_sbert_pca.csv
     - {output_dir}/salary_data_nlp_hybrid.csv
  3. Evaluate which NLP approach provides best performance
    """)

    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
