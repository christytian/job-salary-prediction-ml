"""
NLP Feature Extraction with Selective SBERT PCA
================================================

This script extracts NLP features and applies PCA ONLY to SBERT embeddings:
- TF-IDF features: Keep as-is (199 features)
- SBERT embeddings: Apply PCA to reduce from 2,304 → ~150-200 dims (95% variance)
- Other features: Keep unchanged

Input:  salary_data_ready_for_new_nlp.csv
Output: salary_data_with_nlp_sbert_pca.csv

Expected Results:
- Original features: ~27 baseline features
- TF-IDF features: 199 (unchanged)
- SBERT PCA: ~150-200 dims (reduced from 2,304)
- Total: ~376-426 features (down from ~2,530)
"""

import csv
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Text fields to extract NLP features from
TEXT_FIELDS = [
    'title',
    'description',
    'all_skills',
    'all_industries',
    'benefits_list',
    'description_company'
]

# TF-IDF Configuration
TFIDF_CONFIG = {
    'max_features': 199,
    'min_df': 2,
    'max_df': 0.95,
    'ngram_range': (1, 2),
    'strip_accents': 'unicode',
    'lowercase': True
}

# SBERT Configuration
SBERT_MODEL = 'all-MiniLM-L6-v2'  # 384 dimensions per field
SBERT_PCA_VARIANCE = 0.95  # Preserve 95% of variance

# Fields to exclude from final output
EXCLUDE_FIELDS = [
    'company_name',
    'location',
    'primary_industry',
    'primary_skill',
    'primary_company_industry',
    'primary_speciality'
]


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_data(filename):
    """Load CSV data"""
    print(f"\n📖 Loading {filename}...")
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
        fieldnames = list(reader.fieldnames)

    print(f"   ✓ Loaded {len(data):,} records with {len(fieldnames)} columns")
    return data, fieldnames


def extract_tfidf_features(data, field_name):
    """Extract TF-IDF features for a single text field"""
    print(f"\n   [TF-IDF] Processing {field_name}...")

    # Extract text
    texts = [row.get(field_name, '').strip() for row in data]
    texts = [t if t else 'missing' for t in texts]

    # Fit TF-IDF
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Get feature names
    feature_names = [f'tfidf_{field_name}_{i}' for i in range(tfidf_matrix.shape[1])]

    print(f"      → Created {len(feature_names)} TF-IDF features")

    return tfidf_matrix.toarray(), feature_names, vectorizer


def extract_sbert_embeddings(data, field_name, model):
    """Extract SBERT embeddings for a single text field"""
    print(f"\n   [SBERT] Processing {field_name}...")

    # Extract text
    texts = [row.get(field_name, '').strip() for row in data]
    texts = [t if t else 'missing' for t in texts]

    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=False)

    # Get feature names
    feature_names = [f'emb_{field_name}_{i}' for i in range(embeddings.shape[1])]

    print(f"      → Created {embeddings.shape[1]} embedding dimensions")

    return embeddings, feature_names


def apply_pca_to_sbert(sbert_features, sbert_feature_names, variance_ratio=0.95):
    """Apply PCA to SBERT embeddings to reduce dimensionality"""
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


def main():
    print("=" * 80)
    print("NLP FEATURE EXTRACTION WITH SELECTIVE SBERT PCA")
    print("=" * 80)
    print("\n📋 Configuration:")
    print(f"   • Text fields: {len(TEXT_FIELDS)}")
    print(f"   • TF-IDF max features: {TFIDF_CONFIG['max_features']}")
    print(f"   • SBERT model: {SBERT_MODEL} (384 dims)")
    print(f"   • SBERT PCA variance: {SBERT_PCA_VARIANCE*100:.0f}%")

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOAD DATA")
    print("=" * 80)

    data, fieldnames = load_data('salary_data_ready_for_new_nlp.csv')

    # =========================================================================
    # STEP 2: Extract TF-IDF Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: EXTRACT TF-IDF FEATURES (NO PCA)")
    print("=" * 80)

    tfidf_features_all = []
    tfidf_feature_names_all = []
    tfidf_vectorizers = {}

    for field in TEXT_FIELDS:
        if field in fieldnames:
            tfidf_matrix, feature_names, vectorizer = extract_tfidf_features(data, field)
            tfidf_features_all.append(tfidf_matrix)
            tfidf_feature_names_all.extend(feature_names)
            tfidf_vectorizers[field] = vectorizer

    # Concatenate all TF-IDF features
    tfidf_features_combined = np.hstack(tfidf_features_all)

    print(f"\n✓ Total TF-IDF features: {len(tfidf_feature_names_all):,} (unchanged)")

    # =========================================================================
    # STEP 3: Extract SBERT Embeddings
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: EXTRACT SBERT EMBEDDINGS")
    print("=" * 80)

    print(f"\n🤖 Loading SBERT model: {SBERT_MODEL}...")
    sbert_model = SentenceTransformer(SBERT_MODEL)
    print(f"   ✓ Model loaded (embedding dimension: {sbert_model.get_sentence_embedding_dimension()})")

    sbert_embeddings_all = []
    sbert_feature_names_all = []

    for field in TEXT_FIELDS:
        if field in fieldnames:
            embeddings, feature_names = extract_sbert_embeddings(data, field, sbert_model)
            sbert_embeddings_all.append(embeddings)
            sbert_feature_names_all.extend(feature_names)

    # Concatenate all SBERT embeddings
    sbert_embeddings_combined = np.hstack(sbert_embeddings_all)

    print(f"\n✓ Total SBERT embeddings before PCA: {len(sbert_feature_names_all):,}")
    print(f"   ({len(TEXT_FIELDS)} fields × 384 dims = {len(TEXT_FIELDS) * 384} expected)")

    # =========================================================================
    # STEP 4: Apply PCA to SBERT Embeddings Only
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: APPLY PCA TO SBERT EMBEDDINGS ONLY")
    print("=" * 80)

    sbert_pca, sbert_pca_names, pca_model = apply_pca_to_sbert(
        sbert_embeddings_combined,
        sbert_feature_names_all,
        variance_ratio=SBERT_PCA_VARIANCE
    )

    # =========================================================================
    # STEP 5: Combine All Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: COMBINE ALL FEATURES")
    print("=" * 80)

    # Get original features (excluding text fields)
    original_features = []
    original_feature_names = []

    for field in fieldnames:
        if field in EXCLUDE_FIELDS or field in TEXT_FIELDS:
            continue

        # Extract column values
        values = [row.get(field, '') for row in data]

        # Check if numeric
        try:
            numeric_values = [float(v) if v and v != '' else 0.0 for v in values]
            original_features.append(numeric_values)
            original_feature_names.append(field)
        except ValueError:
            # Skip non-numeric fields
            pass

    original_features_array = np.column_stack(original_features) if original_features else np.empty((len(data), 0))

    print(f"\n📊 Feature Summary:")
    print(f"   • Original features: {len(original_feature_names):,}")
    print(f"   • TF-IDF features: {len(tfidf_feature_names_all):,} (no PCA)")
    print(f"   • SBERT PCA features: {len(sbert_pca_names):,} (reduced from {len(sbert_feature_names_all):,})")
    print(f"   {'─' * 60}")
    print(f"   • Total features: {len(original_feature_names) + len(tfidf_feature_names_all) + len(sbert_pca_names):,}")
    print(f"\n💾 Feature reduction:")
    print(f"   • Before PCA: {len(original_feature_names) + len(tfidf_feature_names_all) + len(sbert_feature_names_all):,} features")
    print(f"   • After PCA:  {len(original_feature_names) + len(tfidf_feature_names_all) + len(sbert_pca_names):,} features")
    print(f"   • Reduction:  {len(sbert_feature_names_all) - len(sbert_pca_names):,} features ({(1 - (len(sbert_pca_names)/len(sbert_feature_names_all)))*100:.1f}%)")

    # Combine all features
    all_features = np.hstack([
        original_features_array,
        tfidf_features_combined,
        sbert_pca
    ])

    all_feature_names = original_feature_names + tfidf_feature_names_all + sbert_pca_names

    # =========================================================================
    # STEP 6: Save Output
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: SAVE OUTPUT")
    print("=" * 80)

    output_file = 'salary_data_with_nlp_sbert_pca.csv'

    print(f"\n💾 Writing to {output_file}...")

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_feature_names)
        writer.writeheader()

        for i, row in enumerate(data):
            new_row = {}

            # Add all features
            for j, col_name in enumerate(all_feature_names):
                new_row[col_name] = all_features[i, j]

            writer.writerow(new_row)

            # Progress
            if (i + 1) % 1000 == 0:
                print(f"   Written: {i+1:,}/{len(data):,} records")

    print(f"\n✓ Saved: {output_file}")

    # =========================================================================
    # STEP 7: Save Models
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: SAVE MODELS")
    print("=" * 80)

    models = {
        'tfidf_vectorizers': tfidf_vectorizers,
        'sbert_model_name': SBERT_MODEL,
        'sbert_pca': pca_model,
        'feature_names': {
            'original': original_feature_names,
            'tfidf': tfidf_feature_names_all,
            'sbert_pca': sbert_pca_names
        }
    }

    model_file = 'nlp_models_with_sbert_pca.pkl'

    with open(model_file, 'wb') as f:
        pickle.dump(models, f)

    print(f"\n✓ Saved models to: {model_file}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("✅ FEATURE EXTRACTION COMPLETE!")
    print("=" * 80)

    print(f"""
📈 Final Results:

Input:  salary_data_ready_for_new_nlp.csv ({len(data):,} records)
Output: {output_file} ({len(data):,} records, {len(all_feature_names):,} features)

Feature Breakdown:
  ✓ Original features:     {len(original_feature_names):>4,} (baseline features)
  ✓ TF-IDF features:       {len(tfidf_feature_names_all):>4,} (no PCA applied)
  ✓ SBERT PCA features:    {len(sbert_pca_names):>4,} (reduced from {len(sbert_feature_names_all):,})
  {'─' * 60}
  ✓ Total features:        {len(all_feature_names):>4,}

Dimensionality Reduction:
  • SBERT before PCA:      {len(sbert_feature_names_all):,} dimensions
  • SBERT after PCA:       {len(sbert_pca_names):,} dimensions
  • Reduction:             {len(sbert_feature_names_all) - len(sbert_pca_names):,} dimensions ({(1 - (len(sbert_pca_names)/len(sbert_feature_names_all)))*100:.1f}%)
  • Variance preserved:    {SBERT_PCA_VARIANCE*100:.0f}%

Expected Performance Impact:
  • Feature count: {len(original_feature_names) + len(tfidf_feature_names_all) + len(sbert_feature_names_all):,} → {len(all_feature_names):,} ({(1 - len(all_feature_names)/(len(original_feature_names) + len(tfidf_feature_names_all) + len(sbert_feature_names_all)))*100:.1f}% reduction)
  • Training time: ~40-50% faster
  • R² score: Likely maintained or slight increase (0.66-0.68)

Next Steps:
  1. Train models using: {output_file}
  2. Compare performance against baseline (no PCA)
  3. Evaluate training time improvement
    """)

    print("=" * 80)
    print()


if __name__ == '__main__':
    main()
