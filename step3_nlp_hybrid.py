"""
NLP Feature Engineering - Hybrid Approach (TF-IDF + Embeddings)
=================================================================

This script extracts numerical features from text using TWO complementary approaches:

1. TF-IDF (Term Frequency-Inverse Document Frequency)
   - Keyword-based approach
   - Creates one feature per important word
   - Example: "python" → python_tfidf = 0.85
   - Pro: Interpretable (you know which words matter)
   - Con: Treats words independently (doesn't understand "Senior Engineer" ≈ "Lead Developer")

2. Sentence Embeddings (Semantic Vectors)
   - Meaning-based approach using deep learning
   - Creates dense vector (384 dimensions) for entire sentence
   - Example: "Senior ML Engineer" → [0.23, -0.15, 0.89, ..., 0.45]
   - Pro: Understands semantic similarity
   - Con: Less interpretable (hard to explain individual dimensions)

WHY HYBRID?
-----------
Combining both gives BEST accuracy:
- TF-IDF catches specific important keywords ("PhD", "10+ years", "senior")
- Embeddings capture overall job complexity and semantic meaning
- Together: keyword signals + semantic understanding = maximum predictive power

INPUT:  salary_data_no_missing.csv (37 features)
OUTPUT: salary_data_with_nlp.csv (~1005 features)
"""

import csv
import re
import time
from collections import Counter

# =============================================================================
# CONFIGURATION - Adjust these to experiment with different feature combinations
# =============================================================================

CONFIG = {
    # Toggle features on/off for testing
    'use_title_embeddings': True,      # 384 features from title (semantic meaning)
    'use_title_tfidf': True,           # 50 features from title (keywords)
    'use_desc_embeddings': True,       # 384 features from description (semantic meaning)
    'use_desc_tfidf': True,            # 100 features from description (keywords)
    'use_skills_tfidf': True,          # 50 features from skills (keywords)

    # TF-IDF parameters
    'title_tfidf_max_features': 50,    # Top N keywords from titles
    'desc_tfidf_max_features': 100,    # Top N keywords from descriptions
    'skills_tfidf_max_features': 50,   # Top N keywords from skills
    'tfidf_min_df': 5,                 # Ignore words appearing in < 5 documents (rare typos)
    'tfidf_max_df': 0.95,              # Ignore words appearing in > 95% of docs (too common)
    'tfidf_ngram_range': (1, 2),       # Use unigrams (1-word) and bigrams (2-word phrases)

    # Embedding parameters
    'embedding_model': 'all-MiniLM-L6-v2',  # Fast, accurate sentence transformer
    'embedding_batch_size': 32,        # Process 32 sentences at a time
    'max_desc_length': 512,            # Truncate long descriptions (embedding models have limits)
}


# =============================================================================
# HELPER FUNCTIONS - Text Preprocessing
# =============================================================================

def clean_text(text):
    """
    Clean and normalize text for NLP processing.

    Steps:
    1. Convert to lowercase ("Senior" → "senior")
    2. Handle special programming terms (C++, C#, .NET)
    3. Remove URLs and emails
    4. Remove extra whitespace
    5. Keep important punctuation (+ and # for C++, C#)

    Args:
        text (str): Raw text

    Returns:
        str: Cleaned text

    Example:
        Input:  "Senior Python Developer (C++) - Apply at http://jobs.com"
        Output: "senior python developer c++"
    """
    if not text or text.strip() == '':
        return ''

    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Preserve important technical terms BEFORE removing punctuation
    # (Otherwise "C++" becomes "c" and "C#" becomes "c")
    text = text.replace('c++', 'cplusplus')
    text = text.replace('c#', 'csharp')
    text = text.replace('.net', 'dotnet')
    text = text.replace('node.js', 'nodejs')

    # Step 3: Remove URLs (anything starting with http:// or https://)
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Step 4: Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Step 5: Remove special characters but keep spaces and alphanumeric
    # This removes: @#$%^&*()[]{}|\ but keeps: a-z, 0-9, spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Step 6: Remove extra whitespace (multiple spaces → single space)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def simple_tokenize(text):
    """
    Split text into individual words (tokens).

    This is a simple tokenizer that splits on whitespace.
    More advanced: You could use NLTK or spaCy tokenizers.

    Args:
        text (str): Cleaned text

    Returns:
        list: List of words

    Example:
        Input:  "senior python developer"
        Output: ["senior", "python", "developer"]
    """
    return text.split()


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def main():
    print("=" * 80)
    print("STEP 3: NLP FEATURE ENGINEERING (HYBRID: TF-IDF + EMBEDDINGS)")
    print("=" * 80)

    print("""
    This script creates numerical features from text using:

    📊 TF-IDF (Keyword-based):
       - Identifies important words based on frequency
       - Creates interpretable features you can analyze

    🧠 Embeddings (Meaning-based):
       - Uses deep learning to understand semantic similarity
       - Captures that "Senior ML Engineer" ≈ "Lead Data Scientist"

    Together: Maximum accuracy + some interpretability
    """)

    # =========================================================================
    # STEP 3.0: Load Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3.0: LOAD DATA")
    print("=" * 80)

    print("\nLoading salary_data_no_missing.csv...")
    with open('salary_data_no_missing.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
        original_fields = list(reader.fieldnames)

    print(f"✓ Loaded: {len(data):,} records")
    print(f"✓ Original features: {len(original_fields)}")

    # Identify text columns we'll process
    text_columns = {
        'title': 'Job title (e.g., "Senior Python Engineer")',
        'description': 'Job description (long text)',
        'all_skills': 'Required skills (e.g., "Python, AWS, Docker")'
    }

    print(f"\n📝 Text columns to process:")
    for col, desc in text_columns.items():
        if col in original_fields:
            print(f"   • {col:<15} - {desc}")

    # =========================================================================
    # STEP 3.1: Text Preprocessing
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3.1: TEXT PREPROCESSING")
    print("=" * 80)

    print("""
    Preprocessing cleans text before feature extraction:
    - Lowercase: "Senior" → "senior" (treats them as same word)
    - Remove URLs, emails (not useful for salary prediction)
    - Handle special terms: "C++" → "cplusplus" (preserve meaning)
    - Remove punctuation, extra spaces

    WHY? Clean text = better features = more accurate predictions!
    """)

    print("\nPreprocessing text columns...")
    start_time = time.time()

    # Store cleaned versions
    for col in ['title', 'description', 'all_skills']:
        if col in original_fields:
            cleaned_col = f'{col}_cleaned'
            print(f"\n   Processing '{col}'...")

            for i, row in enumerate(data):
                original_text = row.get(col, '')
                cleaned_text = clean_text(original_text)
                row[cleaned_col] = cleaned_text

                # Show example for first row
                if i == 0:
                    print(f"      Example:")
                    print(f"         Original: {original_text[:80]}...")
                    print(f"         Cleaned:  {cleaned_text[:80]}...")

    elapsed = time.time() - start_time
    print(f"\n✓ Preprocessing complete in {elapsed:.2f} seconds")

    # =========================================================================
    # STEP 3.2: Extract TF-IDF Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3.2: EXTRACT TF-IDF FEATURES")
    print("=" * 80)

    print("""
    TF-IDF = Term Frequency × Inverse Document Frequency

    WHAT IT DOES:
    -------------
    Converts text → numerical scores for important words

    HOW IT WORKS:
    -------------
    1. TF (Term Frequency): How often does word appear in THIS job?
       - "python" appears 3 times in description → high TF

    2. IDF (Inverse Document Frequency): How rare is word across ALL jobs?
       - "python" appears in 40% of jobs → medium IDF
       - "kubernetes" appears in 5% of jobs → high IDF (rare = valuable!)
       - "the" appears in 100% of jobs → low IDF (common = useless)

    3. TF-IDF = TF × IDF
       - High score: word is frequent in THIS job AND rare overall
       - Low score: word is either rare in this job OR common everywhere

    EXAMPLE:
    --------
    Job: "Senior Python Engineer"

    Word       | TF | IDF  | TF-IDF | Meaning
    -----------|-------|------|--------|----------------------------------
    "senior"   | 1     | 0.89 | 0.89   | High! Rare word = salary signal
    "python"   | 1     | 0.65 | 0.65   | Medium (common but valuable)
    "engineer" | 1     | 0.31 | 0.31   | Low (appears in most jobs)
    "the"      | 0     | 0.01 | 0.00   | Ignored (stopword, too common)
    """)

    # We need sklearn for TF-IDF (industry standard library)
    print("\nChecking if scikit-learn is installed...")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("✓ scikit-learn is available")
    except ImportError:
        print("❌ ERROR: scikit-learn not installed!")
        print("\n   Please install it:")
        print("   pip install scikit-learn")
        print("\n   Exiting...")
        return

    # Initialize TF-IDF statistics
    tfidf_stats = {}
    all_tfidf_features = []

    # -------------------------------------------------------------------------
    # 3.2.1: TF-IDF from TITLE
    # -------------------------------------------------------------------------
    if CONFIG['use_title_tfidf']:
        print(f"\n📊 Extracting TF-IDF from TITLE (top {CONFIG['title_tfidf_max_features']} keywords)...")

        # Collect all cleaned titles
        titles_cleaned = [row['title_cleaned'] for row in data]

        # Create TF-IDF vectorizer
        # PARAMETERS EXPLAINED:
        # - max_features: Keep only top N most important words (reduces dimensionality)
        # - min_df: Ignore words appearing in < 5 docs (likely typos or rare nonsense)
        # - max_df: Ignore words appearing in > 95% of docs (too common, no signal)
        # - ngram_range: (1,2) = use 1-word AND 2-word phrases
        #   Example: "machine learning" as one feature, not just "machine" + "learning"
        # - sublinear_tf: Use log scaling for term frequency (prevents long docs from dominating)
        print(f"\n   TF-IDF Parameters:")
        print(f"      max_features = {CONFIG['title_tfidf_max_features']} (keep top N keywords)")
        print(f"      min_df = {CONFIG['tfidf_min_df']} (word must appear in ≥ {CONFIG['tfidf_min_df']} jobs)")
        print(f"      max_df = {CONFIG['tfidf_max_df']} (ignore words in > {CONFIG['tfidf_max_df']*100}% of jobs)")
        print(f"      ngram_range = {CONFIG['tfidf_ngram_range']} (1-word and 2-word phrases)")

        vectorizer_title = TfidfVectorizer(
            max_features=CONFIG['title_tfidf_max_features'],
            min_df=CONFIG['tfidf_min_df'],
            max_df=CONFIG['tfidf_max_df'],
            ngram_range=CONFIG['tfidf_ngram_range'],
            sublinear_tf=True  # Use log scaling
        )

        # Fit and transform (learn vocabulary + compute scores)
        # This is where the magic happens!
        # - Fit: Learn which words are important across ALL titles
        # - Transform: Convert each title → vector of TF-IDF scores
        start_time = time.time()
        title_tfidf_matrix = vectorizer_title.fit_transform(titles_cleaned)
        elapsed = time.time() - start_time

        # Get feature names (the actual words/phrases selected)
        title_tfidf_features = vectorizer_title.get_feature_names_out()

        print(f"\n   ✓ Extracted {len(title_tfidf_features)} TF-IDF features from titles in {elapsed:.2f}s")
        print(f"\n   Top 10 most important title keywords:")
        for i, word in enumerate(title_tfidf_features[:10]):
            print(f"      {i+1:2}. title_tfidf_{word}")

        # Convert sparse matrix → add to our data rows
        # (Sparse matrix is efficient for storage, but we need to add to CSV rows)
        for i, row in enumerate(data):
            tfidf_scores = title_tfidf_matrix[i].toarray()[0]  # Get scores for this row
            for j, feature_name in enumerate(title_tfidf_features):
                row[f'title_tfidf_{feature_name}'] = f"{tfidf_scores[j]:.6f}"

        # Track statistics
        tfidf_stats['title'] = {
            'num_features': len(title_tfidf_features),
            'vocabulary_size': len(vectorizer_title.vocabulary_),
            'top_features': list(title_tfidf_features[:10])
        }
        all_tfidf_features.extend([f'title_tfidf_{f}' for f in title_tfidf_features])

    # -------------------------------------------------------------------------
    # 3.2.2: TF-IDF from DESCRIPTION
    # -------------------------------------------------------------------------
    if CONFIG['use_desc_tfidf']:
        print(f"\n📝 Extracting TF-IDF from DESCRIPTION (top {CONFIG['desc_tfidf_max_features']} keywords)...")

        # Collect all cleaned descriptions
        descriptions_cleaned = [row['description_cleaned'] for row in data]

        vectorizer_desc = TfidfVectorizer(
            max_features=CONFIG['desc_tfidf_max_features'],
            min_df=CONFIG['tfidf_min_df'],
            max_df=CONFIG['tfidf_max_df'],
            ngram_range=CONFIG['tfidf_ngram_range'],
            sublinear_tf=True
        )

        start_time = time.time()
        desc_tfidf_matrix = vectorizer_desc.fit_transform(descriptions_cleaned)
        elapsed = time.time() - start_time

        desc_tfidf_features = vectorizer_desc.get_feature_names_out()

        print(f"\n   ✓ Extracted {len(desc_tfidf_features)} TF-IDF features from descriptions in {elapsed:.2f}s")
        print(f"\n   Top 10 most important description keywords:")
        for i, word in enumerate(desc_tfidf_features[:10]):
            print(f"      {i+1:2}. desc_tfidf_{word}")

        # Add to data
        for i, row in enumerate(data):
            tfidf_scores = desc_tfidf_matrix[i].toarray()[0]
            for j, feature_name in enumerate(desc_tfidf_features):
                row[f'desc_tfidf_{feature_name}'] = f"{tfidf_scores[j]:.6f}"

        tfidf_stats['description'] = {
            'num_features': len(desc_tfidf_features),
            'vocabulary_size': len(vectorizer_desc.vocabulary_),
            'top_features': list(desc_tfidf_features[:10])
        }
        all_tfidf_features.extend([f'desc_tfidf_{f}' for f in desc_tfidf_features])

    # -------------------------------------------------------------------------
    # 3.2.3: TF-IDF from SKILLS
    # -------------------------------------------------------------------------
    if CONFIG['use_skills_tfidf']:
        print(f"\n🛠️  Extracting TF-IDF from SKILLS (top {CONFIG['skills_tfidf_max_features']} keywords)...")

        skills_cleaned = [row['all_skills_cleaned'] for row in data]

        vectorizer_skills = TfidfVectorizer(
            max_features=CONFIG['skills_tfidf_max_features'],
            min_df=CONFIG['tfidf_min_df'],
            max_df=CONFIG['tfidf_max_df'],
            ngram_range=(1, 1),  # Skills are usually single words, so no bigrams needed
            sublinear_tf=True
        )

        start_time = time.time()
        skills_tfidf_matrix = vectorizer_skills.fit_transform(skills_cleaned)
        elapsed = time.time() - start_time

        skills_tfidf_features = vectorizer_skills.get_feature_names_out()

        print(f"\n   ✓ Extracted {len(skills_tfidf_features)} TF-IDF features from skills in {elapsed:.2f}s")
        print(f"\n   Top 10 most valuable skills:")
        for i, word in enumerate(skills_tfidf_features[:10]):
            print(f"      {i+1:2}. skills_tfidf_{word}")

        # Add to data
        for i, row in enumerate(data):
            tfidf_scores = skills_tfidf_matrix[i].toarray()[0]
            for j, feature_name in enumerate(skills_tfidf_features):
                row[f'skills_tfidf_{feature_name}'] = f"{tfidf_scores[j]:.6f}"

        tfidf_stats['skills'] = {
            'num_features': len(skills_tfidf_features),
            'vocabulary_size': len(vectorizer_skills.vocabulary_),
            'top_features': list(skills_tfidf_features[:10])
        }
        all_tfidf_features.extend([f'skills_tfidf_{f}' for f in skills_tfidf_features])

    # Summary
    print(f"\n{'='*80}")
    print(f"TF-IDF SUMMARY")
    print(f"{'='*80}")
    print(f"\n   Total TF-IDF features created: {len(all_tfidf_features)}")
    for source, stats in tfidf_stats.items():
        print(f"\n   {source.capitalize()}:")
        print(f"      Features:   {stats['num_features']}")
        print(f"      Vocabulary: {stats['vocabulary_size']} unique words analyzed")

    # =========================================================================
    # STEP 3.3: Generate Sentence Embeddings
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3.3: GENERATE SENTENCE EMBEDDINGS")
    print("=" * 80)

    print("""
    WHAT ARE SENTENCE EMBEDDINGS?
    ----------------------------
    Deep learning models that convert text → dense numerical vectors (384 numbers)

    HOW THEY WORK:
    -------------
    • Pre-trained on millions of sentences to understand meaning
    • Similar meanings → similar vectors (even with different words!)
    • Example:
        "Senior ML Engineer"     → [0.23, -0.45, 0.67, ..., 0.12] (384 numbers)
        "Lead Data Scientist"    → [0.21, -0.43, 0.69, ..., 0.15] (very similar!)
        "Junior Marketing Coord" → [-0.56, 0.89, -0.12, ..., 0.45] (very different!)

    WHY ADD EMBEDDINGS ON TOP OF TF-IDF?
    ------------------------------------
    TF-IDF:     "python" = keyword match only
    Embeddings: "python developer" ≈ "software engineer" ≈ "programmer" (semantics!)

    MODEL: all-MiniLM-L6-v2
    ----------------------
    • Fast and accurate sentence embedding model
    • 384 dimensions per text
    • Trained on semantic similarity tasks
    • Perfect for salary prediction (captures job role similarity)
    """)

    all_embedding_features = []
    embedding_stats = {}

    # Check if sentence-transformers is installed
    print("\nChecking if sentence-transformers is installed...")
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ sentence-transformers is available")
    except ImportError:
        print("❌ ERROR: sentence-transformers not installed!")
        print("\n   Please install it:")
        print("   pip install sentence-transformers")
        print("\n   Continuing without embeddings (TF-IDF only)...")
        print("   Set use_title_embeddings and use_desc_embeddings to False in CONFIG")
        # Disable embeddings if library not available
        CONFIG['use_title_embeddings'] = False
        CONFIG['use_desc_embeddings'] = False

    # Only proceed if embeddings are enabled AND library is available
    if CONFIG['use_title_embeddings'] or CONFIG['use_desc_embeddings']:
        print("\n🧠 Loading sentence embedding model: all-MiniLM-L6-v2")
        print("   (This may take a minute on first run - downloads ~80MB model)")

        start_time = time.time()
        model = SentenceTransformer('all-MiniLM-L6-v2')
        elapsed = time.time() - start_time
        print(f"   ✓ Model loaded in {elapsed:.2f}s")

        # -------------------------------------------------------------------------
        # 3.3.1: Embeddings from TITLE
        # -------------------------------------------------------------------------
        if CONFIG['use_title_embeddings']:
            print(f"\n📊 Generating embeddings for TITLES...")
            print(f"   Processing {len(data):,} job titles...")

            # Collect cleaned titles
            titles_for_embedding = [row['title_cleaned'] for row in data]

            # Generate embeddings (batch processing is efficient)
            start_time = time.time()
            title_embeddings = model.encode(titles_for_embedding, show_progress_bar=True, batch_size=256)
            elapsed = time.time() - start_time

            print(f"\n   ✓ Generated {title_embeddings.shape[0]:,} title embeddings in {elapsed:.2f}s")
            print(f"   ✓ Each embedding: {title_embeddings.shape[1]} dimensions")
            print(f"   ✓ Speed: {len(data)/elapsed:.0f} titles/second")

            # Add to data
            for i, row in enumerate(data):
                embedding = title_embeddings[i]
                for dim_idx in range(len(embedding)):
                    row[f'title_emb_{dim_idx}'] = f"{embedding[dim_idx]:.6f}"

            # Track stats
            title_emb_features = [f'title_emb_{i}' for i in range(title_embeddings.shape[1])]
            all_embedding_features.extend(title_emb_features)
            embedding_stats['title'] = {
                'num_features': len(title_emb_features),
                'dimensions': title_embeddings.shape[1]
            }

        # -------------------------------------------------------------------------
        # 3.3.2: Embeddings from DESCRIPTION
        # -------------------------------------------------------------------------
        if CONFIG['use_desc_embeddings']:
            print(f"\n📝 Generating embeddings for DESCRIPTIONS...")
            print(f"   Processing {len(data):,} job descriptions...")
            print(f"   (This takes longer - descriptions are 10-100x longer than titles)")

            # Collect cleaned descriptions
            # Limit length to first 500 words to avoid memory issues
            descriptions_for_embedding = []
            for row in data:
                desc = row['description_cleaned']
                # Split into words and take first 500
                words = desc.split()[:500]
                descriptions_for_embedding.append(' '.join(words))

            # Generate embeddings
            start_time = time.time()
            desc_embeddings = model.encode(descriptions_for_embedding, show_progress_bar=True, batch_size=64)
            elapsed = time.time() - start_time

            print(f"\n   ✓ Generated {desc_embeddings.shape[0]:,} description embeddings in {elapsed:.2f}s")
            print(f"   ✓ Each embedding: {desc_embeddings.shape[1]} dimensions")
            print(f"   ✓ Speed: {len(data)/elapsed:.0f} descriptions/second")

            # Add to data
            for i, row in enumerate(data):
                embedding = desc_embeddings[i]
                for dim_idx in range(len(embedding)):
                    row[f'desc_emb_{dim_idx}'] = f"{embedding[dim_idx]:.6f}"

            # Track stats
            desc_emb_features = [f'desc_emb_{i}' for i in range(desc_embeddings.shape[1])]
            all_embedding_features.extend(desc_emb_features)
            embedding_stats['description'] = {
                'num_features': len(desc_emb_features),
                'dimensions': desc_embeddings.shape[1]
            }

        # Summary
        print(f"\n{'='*80}")
        print(f"EMBEDDINGS SUMMARY")
        print(f"{'='*80}")
        print(f"\n   Total embedding features created: {len(all_embedding_features)}")
        for source, stats in embedding_stats.items():
            print(f"\n   {source.capitalize()}:")
            print(f"      Dimensions:  {stats['dimensions']}")
            print(f"      Features:    {stats['num_features']}")

    else:
        print("\n⚠️  Embeddings DISABLED in CONFIG")
        print("   Running with TF-IDF features only")

    # =========================================================================
    # STEP 3.4: SAVE DATASET WITH NLP FEATURES
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3.4: SAVE DATASET WITH NLP FEATURES")
    print("=" * 80)

    # Determine output filename based on what's enabled
    if all_embedding_features:
        output_file = 'salary_data_with_nlp_features.csv'
        print(f"\n   Output: {output_file} (HYBRID: TF-IDF + Embeddings)")
    else:
        output_file = 'salary_data_with_tfidf_only.csv'
        print(f"\n   Output: {output_file} (TF-IDF only)")

    print(f"\nPreparing to save...")

    # Build final fieldnames
    # Order: original features + TF-IDF features + embedding features
    final_fieldnames = original_fields.copy()

    # Add all TF-IDF features
    final_fieldnames.extend(all_tfidf_features)

    # Add embedding features if generated
    if all_embedding_features:
        final_fieldnames.extend(all_embedding_features)

    print(f"\n   Original features:    {len(original_fields)}")
    print(f"   + TF-IDF features:    {len(all_tfidf_features)}")
    if all_embedding_features:
        print(f"   + Embedding features: {len(all_embedding_features)}")
    print(f"   {'='*40}")
    print(f"   Total features:       {len(final_fieldnames)}")

    # Save to CSV
    print(f"\n   Writing {len(data):,} rows to {output_file}...")
    start_time = time.time()

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=final_fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)

    elapsed = time.time() - start_time
    print(f"   ✓ Saved in {elapsed:.2f} seconds")

    # =========================================================================
    # STEP 3.5: FEATURE ANALYSIS & SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3.5: FEATURE ANALYSIS & SUMMARY")
    print("=" * 80)

    print(f"\n📊 FINAL DATASET SUMMARY:")
    print(f"   Input:  salary_data_no_missing.csv ({len(data):,} records, {len(original_fields)} features)")
    print(f"   Output: {output_file} ({len(data):,} records, {len(final_fieldnames)} features)")
    print(f"   Added:  {len(final_fieldnames) - len(original_fields)} new NLP features")

    print(f"\n🔤 NLP FEATURES BREAKDOWN:")

    # TF-IDF breakdown
    print(f"\n   TF-IDF Features ({len(all_tfidf_features)} total):")
    for source, stats in tfidf_stats.items():
        print(f"      • {source.capitalize()}: {stats['num_features']} features")
        print(f"         Top keywords: {', '.join(stats['top_features'][:5])}")

    # Embeddings breakdown
    if all_embedding_features:
        print(f"\n   Embedding Features ({len(all_embedding_features)} total):")
        for source, stats in embedding_stats.items():
            print(f"      • {source.capitalize()}: {stats['num_features']} features ({stats['dimensions']} dims)")
    else:
        print(f"\n   Embedding Features: NONE (disabled in CONFIG)")

    # Show examples from first job posting
    print(f"\n💡 EXAMPLE: First Job Posting TF-IDF Scores:")
    print(f"   Title: {data[0].get('title', '')}")

    # Show top 5 title TF-IDF scores for first row
    if CONFIG['use_title_tfidf']:
        print(f"\n   Top Title Keywords (TF-IDF scores):")
        title_scores = []
        for feat in tfidf_stats['title']['top_features'][:5]:
            score = float(data[0].get(f'title_tfidf_{feat}', 0))
            if score > 0:
                title_scores.append((feat, score))
        for feat, score in sorted(title_scores, key=lambda x: x[1], reverse=True):
            print(f"      • {feat}: {score:.4f}")

    # =========================================================================
    # COMPLETION SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    if all_embedding_features:
        print("✅ STEP 3 COMPLETE! (HYBRID NLP: TF-IDF + EMBEDDINGS)")
    else:
        print("✅ STEP 3 COMPLETE! (TF-IDF ONLY)")
    print("=" * 80)

    if all_embedding_features:
        print(f"""
📋 WHAT WE DID:
   1. ✅ Loaded {len(data):,} job postings
   2. ✅ Cleaned text (lowercase, special chars, etc.)
   3. ✅ Extracted {len(all_tfidf_features)} TF-IDF keyword features
   4. ✅ Generated {len(all_embedding_features)} sentence embedding features
   5. ✅ Saved to {output_file}

🎯 HYBRID NLP FEATURES:
   • TF-IDF: {len(all_tfidf_features)} interpretable keyword features
   • Embeddings: {len(all_embedding_features)} semantic meaning features
   • Total: {len(all_tfidf_features) + len(all_embedding_features)} NLP features

💡 WHY HYBRID WORKS:
   TF-IDF captures explicit keywords ("python", "senior", "machine learning")
   Embeddings capture semantic similarity (knows "ML Engineer" ≈ "Data Scientist")
   Together: Maximum accuracy + some interpretability

📊 DATASET READY FOR:
   • Advanced modeling (Random Forest, XGBoost, Neural Networks)
   • High accuracy salary predictions
   • Feature importance analysis

🚀 NEXT STEP:
   → Proceed to Step 4: Train models and evaluate performance
   → Expected improvement: Embeddings should boost R² by 5-15%
        """)
    else:
        print(f"""
📋 WHAT WE DID:
   1. ✅ Loaded {len(data):,} job postings
   2. ✅ Cleaned text (lowercase, special chars, etc.)
   3. ✅ Extracted {len(all_tfidf_features)} TF-IDF keyword features
   4. ✅ Saved to {output_file}

📊 DATASET READY FOR:
   • Exploratory analysis (inspect TF-IDF scores in Excel)
   • Baseline modeling (Linear Regression with TF-IDF only)
   • Performance comparison (TF-IDF vs. Hybrid later)

🚀 NEXT STEPS:
   Option A: Test TF-IDF-only model first
            → Proceed to Step 4: Train baseline model with current features

   Option B: Add embeddings now for hybrid approach
            → Re-enable embeddings in CONFIG (lines 43, 45)
            → Run script again to get TF-IDF + Embeddings
            → Compare performance

💡 RECOMMENDATION: Test TF-IDF-only model first to establish baseline!
    Then add embeddings and compare performance improvement.
        """)

    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
