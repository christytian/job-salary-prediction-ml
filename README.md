# Salary Prediction with NLP Feature Engineering

A machine learning project for predicting job salaries using hybrid NLP features (TF-IDF + Sentence Embeddings).

## Project Overview

This project builds a salary prediction model using job posting data. The pipeline includes:
- Data cleaning and normalization
- Missing value handling with missingness indicators
- Advanced NLP feature engineering (TF-IDF + Sentence Embeddings)
- Ready for ML modeling (Random Forest, XGBoost, Neural Networks)

## Pipeline Steps

### Step 0: Create Final Dataset
**Script**: `create_final_dataset.py`
- Input: `salary_data_cleaned.csv`
- Output: `salary_data_final.csv`
- Creates normalized salary target variable
- Drops non-essential features (IDs, URLs, timestamps)

### Step 1: Analyze Features
**Script**: `step1_analyze_features.py`
- Input: `salary_data_final.csv`
- Analyzes missing values and categorizes into tiers:
  - **Tier 1** (≥90% missing): Drop
  - **Tier 2** (30-90% missing): Impute + add indicator
  - **Tier 3** (<30% missing): Impute only

### Step 2: Handle Missing Values
**Script**: `step2_handle_missing_values.py`
- Input: `salary_data_final.csv`
- Output: `salary_data_no_missing.csv`
- Implements "impute + indicator" approach
- Creates binary `_missing` columns for Tier 2 features
- Preserves MNAR (Missing Not At Random) patterns

### Step 3: NLP Feature Engineering (HYBRID)
**Script**: `step3_nlp_hybrid.py`
- Input: `salary_data_no_missing.csv`
- Output: `salary_data_with_nlp_features.csv`
- **TF-IDF Features**: 199 keyword-based features
  - 50 from job titles
  - 100 from job descriptions
  - 49 from skills lists
- **Sentence Embeddings**: 768 semantic features
  - 384 from job titles (all-MiniLM-L6-v2)
  - 384 from job descriptions (all-MiniLM-L6-v2)
- **Total**: 1,004 features (37 original + 967 NLP)

## Features

### Hybrid NLP Approach
- **TF-IDF**: Captures explicit keywords ("python", "senior", "machine learning")
- **Embeddings**: Captures semantic meaning (knows "ML Engineer" ≈ "Data Scientist")
- **Why Both?**: Maximum accuracy + interpretability

### Missingness Indicators
- Binary columns that preserve the signal of missing values
- Models can learn patterns like "missing applies → newer jobs → different salary"
- Works great with tree-based models (Random Forest, XGBoost)

## Final Dataset

**File**: `salary_data_with_nlp_features.csv` (487MB)
- **Records**: 35,385 job postings
- **Features**: 1,004 total
  - 37 original features (numerical, categorical, text)
  - 199 TF-IDF features
  - 768 embedding features
- **Target**: `salary_normalized` (median annual salary)

## Requirements

```bash
pip install scikit-learn sentence-transformers
```

## Usage

Run the pipeline in order:

```bash
# Step 0: Create final dataset
python create_final_dataset.py

# Step 1: Analyze features (optional - just analysis)
python step1_analyze_features.py

# Step 2: Handle missing values
python step2_handle_missing_values.py

# Step 3: Extract NLP features (TF-IDF + Embeddings)
python step3_nlp_hybrid.py
```

## Configuration

Edit `step3_nlp_hybrid.py` CONFIG section to toggle features:

```python
CONFIG = {
    'use_title_embeddings': True,      # 384 features
    'use_title_tfidf': True,           # 50 features
    'use_desc_embeddings': True,       # 384 features
    'use_desc_tfidf': True,            # 100 features
    'use_skills_tfidf': True,          # 50 features

    'title_tfidf_max_features': 50,
    'desc_tfidf_max_features': 100,
    'skills_tfidf_max_features': 50,
    'tfidf_min_df': 5,
    'tfidf_max_df': 0.95,
    'tfidf_ngram_range': (1, 2),
}
```

## Next Steps

1. Train baseline model (Linear Regression with TF-IDF only)
2. Train advanced models (Random Forest, XGBoost)
3. Compare performance: TF-IDF vs. Hybrid approach
4. Feature importance analysis
5. Hyperparameter tuning

## Technical Details

- **Embedding Model**: all-MiniLM-L6-v2
  - Fast (2,853 titles/sec, 272 descriptions/sec)
  - Accurate (384 dimensions)
  - Pre-trained on semantic similarity tasks

- **TF-IDF Parameters**:
  - min_df=5: Ignore words in <5 jobs (rare typos)
  - max_df=0.95: Ignore words in >95% of jobs (too common)
  - ngram_range=(1,2): Unigrams + bigrams

## Data Files (Not in Git)

**Note**: CSV files are excluded from git due to size (100MB+ each)

Required input files:
- `salary_data_cleaned.csv` (195MB) - Initial cleaned data
- Or start from `postings.csv` (493MB) - Raw data

Intermediate files:
- `salary_data_final.csv` (181MB)
- `salary_data_no_missing.csv` (181MB)

Final output:
- `salary_data_with_nlp_features.csv` (487MB)

## License

MIT

## Author

CS6140 Machine Learning Project
