# Salary Prediction with NLP Feature Engineering

A machine learning project for predicting job salaries using hybrid NLP features (TF-IDF + Sentence Embeddings).

## Project Overview

This project builds a salary prediction model using job posting data. The pipeline includes:
- Data cleaning and normalization
- Missing value handling with missingness indicators
- Advanced NLP feature engineering (TF-IDF + Sentence Embeddings)
- Ready for ML modeling (Random Forest, XGBoost, Neural Networks)

## Pipeline Steps

**Scientific Workflow**: This pipeline follows proper scientific methodology by first comparing different NLP approaches empirically (Step 5) before training the final production model (Step 6) with the proven winner. This ensures our modeling decisions are data-driven and reproducible.

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

### Step 4: Prepare for Linear Regression
**Script**: `step4_prepare_for_linear_regression.py`
- Input: `salary_data_with_nlp_features.csv`
- Output: `salary_data_lr_ready.csv`
- **Validates critical fields**:
  - Title: min 3 characters
  - Description: min 20 characters
  - Salary: $10K - $1M range
- **Removes data leakage**: Excludes `normalized_salary` and other salary-related columns
- **Drops non-numeric columns**: Removes text/categorical features (already encoded in NLP)
- **Result**: 35,368 records × 982 numeric features + 1 target

### Step 5: Comprehensive NLP Validation (Empirical Comparison)
**Script**: `step5_compare_nlp_approaches.py`
- Input: `salary_data_with_nlp_features.csv`
- **Purpose**: Empirically determine which NLP approach performs best for salary prediction

#### Methodology
- **Evaluation**: 10-fold cross-validation for robust statistical comparison
- **Model**: Linear Regression (fair comparison across all methods)
- **Metrics**: R² (variance explained), RMSE (absolute error), MAE (mean error)
- **Dataset**: 35,385 job postings with text from titles, descriptions, and skills

#### NLP Methods Compared

**1. TF-IDF Only (Keyword-Based)**
- Counts word frequencies weighted by document rarity
- Features: 199 dimensions (50 from titles, 100 from descriptions, 49 from skills)
- Pros: Interpretable, fast, captures important keywords
- Cons: No semantic understanding (treats "engineer" and "developer" as different)

**2. Word2Vec Only (Word-Level Embeddings)**
- Trained custom Word2Vec model (Skip-gram, 100 dimensions, 91K vocabulary)
- Document vectors = averaged word embeddings
- Training: 342s on 35K documents, window=5, min_count=5
- Pros: Captures word similarity
- Cons: Averaging loses sentence structure, trained from scratch (limited data)

**3. SBERT Only (Sentence-Level Embeddings)**
- Uses pre-trained all-MiniLM-L6-v2 transformer model
- Features: 768 dimensions (384 from titles, 384 from descriptions)
- Pros: Understands sentence semantics, pre-trained on massive corpus
- Cons: Black box, higher dimensionality

**4. TF-IDF + Word2Vec (Hybrid 1)**
- Concatenates TF-IDF (199) + Word2Vec (100) = 299 features
- Combines keyword matching with word-level semantics

**5. TF-IDF + SBERT (Hybrid 2 - Our Approach)**
- Concatenates TF-IDF (199) + SBERT (768) = 967 features
- Combines keyword matching with sentence-level semantics

#### Results (10-Fold Cross-Validation)

| Approach | Features | R² Score | RMSE | MAE | Interpretation |
|----------|----------|----------|------|-----|----------------|
| TF-IDF Only | 199 | 0.4081 ± 0.0441 | $42,533 | $29,523 | Baseline keyword matching |
| Word2Vec Only | 100 | 0.4211 ± 0.0451 | $42,051 | $29,384 | Word similarity helps slightly |
| SBERT Only | 768 | 0.5152 ± 0.0422 | $38,485 | $26,641 | Strong semantic understanding |
| TF-IDF + Word2Vec | 299 | 0.4956 ± 0.0453 | $39,237 | $26,938 | Hybrid improves over individuals |
| **TF-IDF + SBERT** | **967** | **0.5424 ± 0.0436** | **$37,381** | **$25,785** | **BEST: Keywords + Semantics** |

#### Statistical Analysis

**Winner**: TF-IDF + SBERT (Hybrid 2)
- Explains **54.24% of salary variance** (best across all methods)
- **+32.9%** improvement over TF-IDF only (0.5424 vs 0.4081)
- **+28.8%** improvement over Word2Vec only (0.5424 vs 0.4211)
- **+5.3%** improvement over SBERT only (0.5424 vs 0.5152)
- **+9.4%** improvement over TF-IDF + Word2Vec (0.5424 vs 0.4956)
- Robust across all folds (std = 0.0436, low variance)

#### Key Insights

1. **Sentence-level > Word-level**: SBERT (0.5152) beats Word2Vec (0.4211) by 22.3%
   - Pre-trained transformers > custom Word2Vec on limited data
   - Sentence context matters more than averaged word vectors

2. **Hybrid > Individual**: TF-IDF + SBERT beats SBERT alone by 5.3%
   - TF-IDF captures specific salary-predictive keywords ("director", "senior", "python")
   - SBERT captures semantic role similarity ("data scientist" ≈ "ML engineer")
   - Combined they capture complementary signals

3. **Why TF-IDF + SBERT Wins**:
   - TF-IDF provides interpretability (see which keywords matter for salary)
   - SBERT provides robustness (handles synonyms, paraphrases, semantic similarity)
   - Together they achieve best predictive accuracy

#### Practical Implications

- **Use TF-IDF + SBERT for production models** (empirically validated)
- **Avoid Word2Vec** on small datasets (SBERT's pre-training is superior)
- **Always test hybrid approaches** (may capture complementary signals)
- **Cross-validation is critical** (single train/test split can be misleading)

### Step 6: Train Final Production Model
**Script**: `step6_train_final_model.py`
- Input: `salary_data_lr_ready.csv`
- Trains final linear regression model using the proven best approach (TF-IDF + SBERT)
- **Performance**:
  - R² Score: 0.5574 (explains 55.7% of salary variance)
  - RMSE: $37,119
  - MAE: $25,615 (27% of avg salary)
- **Key Findings**:
  - Top predictive features: business skills, director title, development skills
  - Minimal overfitting (R² diff: 0.0388)
  - Model generalizes well to unseen data

## Evaluation Methodology: Step 5 vs Step 6

**Important**: Both steps use **Linear Regression** as the machine learning model. The difference is in **how we evaluate** the model.

### Step 5: 10-Fold Cross-Validation (Comparison)
**Purpose**: Empirically compare 5 different NLP approaches

**Evaluation Method**: 10-fold cross-validation
- Splits data into 10 equal parts (folds)
- Trains 10 different models, each time using 9 folds for training and 1 fold for testing
- Each fold serves as test data exactly once
- Returns **average** performance across all 10 runs ± standard deviation

**Why use it?**
- **Robust comparison**: Reduces variance from a single random split
- **Statistical reliability**: Standard deviation shows how stable the results are
- **Fair comparison**: All approaches tested under identical conditions
- **Uses all data**: Every sample is used for both training and testing

**Code Example**:
```python
from sklearn.model_selection import cross_val_score

model = LinearRegression()
r2_scores = cross_val_score(model, X, y, cv=10, scoring='r2', n_jobs=-1)

# Returns ARRAYS with 10 values (one per fold)
print(f"R² = {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
# Example: R² = 0.5424 ± 0.0436
```

**Output**: 50 models trained (10 folds × 5 approaches)

### Step 6: Single Train/Test Split (Production Model)
**Purpose**: Train final production model with the proven winner (TF-IDF + SBERT)

**Evaluation Method**: Single 80/20 train/test split
- 80% of data for training
- 20% of data for testing (held out, never seen during training)
- Trains **ONE** model on the training set
- Evaluates on the test set **once**

**Why use it?**
- **Production deployment**: Creates a single model to save and deploy
- **Faster**: No need to train 10 models
- **Interpretable**: Can analyze specific feature importance
- **Decision made**: Already know which approach works best from Step 5

**Code Example**:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)  # Single value

print(f"Test R² = {r2:.4f}")
# Example: Test R² = 0.5574
```

**Output**: 1 deployable model

### Comparison Table

| Aspect | Step 5 (10-Fold CV) | Step 6 (Single Split) |
|--------|---------------------|----------------------|
| **Goal** | Compare 5 NLP approaches | Train final deployable model |
| **Priority** | Statistical robustness | Production readiness |
| **Models trained** | 50 models (10 folds × 5 approaches) | 1 model |
| **Time** | Slower (multiple training runs) | Faster (one training run) |
| **Output** | Mean ± std deviation | Single performance value |
| **Model saved?** | No (just comparison) | Yes (for deployment) |
| **Use case** | Research & validation | Deployment & inference |

### Scientific Workflow Summary

**Step 5**: "Which NLP approach is best?" → Use 10-fold CV for reliable comparison
- TF-IDF only: R² = 0.4081 ± 0.0441
- Word2Vec only: R² = 0.4211 ± 0.0451
- SBERT only: R² = 0.5152 ± 0.0422
- TF-IDF + Word2Vec: R² = 0.4956 ± 0.0453
- **TF-IDF + SBERT: R² = 0.5424 ± 0.0436** ← Winner!

**Step 6**: "Build production model with winner" → Use single split for deployable model
- Train R² = 0.5962
- Test R² = 0.5574 ← Deploy this model

This workflow ensures we **validate first** (Step 5), **then deploy** (Step 6)!

## Features

### Hybrid NLP Approach
- **TF-IDF**: Captures explicit keywords ("python", "senior", "machine learning")
- **Embeddings**: Captures semantic meaning (knows "ML Engineer" ≈ "Data Scientist")
- **Why Both?**: Maximum accuracy + interpretability

### Missingness Indicators
- Binary columns that preserve the signal of missing values
- Models can learn patterns like "missing applies → newer jobs → different salary"
- Works great with tree-based models (Random Forest, XGBoost)

## Final Datasets

**NLP Features**: `salary_data_with_nlp_features.csv` (487MB)
- **Records**: 35,385 job postings
- **Features**: 1,004 total
  - 37 original features (numerical, categorical, text)
  - 199 TF-IDF features
  - 768 embedding features
- **Target**: `salary_normalized` (median annual salary)

**Model-Ready**: `salary_data_lr_ready.csv` (44MB)
- **Records**: 35,368 (17 invalid rows removed)
- **Features**: 982 numeric features + 1 target
  - 199 TF-IDF features
  - 768 embedding features
  - 11 original numeric features
  - 4 missingness indicators
- **Data Quality**: All features numeric, no missing values, validated salary range ($10K-$1M)

## Data Files

**Important**: CSV data files are NOT included in this repository due to GitHub's 100MB file size limit.

### Download Data Files

All required CSV files are available on Google Drive:

**[Download Data Files from Google Drive](https://drive.google.com/drive/folders/14sjOhlRJ2H5N1anGQhx08MhOT3dIsp5e?usp=share_link)**

### Required Files by Step

After downloading, place the CSV files in the project root directory:

- **Step 0**: Requires `salary_data_cleaned.csv` (195MB)
  - Generates: `salary_data_final.csv` (181MB)

- **Step 1**: Requires `salary_data_final.csv`
  - Analysis only, no output file

- **Step 2**: Requires `salary_data_final.csv`
  - Generates: `salary_data_no_missing.csv` (181MB)

- **Step 3**: Requires `salary_data_no_missing.csv`
  - Generates: `salary_data_with_nlp_features.csv` (487MB)

- **Step 4**: Requires `salary_data_with_nlp_features.csv`
  - Generates: `salary_data_lr_ready.csv` (44MB)

- **Step 5**: Requires `salary_data_with_nlp_features.csv`
  - Empirical NLP validation with 10-fold cross-validation
  - Requires gensim for Word2Vec comparison

- **Step 6**: Requires `salary_data_lr_ready.csv`
  - Trains and evaluates final production model

### File Descriptions

| File | Size | Description |
|------|------|-------------|
| `salary_data_cleaned.csv` | 195MB | Initial cleaned dataset (input for pipeline) |
| `salary_data_final.csv` | 181MB | Normalized salary + essential features only |
| `salary_data_no_missing.csv` | 181MB | Missing values handled + missingness indicators |
| `salary_data_with_nlp_features.csv` | 487MB | NLP features added (1,004 features) |
| `salary_data_lr_ready.csv` | 44MB | **Final clean dataset** - 35,368 records × 982 numeric features + target |

**Note**: You can download just `salary_data_cleaned.csv` and regenerate all other files by running the pipeline scripts in order.

## Requirements

```bash
pip install scikit-learn sentence-transformers gensim
```

- `scikit-learn`: Machine learning (Linear Regression, StandardScaler, cross-validation, TF-IDF)
- `sentence-transformers`: Pre-trained SBERT models for semantic embeddings
- `gensim`: Word2Vec training for NLP comparison (Step 5 only)

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

# Step 4: Prepare for linear regression
python step4_prepare_for_linear_regression.py

# Step 5: Compare NLP approaches (empirical validation)
python step5_compare_nlp_approaches.py

# Step 6: Train final production model with proven winner
python step6_train_final_model.py
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

## Model Results

### Baseline Linear Regression (Completed)
- **R² Score**: 0.5574 (explains 55.7% of variance)
- **RMSE**: $37,119
- **MAE**: $25,615 (27% of average salary)
- **Overfitting**: Minimal (R² diff: 0.0388)

### Next Steps for Model Improvement

1. **Regularization**: Ridge/Lasso regression to reduce overfitting
2. **Advanced Models**: Random Forest, XGBoost, Gradient Boosting
3. **Feature Engineering**:
   - Feature interactions (experience × seniority)
   - PCA on embeddings for dimensionality reduction
   - Log transformation of target variable
4. **Diagnostics**:
   - VIF analysis for multicollinearity
   - Residual plots
   - Learning curves
5. **Model Deployment**: Save model, create prediction API

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
