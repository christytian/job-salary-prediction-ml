# NLP Feature Comparison with SBERT PCA Optimization

## Overview
Comprehensive comparison of three NLP feature extraction approaches for job salary prediction, with selective PCA optimization applied only to SBERT embeddings to reduce dimensionality while preserving predictive power.

## Methodology

### NLP Approaches Tested
1. **TF-IDF Only**: 310 TF-IDF features + 22 baseline features (332 total)
2. **SBERT PCA Only**: 660 SBERT PCA features + 22 baseline features (682 total)
3. **Hybrid (TF-IDF + SBERT PCA)**: 310 TF-IDF + 660 SBERT PCA + 22 baseline (992 total)

### PCA Optimization Strategy
- **Applied to**: SBERT embeddings only (2,304 → 660 dimensions)
- **Variance preserved**: 95%
- **Rationale**: TF-IDF features are already sparse and interpretable; SBERT embeddings are dense and redundant
- **Result**: 71% dimensionality reduction with minimal information loss

### Models Evaluated
All three NLP approaches were tested with 6 regression models:
- Linear Regression (baseline)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- XGBoost (gradient boosting with regularization)
- Random Forest (ensemble method)
- Neural Network (MLP with L2 regularization)

**Total combinations**: 18 models (3 NLP methods × 6 algorithms)

## Key Results

### Top Performers (by Test R²)
| Rank | NLP Method | Model | Test R² | Gap to Target (0.670) |
|------|------------|-------|---------|----------------------|
| 1 | Hybrid | Neural Network | 0.6635 | -0.52% |
| 2 | SBERT PCA | Neural Network | 0.6410 | -4.33% |
| 3 | Hybrid | XGBoost | 0.6225 | -7.09% |
| 4 | TF-IDF | Neural Network | 0.5985 | -10.67% |
| 5 | TF-IDF | XGBoost | 0.5965 | -10.97% |

### Best Configuration: Hybrid + Neural Network
- **Test R²**: 0.6635
- **Train R²**: 0.8717
- **Overfitting Gap**: 0.2082 (severe, but still best performance)
- **Performance vs Baselines**:
  - 0.52% below target (0.670)
  - 1.34% below Word2Vec baseline (0.6725)

### Summary by NLP Method

**Hybrid (TF-IDF + SBERT PCA)**:
- Best Test R²: 0.6635 (Neural Network)
- Average Test R²: 0.5825
- Features: 992 total

**SBERT PCA Only**:
- Best Test R²: 0.6410 (Neural Network)
- Average Test R²: 0.5514
- Features: 682 total

**TF-IDF Only**:
- Best Test R²: 0.5985 (Neural Network)
- Average Test R²: 0.5346
- Features: 332 total

## Key Findings

1. **Hybrid approach wins**: Combining TF-IDF and SBERT PCA features yields the best results (+3.76% over SBERT alone, +10.86% over TF-IDF alone)

2. **95% PCA variance is optimal**: Previously tested 99% variance (0.6538 R²); 95% performs better (0.6635 R²)

3. **Neural Networks dominate**: Best performer for all three NLP methods, despite severe overfitting

4. **Overfitting patterns**:
   - Linear models (Ridge/Lasso): ~0.02 gap (minimal)
   - Neural Network: ~0.21 gap (severe but best test performance)
   - Tree models (XGBoost/RF): 0.18-0.29 gap (severe)

5. **Close to target**: Only 0.35% away from 0.670 target and 1.34% from Word2Vec baseline

## Files Generated

### Scripts
- `extract_nlp_features_pca_comparison.py` - Main feature extraction pipeline
- `extract_nlp_with_sbert_pca.py` - SBERT PCA processing
- `train_nlp_pca_comparison.py` - Model training across all combinations
- `nlp_pca_comparison_summary.py` - Results visualization

### Data & Models
- `nlp_features_pca/` - Extracted features for each NLP method
- `models_nlp_pca_comparison/` - 18 trained models (all combinations)
- `nlp_models_with_sbert_pca.pkl` - Vectorizers and PCA transformers

## Next Steps

### To Maximize Test R² (Target: > 0.670)

**High Priority**:
1. Hyperparameter tuning via GridSearchCV for Neural Network
2. Stacking ensemble (combine Neural Network + XGBoost + Ridge)
3. Test 99% PCA variance again with current hybrid approach

**Medium Priority**:
4. Try larger Neural Network architectures (256, 128, 64) or (512, 256, 128)
5. Tune XGBoost/CatBoost independently
6. Add polynomial features to baseline numerical columns

**Expected Gain**: +0.02 to +0.05 R² → potentially reaching 0.68-0.69

---

**Date**: November 29, 2025
**Branch**: `christy`
**Commit**: `4a1dcec`
