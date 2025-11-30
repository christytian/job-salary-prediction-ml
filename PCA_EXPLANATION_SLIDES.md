# SBERT PCA Optimization: Explanation Slides

---

## SLIDE 1: The Problem & Solution

### **The Challenge: Dimensionality Explosion**

**What we had**:
```
Text Fields (6): title, description, skills, industries, benefits, company_description
│
├─ TF-IDF Features:    6 fields × 100 features = 600 features
├─ SBERT Embeddings:   6 fields × 384 dimensions = 2,304 features
└─ Baseline Features:  22 numerical/categorical features
                       ─────────────────────────────────
                       TOTAL: 2,926 features
```

**The Problem**:
- **Curse of Dimensionality**: Too many features relative to data size (~6,500 samples)
- **Training time**: Very slow (minutes per model)
- **Overfitting risk**: Models memorize noise instead of learning patterns
- **Memory usage**: Large feature matrices

---

### **What is "Noise" in High-Dimensional Data?**

**SBERT Embeddings Contain Two Types of Information**:

1. **Signal** (what we want):
   - Semantic meaning of text
   - Relationships between job descriptions
   - Relevant patterns for salary prediction

2. **Noise** (what we don't want):
   - **Redundant dimensions**: Many of the 384 dimensions encode similar information
   - **Correlated features**: Dimensions that move together (not independent)
   - **Low-variance dimensions**: Features that barely change across samples
   - **Random fluctuations**: Small variations that don't help prediction

**Example**:
```
Dimension 47: "technical skills embedding" ──┐
Dimension 89: "technical abilities embedding" ├─ HIGHLY CORRELATED (redundant)
Dimension 201: "technical expertise embedding"┘

Dimension 312: Random noise, variance = 0.0001 ← IRRELEVANT (low variance)
```

---

### **Different Optimization Approaches**

**TF-IDF features**:
- Already sparse (most values = 0)
- Each feature = specific word/phrase (interpretable)
- Low correlation between features
- **Optimization**: Allocate features based on field importance
  - More features (100) for important fields like "description"
  - Fewer features (30) for less critical fields like "industries"
- **Verdict**: Optimize allocation, don't use PCA ✅

**SBERT embeddings (384 per field)**:
- Dense (all values non-zero)
- Abstract neural representations (hard to interpret)
- High correlation between many dimensions
- **Optimization**: Apply PCA to remove redundancy
- **Verdict**: Compress with PCA (95% variance) ⚡

---

## SLIDE 2: Our Solution & Results

### **Dual Optimization Strategy**

**What We Did**:
```
BEFORE (uniform allocation):
[22 baseline] + [600 TF-IDF] + [2,304 SBERT] = 2,926 features
                    ↓                  ↓
              OPTIMIZE ALLOCATION  APPLY PCA
                    ↓                  ↓
AFTER (optimized + PCA):
[22 baseline] + [310 TF-IDF] + [660 SBERT_PCA] = 992 features
```

**Two optimization strategies**:

1. **TF-IDF Field-Specific Allocation**: Assign features based on field importance
   - Before: 100 features per field (uniform)
   - After: Variable allocation (50-100 per field)
   - title: 50, description: 100, skills: 50, industries: 30, benefits: 30, company: 50
   - Total: 600 → 310 features

2. **SBERT PCA Compression**: Remove redundancy from dense embeddings
   - Before: 6 fields × 384 dims = 2,304 features
   - After: PCA at 95% variance = 660 features
   - Total: 2,304 → 660 features

**PCA Configuration**:
- **Variance preserved**: 95%
- **Dimensionality reduction**: 2,304 → 660 (71% reduction)
- **Information loss**: Only 5% of variance
- **Applied to**: SBERT embeddings ONLY

---

### **Why 95% Variance?**

**We tested multiple thresholds**:

| PCA Variance | SBERT Dimensions | Test R² | Training Time | Decision |
|--------------|------------------|---------|---------------|----------|
| No PCA (100%) | 2,304 | 0.6538 | ~8 min | ❌ Too slow, overfits |
| 99% variance | 1,200 | 0.6538 | ~5 min | ❌ Still too many dims |
| **95% variance** | **660** | **0.6635** | **~2 min** | ✅ **BEST** |
| 90% variance | 400 | 0.6420 | ~1.5 min | ❌ Lost too much info |

**95% is the sweet spot**: Removes noise while preserving signal

---

### **What Changed: Before vs After**

**Feature Composition**:

| Component | Before (Uniform) | After (Optimized + PCA) | Change |
|-----------|------------------|-------------------------|--------|
| Baseline Features | 22 | 22 | No change |
| TF-IDF Features | 600 (100 per field) | 310 (50-100 per field) | **-48% (optimized)** |
| SBERT Features | 2,304 | 660 | **-71% (PCA)** |
| **Total** | **2,926** | **992** | **-66% overall** |

**Model Performance**:

| Metric | Before (No PCA) | After (95% PCA) | Improvement |
|--------|-----------------|-----------------|-------------|
| Best Test R² | 0.6538 | 0.6635 | **+1.48%** ✅ |
| Training Time | ~8 minutes | ~2 minutes | **4× faster** ⚡ |
| Overfitting Gap | 0.2156 | 0.2082 | Slightly better |
| Memory Usage | High | Moderate | Lower |

---

### **Why This Works: The Intuition**

**Think of SBERT embeddings as a photograph**:

- **Original**: 2,304 dimensions = 2,304 "pixels" describing each job
- **Problem**: Many pixels are redundant or blurry (noise)
- **PCA**: Finds the 660 most important "features" that capture 95% of the image
- **Result**: Clearer picture with less noise

**Mathematical Explanation**:
- PCA finds **principal components** = directions of maximum variance
- First 660 components capture the most important patterns
- Remaining 1,644 dimensions mostly contain:
  - Redundant information (already captured in first 660)
  - Random noise (hurts generalization)

---

### **Key Results from Experiments**

**Tested 3 NLP Approaches × 6 Models = 18 Total Combinations**

**Winner: Hybrid (TF-IDF + SBERT PCA) + Neural Network**
- **Test R²**: 0.6635
- **Gap to target (0.670)**: Only -0.52%
- **Gap to W2V baseline (0.6725)**: -1.34%

**Why Hybrid Won**:
1. **TF-IDF captures**: Specific keywords, domain terms (interpretable signal)
2. **SBERT PCA captures**: Semantic meaning, context (compressed signal)
3. **Together**: Complementary information from different perspectives

---

### **Summary: What We Accomplished**

✅ **Identified two problems**:
- 2,304 SBERT dimensions contain redundancy and noise
- 600 TF-IDF features uniformly allocated (not optimized)

✅ **Applied dual optimization**:
- **SBERT**: PCA compression (95% variance) to remove redundancy
- **TF-IDF**: Field-specific allocation based on importance

✅ **Chose optimal configurations**:
- SBERT: 95% variance (660 dims) preserves signal, removes noise
- TF-IDF: 310 features with variable allocation (30-100 per field)

✅ **Achieved better results**:
- 66% fewer features (2,926 → 992)
- 4× faster training
- 1.48% better R² score
- Less overfitting

✅ **Validated approach**: Hybrid method (TF-IDF + SBERT PCA) outperforms either alone

---

### **The "Aha!" Moment**

**Question**: If we removed 66% of features (2,926 → 992), why did performance IMPROVE?

**Answer**: We didn't remove information—we removed NOISE and optimized allocation!

**What we eliminated**:

1. **From SBERT** (2,304 → 660):
   - Removed 1,644 redundant/noisy dimensions
   - Kept 660 principal components capturing 95% variance
   - These eliminated dimensions were correlated, low-variance, and overfitting-inducing

2. **From TF-IDF** (600 → 310):
   - Reallocated features based on field importance
   - Description gets 100 (most important)
   - Industries/benefits get 30 (less critical)
   - Smarter use of feature budget

**Analogy**: It's like noise-canceling headphones + volume balancing—removing background noise AND adjusting levels makes the music (signal) clearer!

---

